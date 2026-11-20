from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models
import clip  # pip install git+https://github.com/openai/CLIP.git
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import timm
import torch.nn.functional as F
from torchvision.models import resnet50
from dinov2.models.vision_transformer import vit_base, vit_large, vit_small
from functools import partial
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], netset='transaltion'):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netset == 'translation':
        if netG == 'para_gan':
            net = Para_GAN(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
        return init_net(net, init_type, init_gain, gpu_ids)
    elif netset == 'reconstruction':
        net = Resrecon()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
        return net


def define_recon_net(pretrained_path=None):
    net = None
    net = DINO_ViT_S16_FeatureExtractor(pretrained_path=pretrained_path)
    return net


class Resrecon(nn.Module):

    def __init__(self):
        super(Resrecon, self).__init__()
        use_bias = False
        norm_layer = nn.BatchNorm2d
        model_resnet = models.resnet50(pretrained=True)
        self.encoder = list(model_resnet.children())[:-2]
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(1024),
                nn.ReLU(False),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(1024),
                nn.ReLU(False),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn. Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(512),
            nn.ReLU(False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.Tanh())
        ])

    def forward(self, input):
        out_feature = []
        decoder_feature = self.encoder(input)
        out_feature.append(decoder_feature) #
        for decoder_blk in self.decoder:
            decoder_feature = decoder_blk(decoder_feature)
            out_feature.append(decoder_feature)
        return out_feature

class CLIP_res_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的 CLIP RN50 模型
        self.clip_model, _ = clip.load("RN50")

        # 获取视觉模型部分
        self.visual = self.clip_model.visual
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # 提取主干结构
        self.stem = nn.Sequential(
            self.visual.conv1,
            self.visual.bn1,
            self.visual.relu1,
            self.visual.conv2,
            self.visual.bn2,
            self.visual.relu2,
            self.visual.conv3,
            self.visual.bn3,
            self.visual.relu3,
            self.visual.avgpool
        )
        self.layer1 = self.visual.layer1
        self.layer2 = self.visual.layer2
        self.layer3 = self.visual.layer3
        self.layer4 = self.visual.layer4

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([64, 128, 128]),
                ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([256, 64, 64]),
                ),
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([512, 32, 32]),
                ),
            nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([1024, 16, 16]),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, stride=1),
                nn.LayerNorm([2048, 8, 8]),
            )
            ])
    def forward(self, x):
        feature_maps = []

        # 每层依次处理并保存特征图
        x = x.half()
        x = self.stem(x)
        feature_maps.append(x)

        x = self.layer1(x)
        feature_maps.append(x)

        x = self.layer2(x)
        feature_maps.append(x)

        x = self.layer3(x)
        feature_maps.append(x)

        x = self.layer4(x)
        feature_maps.append(x)
        outcome = []
        for feature, adapter in zip(feature_maps, self.adapters):
            feature = feature.float()
            outcome.append(adapter(feature))
        return outcome

class CLIP_ViT_B16_FeatureExtractor(nn.Module):
    def __init__(self, model_name='ViT-B/16', device='cuda', img_size=256):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.visual = self.model.visual
        self.patch_size = self.visual.conv1.kernel_size[0]
        self.embed_dim = self.visual.transformer.width

        self.selected_layers = [0, 2, 5, 7, 10]  # even-numbered layers

        self.up_blocks = nn.ModuleList([
            # Gradual upsampling for 64x128x128
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 64, 3, padding=1),
                nn.LayerNorm([64, 32, 32]),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LayerNorm([64, 64, 64]),
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LayerNorm([64, 128, 128])
            ),
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 32, 32]),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(256, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 64, 64])),
            nn.Sequential(nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.LayerNorm([512, 32, 32])),
            nn.Sequential(nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 1024, 3, padding=1), nn.LayerNorm([1024, 16, 16])),
            nn.Sequential(nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 2048, 3, padding=1), nn.LayerNorm([2048, 8, 8]))
        ])
    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x = x.half()
        x = self.visual.conv1(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)

        cls_tokens = self.visual.class_embedding.expand(b, 1, -1).to(x.device)
        pos_embedding = self.visual.positional_embedding.to(x.device)

        x = torch.cat([cls_tokens, x], dim=1) + pos_embedding
        x = self.visual.ln_pre(x)

        intermediate_features = []
        for idx, layer in enumerate(self.visual.transformer.resblocks):
            x = x.half()
            x = layer(x)
            if idx in self.selected_layers:
                fmap = x[:, 1:, :].permute(0, 2, 1).reshape(b, self.embed_dim, 14, 14)
                intermediate_features.append(fmap.float())

        aligned_features = []
        for fmap, up_block in zip(intermediate_features, self.up_blocks):
            aligned_fmap = up_block(fmap)
            aligned_features.append(aligned_fmap)

        return aligned_features  # sizes: (bs,64,128,128), (bs,128,128,128), ..., (bs,2048,8,8)

class CLIP_L14_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _ = clip.load("ViT-L/14")
        self.visual = self.model.visual
        self.transformer = self.visual.transformer
        self.embed_dim = 1024
        for param in self.model.parameters():
            param.requires_grad = False
        self.up_blocks = nn.ModuleList([
            # Gradual upsampling for 64x128x128
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 64, 3, padding=1),
                nn.LayerNorm([64, 32, 32]),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LayerNorm([64, 64, 64]),
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LayerNorm([64, 128, 128])
            ),
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 32, 32]),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(256, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 64, 64])),
            nn.Sequential(nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.LayerNorm([512, 32, 32])),
            nn.Sequential(nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 1024, 3, padding=1), nn.LayerNorm([1024, 16, 16])),
            nn.Sequential(nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 2048, 3, padding=1), nn.LayerNorm([2048, 8, 8]))
        ])

    def forward(self, x):
        # 强制将输入 resize 到 224x224（和 CLIP 的预训练设置一致）
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x = x.half()
        # Patch embedding：conv1 的输出 shape [B, C, 14, 14]
        x = self.visual.conv1(x)  # => [B, 1024, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 1024, 196]
        x = x.permute(0, 2, 1)  # [B, 196, 1024]

        # 拼接 CLS token
        cls_token = self.visual.class_embedding.to(x.dtype)
        cls_token = cls_token.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B, 1, 1024]
        x = torch.cat([cls_token, x], dim=1)  # [B, 197, 1024]

        # 加上位置编码
        x = x + self.visual.positional_embedding.to(x.dtype)  # [B, 197, 1024]
        x = self.visual.ln_pre(x)

        # 提取每一层的 transformer 中间特征，并 reshape 为 2D 特征图
        feat_maps_2d = []
        for blk in self.transformer.resblocks:
            x = blk(x)  # [B, 197, 1024]
            patch_tokens = x[:, 1:, :]  # 去掉 CLS，得到 [B, 196, 1024]
            B, N, C = patch_tokens.shape
            fmap_2d = patch_tokens.permute(0, 2, 1).reshape(B, C, 16, 16)  # → [B, 1024, 16, 16]
            feat_maps_2d.append(fmap_2d)
        feature_selected = [feat_maps_2d[i] for i in [0, 5, 10, 15, 20]]
        outcome = []
        for feature, up_blk in zip(feature_selected, self.up_blocks):
            outcome.append(up_blk(feature.float()))
        return outcome

class Adapter(nn.Module):
    def __init__(self, in_channels, bottleneck_channels):
        super(Adapter, self).__init__()
        self.down = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1)

    def forward(self, x):
        return x + self.up(self.relu(self.down(x)))

class DINO_res_FeatureExtractor(nn.Module):
    def __init__(self, pretrained_path=None, freeze=True):
        super().__init__()
        # 构建 ResNet50 主干
        backbone = resnet50(pretrained=False)

        # Apply separate stem layers to match pretrained checkpoint keys.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        print(backbone)
        # Load the pretrained checkpoint.
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cuda")
            msg = self.load_state_dict(state_dict, strict=False)
            print("✅ DINO ResNet weights loaded:\n", msg)

        # Freeze backbone parameters.
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            print("✅ Parameters frozen.")

        # Upsample backbone features.
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1, stride=1),
                nn.LayerNorm([64, 128, 128])
            ),
            nn.Sequential(
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(256, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 64, 64])
            ),
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(512, 512, 3, padding=1, stride=1),
                nn.LayerNorm([512, 32, 32])
            ),
            nn.Sequential(
                nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
                nn.Conv2d(1024, 1024, 3, padding=1, stride=1),
                nn.LayerNorm([1024, 16, 16])
            ),
            nn.Sequential(
                nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
                nn.Conv2d(2048, 2048, 3, padding=1, stride=1),
                nn.LayerNorm([2048, 8, 8])
            )
        ])

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        feat_maps = []

        # Apply separate stem layers to match pretrained checkpoint keys.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat_maps.append(x)  # [B, 64, 56, 56]

        x = self.layer1(x)
        feat_maps.append(x)  # [B, 256, 56, 56]

        x = self.layer2(x)
        feat_maps.append(x)  # [B, 512, 28, 28]

        x = self.layer3(x)
        feat_maps.append(x)  # [B, 1024, 14, 14]

        x = self.layer4(x)
        feat_maps.append(x)  # [B, 2048, 7, 7]

        outcome = []
        for feature, up_blk in zip(feat_maps, self.up_blocks):
            outcome.append(up_blk(feature))
        return outcome

class DINO_ViT_S16_FeatureExtractor(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # Resolve the active DINO checkpoint independently of the working directory.
        if pretrained_path is None:
            pretrained_path = Path(__file__).resolve().parents[1] / 'pretrained_dino' / 'dino_deitsmall16_pretrain.pth'
        pretrained_path = Path(pretrained_path).expanduser()
        if not pretrained_path.is_file():
            raise FileNotFoundError(f'DINO ViT-S/16 checkpoint not found: {pretrained_path}')
        self.model = timm.create_model("vit_small_patch16_224", pretrained=False)
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        msg = self.model.load_state_dict(state_dict, strict=False)
        # Official DINO weights omit the unused ImageNet classification head.
        missing = set(msg.missing_keys) - {'head.weight', 'head.bias'}
        if missing or msg.unexpected_keys:
            raise RuntimeError(f'Incompatible DINO backbone checkpoint: {msg}')
        print(f'Loaded DINO checkpoint: {pretrained_path}\n{msg}')
        for param in self.model.parameters():
            param.requires_grad = False

        # 拿出结构部件
        self.patch_embed = self.model.patch_embed
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.pos_drop = self.model.pos_drop
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        self.embed_dim = 384

        self.up_blocks = nn.ModuleList([
            # Gradual upsampling for 64x128x128
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 64, 3, padding=1),
                nn.LayerNorm([64, 32, 32]),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LayerNorm([64, 64, 64]),
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.LayerNorm([64, 128, 128])
            ),
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 32, 32]),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(256, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 64, 64])),
            nn.Sequential(nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.LayerNorm([512, 32, 32])),
            nn.Sequential(nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 1024, 3, padding=1), nn.LayerNorm([1024, 16, 16])),
            nn.Sequential(nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 2048, 3, padding=1), nn.LayerNorm([2048, 8, 8]))
        ])

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

        x = self.patch_embed(x)  # [B, 196, 384]
        B, N, C = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 384]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 384]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x[:, 1:, :].permute(0, 2, 1).reshape(B, C, 14, 14))
        feats_selected = [feats[i] for i in [0, 2, 5, 7, 10]]
        outcome = []
        for feature, up_blk in zip(feats_selected, self.up_blocks):
            outcome.append(up_blk(feature.float()))
        return outcome

class DINO_ViT_B16_FeatureExtractor(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # 用 timm 构造 ViT-B/16 结构
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            # DINO 权重是直接保存的 backbone，需要做一点 key 映射（或只取 matching keys）
            msg = self.model.load_state_dict(state_dict, strict=False)
            print("Loaded checkpoint:", msg)
        for param in self.model.parameters():
            param.requires_grad = False
        # 拿出结构部件
        self.patch_embed = self.model.patch_embed
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.pos_drop = self.model.pos_drop
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        self.embed_dim = 768
        self.up_blocks = nn.ModuleList([
            # Gradual upsampling for 64x128x128
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 64, 3, padding=1, stride=1),
                nn.LayerNorm([64, 32, 32]),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1, stride=1),
                nn.LayerNorm([64, 64, 64]),
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1, stride=1),
                nn.LayerNorm([64, 128, 128])
            ),
            nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 32, 32]),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(256, 256, 3, padding=1, stride=1),
                nn.LayerNorm([256, 64, 64])),
            nn.Sequential(nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.LayerNorm([512, 32, 32])),
            nn.Sequential(nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 1024, 3, padding=1), nn.LayerNorm([1024, 16, 16])),
            nn.Sequential(nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 2048, 3, padding=1), nn.LayerNorm([2048, 8, 8]))
        ])

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)

        x = self.patch_embed(x)  # [B, 196, 768]
        B, N, C = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x[:, 1:, :].permute(0, 2, 1).reshape(B, C, 14, 14))
        feats_selected = [feats[i] for i in [0, 2, 5, 7, 10]]
        outcome = []
        for feature, up_blk in zip(feats_selected, self.up_blocks):
            outcome.append(up_blk(feature.float()))
        return outcome

def remap_flat_to_nested(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if parts[1].isdigit():
                # e.g. blocks.1.norm1.weight → blocks.0.1.norm1.weight
                new_key = f"blocks.0.{parts[1]}." + ".".join(parts[2:])
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict

class DINOv2_ViT_S14_Extractor(nn.Module):
    def __init__(self, pretrained_path=None, freeze=True):
        super().__init__()
        self.model = vit_small(num_register_tokens=0, img_size=518, patch_size=14, init_values=0.1)
        state_dict = torch.load(pretrained_path, map_location='cuda')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        state_dict = remap_flat_to_nested(state_dict)
        msg = self.model.load_state_dict(state_dict, strict=False)
        print('load pretrained model:\n', msg)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.embed_dim = 384
        self.up_blocks = nn.ModuleList([
            # Gradual upsampling for 64x128x128
            nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 64, 3, padding=1), nn.LayerNorm([64, 128, 128])
            ),
            # Gradual upsampling for 128x128x128
            nn.Sequential(nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 256, 3, padding=1), nn.LayerNorm([256, 64, 64])),
            nn.Sequential(nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.LayerNorm([512, 32, 32])),
            nn.Sequential(nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 1024, 3, padding=1), nn.LayerNorm([1024, 16, 16])),
            nn.Sequential(nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 2048, 3, padding=1), nn.LayerNorm([2048, 8, 8]))
        ])

    def forward(self, x):
        # 自动 resize 到 518×518
        B = x.shape[0]
        x = F.interpolate(x, size=(518, 518), mode="bicubic", align_corners=False)

        # Patch embedding
        x = self.model.patch_embed(x)  # [B, 768, H, W]
        x = x.flatten(2)  # [B, N=1369, 768]

        # Token 拼接
        cls_token = self.model.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1374, 768]
        pos_embed = self.model.pos_embed
        # 加位置编码
        x = x + pos_embed  # [B, 1374, 768]

        # Transformer blocks 输出
        features = []
        for blk in self.model.blocks[0]:  # BlockChunk 解包
            x = blk(x)
            features.append(x.clone())

        outcome = []
        feats_selected = [features[i][:, 1:, :].permute(0, 2, 1).reshape((x.shape[0], x.shape[2], 37, 37)) for i in
                          [0, 2, 5, 7, 10]]
        for feature, up_blk in zip(feats_selected, self.up_blocks):
            outcome.append(up_blk(feature))
        return outcome

class DINOv2_ViT_B14_Extractor(nn.Module):
    def __init__(self, pretrained_path=None, img_size=518, freeze=True):
        super().__init__()

        # Build the default DINOv2 ViT-B/14 architecture.
        self.model = vit_base(num_register_tokens=0, patch_size=14, img_size=img_size, init_values=0.1)

        # Load the pretrained checkpoint.
        if pretrained_path is not None:
            print(f"Loading checkpoint from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cuda")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = remap_flat_to_nested(state_dict)
            msg = self.model.load_state_dict(state_dict, strict=False)
            print("Loaded weights:\n", msg)
        self.embed_dim = 768
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.up_blocks = nn.ModuleList([
            # Gradual upsampling for 64x128x128
            nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 64, 3, padding=1), nn.LayerNorm([64, 128, 128])
            ),
            # Gradual upsampling for 128x128x128
            nn.Sequential(nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 256, 3, padding=1), nn.LayerNorm([256, 64, 64])),
            nn.Sequential(nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.LayerNorm([512, 32, 32])),
            nn.Sequential(nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 1024, 3, padding=1), nn.LayerNorm([1024, 16, 16])),
            nn.Sequential(nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 2048, 3, padding=1), nn.LayerNorm([2048, 8, 8]))
        ])

    def forward(self, x):
        # 自动 resize 到 518×518
        B = x.shape[0]
        x = F.interpolate(x, size=(518, 518), mode="bicubic", align_corners=False)

        # Patch embedding
        x = self.model.patch_embed(x)  # [B, 768, H, W]
        x = x.flatten(2)  # [B, N=1369, 768]

        # Token 拼接
        cls_token = self.model.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1374, 768]
        pos_embed = self.model.pos_embed
        # 加位置编码
        x = x + pos_embed # [B, 1374, 768]

        # Transformer blocks 输出
        features = []
        for blk in self.model.blocks[0]:  # BlockChunk 解包
            x = blk(x)
            features.append(x.clone())

        outcome = []
        feats_selected = [features[i][:, 1:, :].permute(0, 2, 1).reshape((x.shape[0], x.shape[2], 37, 37)) for i in [0, 2, 5, 7, 10]]
        for feature, up_blk in zip(feats_selected, self.up_blocks):
            outcome.append(up_blk(feature))
        return outcome

class DINOv2_ViT_L14_Extractor(nn.Module):
    def __init__(self, pretrained_path=None, img_size=518, freeze=True):
        super().__init__()

        self.model = vit_large(num_register_tokens=0, patch_size=14, img_size=img_size, init_values=0.1)

        # Load the pretrained checkpoint.
        if pretrained_path is not None:
            print(f"Loading checkpoint from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cuda")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            state_dict = remap_flat_to_nested(state_dict)
            msg = self.model.load_state_dict(state_dict, strict=False)
            print("Loaded weights:\n", msg)
        self.embed_dim = 1024
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.up_blocks = nn.ModuleList([
            # Gradual upsampling for 64x128x128
            nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                nn.Conv2d(self.embed_dim, 64, 3, padding=1), nn.LayerNorm([64, 128, 128])
            ),
            # Gradual upsampling for 128x128x128
            nn.Sequential(nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 256, 3, padding=1), nn.LayerNorm([256, 64, 64])),
            nn.Sequential(nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 512, 3, padding=1), nn.LayerNorm([512, 32, 32])),
            nn.Sequential(nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 1024, 3, padding=1), nn.LayerNorm([1024, 16, 16])),
            nn.Sequential(nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
                          nn.Conv2d(self.embed_dim, 2048, 3, padding=1), nn.LayerNorm([2048, 8, 8]))
        ])

    def forward(self, x):
        # 自动 resize 到 518×518
        B = x.shape[0]
        x = F.interpolate(x, size=(518, 518), mode="bicubic", align_corners=False)

        # Patch embedding
        x = self.model.patch_embed(x)  # [B, 768, H, W]
        x = x.flatten(2)  # [B, N=1369, 768]

        # Token 拼接
        cls_token = self.model.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls_token, x], dim=1)  # [B, 1374, 768]
        pos_embed = self.model.pos_embed
        # 加位置编码
        x = x + pos_embed  # [B, 1374, 768]

        # Transformer blocks 输出
        features = []
        for blk in self.model.blocks[0]:  # BlockChunk 解包
            x = blk(x)
            features.append(x.clone())

        outcome = []
        feats_selected = [features[i][:, 1:, :].permute(0, 2, 1).reshape((x.shape[0], x.shape[2], 37, 37)) for i in
                          [0, 5, 10, 15, 20]]
        for feature, up_blk in zip(feats_selected, self.up_blocks):
            outcome.append(up_blk(feature))
        return outcome

class Para_GAN(nn.Module):
    def __init__(self, input_nc, output_nc, channel=2048, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(Para_GAN, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        resnet_part = []

        # define the model's encoder by the resnet_50
        model_resnet = models.resnet50(pretrained=True)
        encoder = list(model_resnet.children())[:-2]
        self.encoder = nn.Sequential(*encoder)

        # the input channels of pretrained model(resnet) is 3, when come to 4-channel SAR of SpaceNet,
        # we copy 0th channel parameters into the 4th channel. And this code only use for 4-channel dataset.
        # for 3 channel dataset, comment it directly.
        if input_nc==4:
            with torch.no_grad():
                pretrain_weights = self.encoder[0].weight.clone()
                self.encoder[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.encoder[0].weight[:, :3] = pretrain_weights
                self.encoder[0].weight[:, -2:] = pretrain_weights[:, :1]

        # add resnet blocks between the encoder and the decoder
        for i in range(6):
            resnet_part += [
                ResnetBlock(channel, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        self.resnet_part = nn.Sequential(*resnet_part)

        # add the decoder to output the feature and final img
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(1024),
                nn.ReLU(False),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(1024),
                nn.ReLU(False),
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),
                          norm_layer(512),
                          nn.ReLU(False),
                          nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
                          norm_layer(256),
                          nn.ReLU(False),
                          nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(64),
                          nn.ReLU(False),
                          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias)),

            nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=use_bias),
                          nn.Tanh())
        ])

    def forward(self, input):
        encoder_feature = []
        out_feature = []
        feature = input
        for block in self.encoder:
            feature = block(feature)
            encoder_feature.append(feature)
        decoder_feature = self.resnet_part(encoder_feature[-1])
        out_feature.append(decoder_feature)  #
        i = 0
        for decoder_blk in self.decoder:
            decoder_feature = decoder_blk(decoder_feature)
            out_feature.append(decoder_feature)
            if i < 3:
                decoder_feature = torch.cat([decoder_feature, encoder_feature[6 - i]], 1)
            i += 1
            if i == 5:
                break
        out_feature.append(self.decoder[5](decoder_feature))
        return out_feature


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier .It can classify whether 70×70
        overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg().cuda()
        self.criterion1 = nn.L1Loss()
        self.weights = [1, 1, 1, 1, 1]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion1(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
