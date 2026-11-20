import torch
from .base_model import BaseModel
from . import networks
from util import mmd_loss

import torch.nn as nn



class ParallelGANModel(BaseModel):
    """ This class implements the Parallel_GAN model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG para_GAN' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    Parallel GAN paper: https://ieeexplore.ieee.org/document/9864654
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For Parallel GAN, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')    # 从100改成10，看看效果

        return parser

    def __init__(self, opt):
        """Initialize the Parallel GAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.visual_names = ['fake_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if opt.net == 'translation':
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'feature', 'G_L1', 'MMD']
            if self.isTrain:
                self.recon_net = networks.define_recon_net(opt.dino_checkpoint).to(device=self.device)
                # Alias the trainable adapters for checkpointing; DINO stays frozen.
                self.netAdapter = self.recon_net.up_blocks
                self.model_names.append('Adapter')
                if self.gpu_ids:
                    self.recon_net = torch.nn.DataParallel(self.recon_net, device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0])
        elif opt.net == 'reconstruction':
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'recon_VGG', 'G_L1']

        # weight init
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                if opt.weight_init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif opt.weight_init_type == 'orth':
                    nn.init.orthogonal_(m.weight.data)
                elif opt.weight_init_type == 'xavier_uniform':
                    nn.init.xavier_uniform(m.weight.data, 1.)
                else:
                    raise NotImplementedError('{} unknown inital type'.format(opt.weight_init_type))
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.net)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            if opt.net == 'reconstruction':
                self.criterionVGG = networks.VGGLoss()
            self.criterionMMD = mmd_loss.MMD_loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if opt.net == 'translation':
                self.optimizer_Adapter = torch.optim.Adam(
                    self.netAdapter.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_Adapter)
            self.netD = torch.nn.DataParallel(self.netD, device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0])

        self.netG = torch.nn.DataParallel(self.netG, device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0])


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def normalize_tensor(self, x, norm_type='batch', eps=1e-5):
        """
        对单个 tensor 进行归一化.

        参数:
            x (Tensor): shape 为 [B, C, H, W] 的特征 tensor.
            norm_type (str): 'batch', 'layer' 或 'l2'.
            eps (float): 防止除 0 的小常数.
        返回:
            Tensor: 归一化后的 tensor.
        """
        if norm_type == 'batch':
            # 批归一化：对每个通道计算均值和标准差，计算维度：B, H, W
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True) + eps
            return (x - mean) / std
        elif norm_type == 'layer':
            # 层归一化：对每个样本计算均值和标准差，计算维度：C, H, W
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            std = x.std(dim=[1, 2, 3], keepdim=True) + eps
            return (x - mean) / std
        elif norm_type == 'l2':
            # L2 归一化：对每个样本在所有维度上计算 L2 范数
            norm = x.norm(p=2, dim=[1, 2, 3], keepdim=True) + eps
            return x / norm
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def normalize_features(self, CLIP_fm, trans_fm, norm_type='batch', eps=1e-5):
        """
        对 CLIP_fm 和 trans_fm 两组特征进行归一化.

        参数:
            CLIP_fm (list of Tensor): 包含 5 个特征 tensor 的列表.
            trans_fm (list of Tensor): 包含 5 个特征 tensor 的列表.
            norm_type (str): 归一化类型，可以是 'batch', 'layer', 或 'l2'.
            eps (float): 防止除 0 的小常数.
        返回:
            tuple: (CLIP_fm_nm, trans_fm_nm)，每个都是归一化后的特征列表.
        """
        CLIP_fm_nm = []
        trans_fm_nm = []

        for clip_feat, trans_feat in zip(CLIP_fm, trans_fm):
            clip_norm = self.normalize_tensor(clip_feat, norm_type, eps)
            trans_norm = self.normalize_tensor(trans_feat, norm_type, eps)
            CLIP_fm_nm.append(clip_norm)
            trans_fm_nm.append(trans_norm)

        return CLIP_fm_nm, trans_fm_nm

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.net == 'reconstruction':
            self.fake_B, self.trans_feature = self.netG(self.real_B)
        elif self.opt.net == 'translation':
            self.trans_feature = self.netG(self.real_A)
            self.fake_B = self.trans_feature[-1]
            self.trans_feature_selected_5 = [self.trans_feature[i] for i in [0, 1, 2, 3, 5]]  # CLIP
            if self.isTrain:
                recon_features = self.recon_net(self.real_B)
                self.recon_feature = [recon_features[i] for i in [4, 3, 2, 1, 0]]

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward(retain_graph=True)  # 保留计算图，避免 G 反向传播时报错

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_feature = 0
        self.loss_MMD = 0
        self.weight = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        if self.opt.net == 'translation':
            for i in range(len(self.trans_feature_selected_5)):
                self.loss_feature += self.criterionL1(self.trans_feature_selected_5[i], self.recon_feature[i]) * self.weight[i-1] * 10 # 10
                self.loss_MMD += self.criterionMMD( self.recon_feature[i], self.trans_feature_selected_5[i]) * 0.1
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_feature + self.loss_MMD
        elif self.opt.net == 'reconstruction':
            self.loss_recon_VGG = self.criterionVGG(self.fake_B, self.real_B) * 10
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_recon_VGG
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        if self.opt.net == 'translation':
            self.optimizer_Adapter.zero_grad(set_to_none=True)
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
        if self.opt.net == 'translation':
            # The existing feature and MMD losses also train the adapters.
            self.optimizer_Adapter.step()
