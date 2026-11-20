import torch
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PatchUnembedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.unproj = nn.ConvTranspose2d(embed_dim, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, C = x.shape
        h = w = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, h, w)
        x = self.unproj(x)
        return x

class SymmetricViT(nn.Module):
    def __init__(self, opt, img_size=256, patch_size=16, embed_dim=768, depth=6, num_heads=12):
        super().__init__()
        self.encoder_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        mlp_dim = embed_dim * 4
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                hidden_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=0.1,
                attention_dropout=0.1
            )
            for _ in range(depth)
        ])

        self.decoder_blocks = nn.ModuleList([
            EncoderBlock(
                hidden_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=0.1,
                attention_dropout=0.1
            )
            for _ in range(depth)
        ])

        self.unembed = PatchUnembedding(img_size, patch_size, embed_dim)

    def forward(self, x):
        enc_features = []
        dec_features = []

        x = self.encoder_embed(x)

        # Encoder
        for blk in self.encoder_blocks:
            x = blk(x)
            enc_features.append(x)

        # Bottleneck
        x = self.decoder_embed(x)

        # Decoder
        for blk in self.decoder_blocks:
            dec_features.append(x)
            x = blk(x)

        # Reconstruct output image
        out_img = self.unembed(x)

        # Convert decoder features into spatial maps
        dec_feature_maps = []
        for feat in dec_features:
            B, N, C = feat.shape
            h = w = int(N ** 0.5)
            fmap = feat.transpose(1, 2).reshape(B, C, h, w)
            dec_feature_maps.append(fmap)

        return out_img, dec_feature_maps


# 测试模型结构
if __name__ == '__main__':
    model = SymmetricViT()
    inp = torch.randn(1, 3, 256, 256)
    out_img, feature_maps = model(inp)

    print(f'Output image shape: {out_img.shape}')
    for idx, fmap in enumerate(feature_maps):
        print(f'Decoder feature map {idx + 1}: {fmap.shape}')