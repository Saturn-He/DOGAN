# DOGAN

Training, inference and image-quality evaluation code for [DOGAN: DINO-Based Optical-Prior-Driven GAN for SAR-to-Optical Image Translation](https://doi.org/10.1109/TGRS.2025.3606979), IEEE TGRS, 2025.

## Setup

Use Python 3.10. Install PyTorch 2.1 and torchvision 0.16 for your CUDA version, then run:

```bash
pip install -r requirements.txt
```

Download the [DINO ViT-S/16 weights](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth) to `pretrained_dino/dino_deitsmall16_pretrain.pth` for training. The training code can download the torchvision ResNet-50 weights on first use. For inference supply a trained generator checkpoint under `checkpoints/DOGAN/200_net_G_trans.pth`, or use the checkpoint produced by the training command below.

## Data and commands

Provide your own paired SAR and optical images. Put SAR images in `trainA/` and `testA/`, and optical targets in `trainB/` and `testB/`. Paired images must have matching relative filenames.

```bash
python scripts/run_dogan.py --mode train --dataroot /path/to/paired_dataset
python scripts/run_dogan.py --mode test --dataroot /path/to/paired_dataset
```

Run from the repository root. By default, training uses 256-pixel crops and testing processes all pairs in the supplied test folders. Use --num_test to set an optional test limit. To use the 512-pixel preprocessing profile, pass --config configs/crop512.json to both commands and use the corresponding DOGAN_512 checkpoint and output folders.

## Evaluation

Inference writes images under `results/DOGAN/test_200/images/`. Run the six paper image-quality metrics separately:

```bash
python evaluate.py --gt /path/to/paired_dataset/testB \
  --pred results/DOGAN/test_200/images \
  --output results/DOGAN/metrics.json
```

The evaluator matches GT and predictions by stem after removing `_fake_B` from the prediction name. Set `--pred-suffix ""` if the stems already match. Use `--check-pairs` to check alignment. By default it evaluates at the native resolution. Set `--resize WIDTH HEIGHT` for image pairs requiring common dimensions.

The evaluator reports PSNR, SSIM, CW-SSIM, RMSE, LPIPS and SAM. PSNR defaults to the legacy PIL luminance convention. Select `--psnr-color rgb` to measure RGB PSNR. SSIM and RMSE operate on RGB at an intensity range of 0–255. CW-SSIM uses pyssim 0.7. LPIPS uses AlexNet with spatial averaging, and SAM uses the legacy image-similarity-measures implementation. Resizing applies to all six metrics when requested. Scores on historical splits depend on the exact files, preprocessing and checkpoint used.

## Acknowledgments

The implementation builds on [Parallel-GAN](https://doi.org/10.1109/TGRS.2022.3200996), [CycleGAN and pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and [DINOv2](https://github.com/facebookresearch/dinov2). See the upstream projects for their terms and notices.
