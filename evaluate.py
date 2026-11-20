"""Evaluate the six image-quality metrics reported in the DOGAN paper."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

METRICS = ("PSNR", "SSIM", "CW-SSIM", "RMSE", "LPIPS", "SAM")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def index_images(directory, suffix):
    images = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if not path.stem.endswith(suffix):
            continue
        key = path.stem.removesuffix(suffix)
        if key in images:
            raise ValueError(f"Duplicate image ID {key!r} in {directory}")
        images[key] = path
    if not images:
        raise ValueError(f"No matching images in {directory}")
    return images


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def pixel_psnr(gt, pred, color):
    if color == "luma":
        gt, pred = gt.convert("L"), pred.convert("L")
    delta = np.asarray(gt, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    error = float(np.mean(delta ** 2))
    return 100.0 if error == 0 else 10.0 * math.log10(255.0 ** 2 / error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--pred-suffix", default="_fake_B",
                        help="Prediction stem suffix. Pass an empty string for exact stem matching.")
    parser.add_argument("--resize", nargs=2, metavar=("WIDTH", "HEIGHT"), type=int)
    parser.add_argument("--psnr-color", choices=("luma", "rgb"), default="luma")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-pairs", action="store_true")
    args = parser.parse_args()

    gt_files = index_images(args.gt, "")
    pred_files = index_images(args.pred, args.pred_suffix)
    missing = sorted(gt_files.keys() - pred_files.keys())
    extra = sorted(pred_files.keys() - gt_files.keys())
    if missing or extra:
        parser.error(f"Unmatched IDs: {len(missing)} missing predictions {missing[:5]}; "
                     f"{len(extra)} extra predictions {extra[:5]}")
    if args.check_pairs:
        print(f"{len(gt_files)} matched pairs")
        return

    import lpips
    import ssim
    import torch
    from image_similarity_measures.quality_metrics import sam
    from skimage.metrics import structural_similarity

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    perceptual = lpips.LPIPS(net="alex", spatial=True).to(device).eval()

    def lpips_tensor(image):
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    rows = []
    for key, gt_path in gt_files.items():
        gt = load_rgb(gt_path)
        pred = load_rgb(pred_files[key])
        if args.resize:
            size = tuple(args.resize)
            gt = gt.resize(size, Image.Resampling.BICUBIC)
            pred = pred.resize(size, Image.Resampling.BICUBIC)
        if gt.size != pred.size or min(gt.size) < 32:
            raise ValueError(f"{key}: expected matching image sizes of at least 32 pixels per side, "
                             f"got {gt.size} and {pred.size}. Use --resize if needed.")
        ga, pa = np.asarray(gt), np.asarray(pred)
        with torch.inference_mode():
            perceptual_score = float(perceptual(lpips_tensor(gt), lpips_tensor(pred)).mean().cpu())
        rows.append((
            pixel_psnr(gt, pred, args.psnr_color),
            float(structural_similarity(ga, pa, channel_axis=2, data_range=255)),
            float(ssim.SSIM(gt).cw_ssim_value(pred)),
            float(np.sqrt(np.mean((ga.astype(np.float64) - pa.astype(np.float64)) ** 2))),
            perceptual_score,
            float(sam(ga.astype(np.uint32), pa.astype(np.uint32))),
        ))

    scores = np.asarray(rows)
    summary = {"pairs": len(rows), "gt": str(args.gt.resolve()),
               "pred": str(args.pred.resolve()), "resize": args.resize,
               "psnr_color": args.psnr_color, "metrics": {}}
    for name, column in zip(METRICS, scores.T):
        valid = column[np.isfinite(column)]
        summary["metrics"][name] = {
            "mean": float(valid.mean()) if valid.size else None,
            "defined_pairs": int(valid.size),
        }
    result = json.dumps(summary, indent=2, allow_nan=False)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
