"""
evaluate.py
-----------
Loads a trained checkpoint (either architecture -- SIRNet or SIRNetSR,
auto-detected from the checkpoint's saved run_config), runs it on a
held-out set, and reports:
  - PSNR, SSIM, LPIPS (all three required by the hackathon spec)
  - a comparison against a BASELINE (bicubic upsample only, no learned
    restoration) -- required: "compare at least one baseline with the
    final method"
  - a visual comparison grid (degraded | bicubic baseline | model | GT)

Usage:
    # Official dataset val split:
    python3 evaluate.py --ckpt weights/sirnet_best.pt --data-root path/to/train

    # Synthetic held-out set:
    python3 evaluate.py --ckpt weights/sirnet_best.pt
"""

import argparse
import os
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from src.data import SemiconductorRestorationDataset, RealImageRestorationDataset, make_train_val_split
from src.checkpoint_utils import load_checkpoint, needs_pre_upsample

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def bicubic_baseline(degraded_np, target_shape):
    """The simplest possible baseline: bicubic-resize the degraded input to
    the target resolution, no learned denoising/deblurring at all. This is
    what the model needs to beat to justify training one at all."""
    h, w = target_shape
    return cv2.resize(np.clip(degraded_np, 0, 1), (w, h), interpolation=cv2.INTER_CUBIC)


def to_lpips_tensor(img_np, device):
    """(H,W) float32 in [0,1] -> [1,3,H,W] in [-1,1], what the lpips package expects."""
    t = torch.from_numpy(img_np).float().clamp(0, 1)
    t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # grayscale -> 3ch
    t = t * 2 - 1
    return t.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="weights/sirnet_best.pt")
    ap.add_argument("--img-size", type=int, default=80)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--n-metric-samples", type=int, default=60)
    ap.add_argument("--out-fig", type=str, default="results/comparison_grid.png")
    ap.add_argument("--real-dir", type=str, default=None,
                     help="Folder of your own raw image(s) to evaluate on instead of synthetic data.")
    ap.add_argument("--data-root", type=str, default=None,
                     help="Path to the OFFICIAL hackathon dataset folder (containing GT/ and "
                          "NoisyLR/ subfolders) -- evaluates on its held-out validation split "
                          "(same split train.py used, via matching --val-fraction/seed).")
    ap.add_argument("--val-fraction", type=float, default=0.1,
                     help="Must match the value used in train.py's --data-root run.")
    ap.add_argument("--no-lpips", action="store_true",
                     help="Skip LPIPS (e.g. if the pretrained LPIPS weights can't be downloaded offline).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, run_config, ckpt = load_checkpoint(args.ckpt, device=device)
    arch = run_config.get("arch", "unet")
    upscale_factor = run_config.get("upscale_factor", 2)
    pre_upsample = needs_pre_upsample(run_config)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val PSNR {ckpt['val_psnr']:.2f} dB, "
          f"arch={arch}, upscale_factor={upscale_factor})")

    use_lpips = _LPIPS_AVAILABLE and not args.no_lpips
    if use_lpips:
        try:
            lpips_model = lpips_lib.LPIPS(net="alex").to(device)
            lpips_model.eval()
        except Exception as e:
            print(f"Could not load LPIPS pretrained weights ({e}); continuing without LPIPS. "
                  f"This usually means no internet access -- LPIPS needs to download weights once.")
            use_lpips = False
    else:
        if not _LPIPS_AVAILABLE:
            print("`lpips` package not installed (pip install lpips); continuing without LPIPS.")

    # Fresh held-out set, seeds disjoint from both train (0..) and val (100000..)
    if args.data_root:
        gt_dir = os.path.join(args.data_root, "GT")
        noisy_dir = os.path.join(args.data_root, "NoisyLR")
        print(f"Evaluating on the OFFICIAL hackathon dataset's val split: {gt_dir} / {noisy_dir}")
        _, test_ds = make_train_val_split(gt_dir, noisy_dir, val_fraction=args.val_fraction,
                                           pre_upsample=pre_upsample, augment=False)
        print(f"  {len(test_ds)} val pairs")
    elif args.real_dir:
        print(f"Evaluating on real image(s) from: {args.real_dir}")
        test_ds = RealImageRestorationDataset(args.real_dir, length=args.n_metric_samples, size=args.img_size, seed_offset=500_000)
    else:
        test_ds = SemiconductorRestorationDataset(
            length=args.n_metric_samples, size=args.img_size, seed_offset=500_000,
            lr_mode=(not pre_upsample), upscale_factor=upscale_factor,
        )

    metrics = {k: [] for k in [
        "psnr_baseline", "psnr_model", "ssim_baseline", "ssim_model",
        "lpips_baseline", "lpips_model",
    ]}
    samples_for_plot = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            degraded, clean = test_ds[i]
            restored = model(degraded.unsqueeze(0).to(device)).cpu().squeeze(0)

            d = degraded.squeeze(0).numpy()
            c = clean.squeeze(0).numpy()
            r = restored.squeeze(0).numpy()
            baseline = bicubic_baseline(d, c.shape)
            d_display = np.clip(d, 0, 1) if d.shape == c.shape else baseline

            metrics["psnr_baseline"].append(sk_psnr(c, baseline, data_range=1.0))
            metrics["psnr_model"].append(sk_psnr(c, r, data_range=1.0))
            metrics["ssim_baseline"].append(sk_ssim(c, baseline, data_range=1.0))
            metrics["ssim_model"].append(sk_ssim(c, r, data_range=1.0))

            if use_lpips:
                c_t, b_t, r_t = to_lpips_tensor(c, device), to_lpips_tensor(baseline, device), to_lpips_tensor(r, device)
                metrics["lpips_baseline"].append(lpips_model(c_t, b_t).item())
                metrics["lpips_model"].append(lpips_model(c_t, r_t).item())

            if i < args.n_samples:
                samples_for_plot.append((d_display, baseline, r, c))

    dataset_label = "OFFICIAL hackathon val split" if args.data_root else (
        "real image(s)" if args.real_dir else "synthetic test set")
    print(f"\n=== Restoration quality on held-out {dataset_label} (n={len(test_ds)}) ===")
    print(f"{'Metric':<10}{'Bicubic baseline':>20}{'Model (' + arch + ')':>22}{'Improvement':>14}")
    print(f"{'PSNR':<10}{np.mean(metrics['psnr_baseline']):>17.2f}dB{np.mean(metrics['psnr_model']):>19.2f}dB"
          f"{np.mean(metrics['psnr_model'])-np.mean(metrics['psnr_baseline']):>+11.2f}dB")
    print(f"{'SSIM':<10}{np.mean(metrics['ssim_baseline']):>20.3f}{np.mean(metrics['ssim_model']):>22.3f}"
          f"{np.mean(metrics['ssim_model'])-np.mean(metrics['ssim_baseline']):>+14.3f}")
    if use_lpips:
        print(f"{'LPIPS*':<10}{np.mean(metrics['lpips_baseline']):>20.3f}{np.mean(metrics['lpips_model']):>22.3f}"
              f"{np.mean(metrics['lpips_model'])-np.mean(metrics['lpips_baseline']):>+14.3f}")
        print("*lower LPIPS is better (unlike PSNR/SSIM)")
    else:
        print("LPIPS: skipped (see message above)")

    # Comparison figure: degraded | bicubic baseline | model | GT
    n = len(samples_for_plot)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Degraded (input)", "Bicubic baseline", f"Model output ({arch})", "Ground truth (clean)"]
    for row, imgs in enumerate(samples_for_plot):
        for col, (img, title) in enumerate(zip(imgs, col_titles)):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150)
    print(f"\nSaved comparison grid to {args.out_fig}")


if __name__ == "__main__":
    main()
