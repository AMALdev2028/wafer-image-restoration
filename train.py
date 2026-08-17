"""
train.py
--------
Trains a restoration model on (degraded -> clean) semiconductor image pairs
and saves the best checkpoint + training curves + the exact run
configuration (for reproducibility, per the hackathon's "training & compute
hygiene" evaluation axis).

Two architectures (--arch):
    unet     (default) SIRNet -- expects NoisyLR pre-upsampled to GT
             resolution before it reaches the network (fixed bicubic step
             outside the model).
    unet_sr  SIRNetSR -- takes the genuine low-resolution NoisyLR directly
             and learns the upsampling itself via PixelShuffle. Usually
             sharper, since the upsampling is no longer a fixed baseline.

Loss: L1 + SSIM by default, per the loss-functions survey KLA cited; add
--lpips-weight > 0 to also optimize directly for perceptual quality (the
same metric used for scoring), at extra compute cost per step.

Usage:
    # Official hackathon dataset, original architecture:
    python3 train.py --data-root path/to/train --epochs 20 --batch-size 16

    # True super-resolution architecture:
    python3 train.py --data-root path/to/train --arch unet_sr --upscale-factor 2 --epochs 20

    # Mix in synthetic data alongside the real pairs (spec explicitly allows this):
    python3 train.py --data-root path/to/train --synthetic-mix-len 200 --epochs 20

    # Add a perceptual (LPIPS) loss term:
    python3 train.py --data-root path/to/train --lpips-weight 0.1 --epochs 20

    # Continue training an existing checkpoint up to a higher total epoch
    # count (e.g. it was trained to epoch 5, this takes it to epoch 15):
    python3 train.py --data-root path/to/train --epochs 15 --resume weights/sirnet_best.pt
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_msssim import SSIM

from src.data import (
    SemiconductorRestorationDataset,
    RealImageRestorationDataset,
    make_train_val_split,
)
from src.model import build_model

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


class CombinedLoss(nn.Module):
    """loss = l1_weight*L1 + ssim_weight*(1-SSIM) [+ lpips_weight*LPIPS]

    LPIPS expects 3-channel images in [-1,1]; our images are 1-channel
    [0,1], so they're repeated to 3 channels and rescaled before going in.
    """

    def __init__(self, l1_weight=0.8, ssim_weight=0.2, lpips_weight=0.0, device="cpu"):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
        self.lpips_model = None
        if lpips_weight > 0:
            if not _LPIPS_AVAILABLE:
                raise RuntimeError("--lpips-weight > 0 but the `lpips` package isn't installed "
                                    "(pip install lpips).")
            try:
                self.lpips_model = lpips_lib.LPIPS(net="alex").to(device)
            except Exception as e:
                raise RuntimeError(
                    "--lpips-weight > 0 but couldn't load LPIPS's pretrained AlexNet weights "
                    f"(usually means no internet access on first run): {e}\n"
                    "Fix: run with internet access once (weights are cached after that), or "
                    "drop --lpips-weight to train with L1+SSIM only."
                ) from e
            self.lpips_model.eval()
            for p in self.lpips_model.parameters():
                p.requires_grad_(False)

    def forward(self, pred, target):
        loss = self.l1_weight * self.l1(pred, target)
        loss = loss + self.ssim_weight * (1 - self.ssim(pred, target))
        if self.lpips_model is not None:
            pred_3ch = (pred.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)
            target_3ch = (target.clamp(0, 1) * 2 - 1).repeat(1, 3, 1, 1)
            loss = loss + self.lpips_weight * self.lpips_model(pred_3ch, target_3ch).mean()
        return loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return 99.0
    return 10 * np.log10(1.0 / mse)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for degraded, clean in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        if train:
            optimizer.zero_grad()
        restored = model(degraded)
        loss = criterion(restored, clean)
        if train:
            loss.backward()
            optimizer.step()
        bs = degraded.size(0)
        total_loss += loss.item() * bs
        total_psnr += psnr(restored.detach(), clean) * bs
        n += bs
    return total_loss / n, total_psnr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=128,
                     help="Only used for synthetic/--real-dir data; --data-root uses the real image size.")
    ap.add_argument("--train-len", type=int, default=400,
                     help="Synthetic/--real-dir only: samples generated per epoch.")
    ap.add_argument("--val-len", type=int, default=60,
                     help="Synthetic/--real-dir only: validation samples.")
    ap.add_argument("--out", type=str, default="weights/sirnet_best.pt")
    ap.add_argument("--ssim-weight", type=float, default=0.2,
                     help="Weight of the (1-SSIM) term; L1 gets (1 - ssim_weight - lpips_weight).")
    ap.add_argument("--lpips-weight", type=float, default=0.0,
                     help="Weight of an LPIPS perceptual loss term added during training "
                          "(0 = disabled). Requires `pip install lpips` and internet access "
                          "on first run to fetch pretrained weights.")
    ap.add_argument("--seed", type=int, default=42,
                     help="Random seed for reproducibility (weights init, shuffling, synthetic data).")
    ap.add_argument("--real-dir", type=str, default=None,
                     help="Folder of your own raw image(s) to train on instead of synthetic data "
                          "(these get synthetically degraded, unlike --data-root).")
    ap.add_argument("--data-root", type=str, default=None,
                     help="Path to the OFFICIAL hackathon dataset folder containing GT/ and "
                          "NoisyLR/ subfolders of paired .npy files (e.g. 'train'). Takes "
                          "priority over --real-dir and synthetic data if set.")
    ap.add_argument("--val-fraction", type=float, default=0.1,
                     help="Fraction of --data-root held out for validation (only used with --data-root).")
    ap.add_argument("--results-dir", type=str, default="results",
                     help="Where training curves / history / run config are written.")
    ap.add_argument("--resume", type=str, default=None,
                     help="Path to an existing checkpoint to continue training from "
                          "(model + optimizer + scheduler state). --epochs is the TOTAL "
                          "epoch count to reach, not additional epochs -- e.g. a checkpoint "
                          "at epoch 5 with --epochs 15 trains 10 more epochs (6-15). Architecture "
                          "is read from the checkpoint itself, ignoring --arch/--upscale-factor.")
    ap.add_argument("--arch", type=str, default="unet", choices=["unet", "unet_sr"],
                     help="unet: pre-upsampled fixed-size input (original). "
                          "unet_sr: true super-resolution, takes native low-res input directly.")
    ap.add_argument("--upscale-factor", type=int, default=2,
                     help="GT:NoisyLR resolution ratio. Only affects --arch unet_sr and synthetic "
                          "data generation; --data-root with --arch unet infers pre-upsample "
                          "target from the GT files themselves.")
    ap.add_argument("--no-augment", action="store_true",
                     help="Disable flip/rotation augmentation on the real (--data-root) training split.")
    ap.add_argument("--synthetic-mix-len", type=int, default=0,
                     help="When using --data-root, additionally mix in this many synthetic "
                          "degraded pairs per epoch (generated from procedural GT images, per "
                          "the spec's 'you may create extra synthetic degraded pairs' allowance). "
                          "0 = disabled (real data only).")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    arch = args.arch
    upscale_factor = args.upscale_factor

    # If resuming, the checkpoint's own architecture is authoritative -- read
    # it BEFORE building datasets, since pre_upsample/lr_mode depend on it.
    resume_ckpt = None
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        resumed_config = resume_ckpt.get("run_config", {})
        ckpt_arch = resumed_config.get("arch", args.arch)
        ckpt_upscale = resumed_config.get("upscale_factor", args.upscale_factor)
        if ckpt_arch != arch or ckpt_upscale != upscale_factor:
            print(f"  NOTE: checkpoint was trained with arch={ckpt_arch}, upscale_factor={ckpt_upscale} "
                  f"-- overriding --arch/--upscale-factor to match (architecture is fixed by the checkpoint).")
            arch = ckpt_arch
            upscale_factor = ckpt_upscale

    pre_upsample = (arch == "unet")

    if args.data_root:
        gt_dir = os.path.join(args.data_root, "GT")
        noisy_dir = os.path.join(args.data_root, "NoisyLR")
        print(f"Training on the OFFICIAL hackathon dataset: {gt_dir} / {noisy_dir}")
        train_ds, val_ds = make_train_val_split(
            gt_dir, noisy_dir, val_fraction=args.val_fraction, seed=args.seed,
            pre_upsample=pre_upsample, augment=not args.no_augment,
        )
        print(f"  {len(train_ds)} train pairs, {len(val_ds)} val pairs "
              f"({args.val_fraction:.0%} held out), augment={not args.no_augment}")
        data_source = "official_dataset"

        if args.synthetic_mix_len > 0:
            # Infer GT resolution from one real sample so synthetic images match.
            sample_clean = train_ds[0][1]
            gt_size = sample_clean.shape[-1]
            synth_ds = SemiconductorRestorationDataset(
                length=args.synthetic_mix_len, size=gt_size, seed_offset=0,
                lr_mode=(arch == "unet_sr"), upscale_factor=upscale_factor,
            )
            train_ds = ConcatDataset([train_ds, synth_ds])
            print(f"  + {args.synthetic_mix_len} synthetic samples/epoch mixed in "
                  f"(total train pool: {len(train_ds)})")
    elif args.real_dir:
        print(f"Training on real image(s) from: {args.real_dir}")
        train_ds = RealImageRestorationDataset(args.real_dir, length=args.train_len, size=args.img_size, seed_offset=0)
        val_ds = RealImageRestorationDataset(args.real_dir, length=args.val_len, size=args.img_size, seed_offset=100_000)
        data_source = "real_dir_synthetic_degradation"
    else:
        train_ds = SemiconductorRestorationDataset(
            length=args.train_len, size=args.img_size, seed_offset=0,
            lr_mode=(arch == "unet_sr"), upscale_factor=upscale_factor,
        )
        val_ds = SemiconductorRestorationDataset(
            length=args.val_len, size=args.img_size, seed_offset=100_000,
            lr_mode=(arch == "unet_sr"), upscale_factor=upscale_factor,
        )
        data_source = "fully_synthetic"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    start_epoch = 1
    best_val_psnr = -1
    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}

    model = build_model(arch=arch, upscale_factor=upscale_factor).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    lpips_weight = args.lpips_weight
    l1_weight = 1.0 - args.ssim_weight - lpips_weight
    criterion = CombinedLoss(l1_weight=l1_weight, ssim_weight=args.ssim_weight,
                              lpips_weight=lpips_weight, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if resume_ckpt is not None:
        ckpt = resume_ckpt
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_psnr = ckpt.get("val_psnr", -1)
        print(f"  Resuming at epoch {start_epoch} (checkpoint was at epoch {ckpt.get('epoch')}, "
              f"val PSNR {best_val_psnr:.2f} dB)")

        history_path = os.path.join(args.results_dir, "history.npy")
        if os.path.exists(history_path):
            try:
                loaded = np.load(history_path, allow_pickle=True).item()
                if all(k in loaded for k in history):
                    history = loaded
                    print(f"  Loaded existing history.npy ({len(history['train_loss'])} epochs so far)")
            except Exception as e:
                print(f"  Could not load existing history.npy ({e}); starting a fresh history log.")

        if start_epoch > args.epochs:
            print(f"Checkpoint is already at epoch {start_epoch - 1}, which is >= "
                  f"--epochs {args.epochs}. Nothing to do -- pass a higher --epochs to continue training.")
            return

    # Persist the exact run configuration alongside the checkpoint -- this
    # is what lets anyone (including you, later) reproduce this exact run.
    run_config = {
        "data_source": data_source,
        "data_root": args.data_root,
        "real_dir": args.real_dir,
        "val_fraction": args.val_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "img_size": args.img_size,
        "train_len": args.train_len,
        "val_len": args.val_len,
        "ssim_weight": args.ssim_weight,
        "lpips_weight": lpips_weight,
        "seed": args.seed,
        "arch": arch,
        "upscale_factor": upscale_factor,
        "augment": not args.no_augment,
        "synthetic_mix_len": args.synthetic_mix_len,
        "model": "SIRNetSR" if arch == "unet_sr" else "SIRNet",
        "model_params": n_params,
        "loss": "CombinedLoss",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "device": str(device),
        "resumed_from": args.resume,
        "start_epoch": start_epoch,
    }
    with open(os.path.join(args.results_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_psnr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_psnr = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_psnr"].append(tr_psnr)
        history["val_psnr"].append(va_psnr)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} PSNR {tr_psnr:5.2f}dB | "
              f"val loss {va_loss:.4f} PSNR {va_psnr:5.2f}dB")

        if va_psnr > best_val_psnr:
            best_val_psnr = va_psnr
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_psnr": float(va_psnr),
                "epoch": int(epoch),
                "run_config": run_config,
            }, args.out)

    total_time = time.time() - t0
    print(f"Training done in {total_time:.1f}s. Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoint saved to {args.out}")

    np.save(os.path.join(args.results_dir, "history.npy"), history)
    run_config["best_val_psnr"] = best_val_psnr
    run_config["training_time_sec"] = total_time
    with open(os.path.join(args.results_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)


if __name__ == "__main__":
    main()
