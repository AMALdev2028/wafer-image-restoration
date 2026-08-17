"""
inference.py
------------
MANDATORY standalone inference script per the hackathon spec (Section 4C):
  - accepts an input-directory argument and an output-directory argument
  - loads every degraded (NoisyLR) .npy image in the input directory,
    restores it, and saves each output to the output directory
  - preserves filenames
  - supports NVIDIA GPU execution, with batch processing
  - requires no source-code edits or manual path changes -- everything is
    a CLI argument
  - reports END-TO-END runtime: disk reading, preprocessing, CPU<->GPU
    transfer, model execution, post-processing, and saving -- not just the
    forward pass

Architecture (SIRNet vs SIRNetSR) and upscale factor are read from the
checkpoint's own saved run_config -- no need to specify --upscale-factor
manually if the checkpoint was trained with train.py.

Usage:
    python3 inference.py --input-dir Test_NoisyLR --output-dir results/predictions \\
        --ckpt weights/sirnet_best.pt --batch-size 16

    # With test-time augmentation (flips/rotations averaged -- usually a
    # small quality bump, at ~4x the per-image compute cost):
    python3 inference.py --input-dir Test_NoisyLR --output-dir results/predictions \\
        --ckpt weights/sirnet_best.pt --tta

Input/output contract:
  - Input: a directory of grayscale .npy arrays, shape (H,W), the official
    NoisyLR format. Values may fall outside [0,1] (intentional, per spec).
  - Output: one .npy per input file (same filename), shape matching the
    GT resolution, values clamped to [0,1] to match GT's range. No further
    clipping/normalization is applied by KLA's scoring, so this script is
    responsible for producing the final values (per spec 4A).
"""

import argparse
import os
import time
import numpy as np
import torch
import cv2

from src.checkpoint_utils import load_checkpoint, needs_pre_upsample


def load_batch(paths, upscale_factor, pre_upsample):
    """Load a batch of .npy files. If pre_upsample is True (SIRNet), resize
    each to (H*factor, W*factor) here; if False (SIRNetSR), leave at native
    resolution -- the model does its own learned upsampling."""
    arrays = []
    for p in paths:
        arr = np.load(p).astype(np.float32)
        arr = np.squeeze(arr)
        if pre_upsample and upscale_factor != 1:
            h, w = arr.shape
            arr = cv2.resize(arr, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_CUBIC)
        arrays.append(arr)
    batch = np.stack(arrays, axis=0)  # [B,H,W]
    return torch.from_numpy(batch).unsqueeze(1)  # [B,1,H,W]


# The 4 transforms used for TTA: identity, horizontal flip, vertical flip,
# 180-degree rotation. Each is its own inverse, so "undo" reuses the same
# function -- keeps this simple and correct without a separate inverse map.
_TTA_TRANSFORMS = {
    "identity": lambda t: t,
    "hflip": lambda t: torch.flip(t, dims=[-1]),
    "vflip": lambda t: torch.flip(t, dims=[-2]),
    "rot180": lambda t: torch.flip(t, dims=[-1, -2]),
}


def run_model_tta(model, batch_tensor):
    """Average the model's output over 4 flip/rotation augmentations of the
    input, each un-transformed back before averaging. Costs ~4x compute."""
    outputs = []
    for name, fn in _TTA_TRANSFORMS.items():
        aug_input = fn(batch_tensor)
        aug_output = model(aug_input)
        outputs.append(fn(aug_output))  # each transform is its own inverse
    return torch.stack(outputs, dim=0).mean(dim=0)


def main():
    ap = argparse.ArgumentParser(description="Standalone inference: input-dir -> output-dir.")
    ap.add_argument("--input-dir", type=str, required=True,
                     help="Directory of degraded (NoisyLR) .npy files to restore.")
    ap.add_argument("--output-dir", type=str, required=True,
                     help="Directory to write restored .npy files (same filenames as input).")
    ap.add_argument("--ckpt", type=str, default="weights/sirnet_best.pt")
    ap.add_argument("--batch-size", type=int, default=16,
                     help="Batch size for GPU inference. Reduce if you hit out-of-memory.")
    ap.add_argument("--upscale-factor", type=int, default=None,
                     help="Override the upscale factor read from the checkpoint's run_config "
                          "(rarely needed -- only if evaluating on a differently-scaled input "
                          "than the checkpoint was trained on).")
    ap.add_argument("--tta", action="store_true",
                     help="Test-time augmentation: average predictions over 4 flip/rotation "
                          "variants of each input for a small quality bump, at ~4x compute cost "
                          "per image. Affects the throughput numbers in the runtime report.")
    args = ap.parse_args()

    t_start = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({device_name})")

    t_load_model = time.time()
    model, run_config, ckpt = load_checkpoint(args.ckpt, device=device)
    pre_upsample = needs_pre_upsample(run_config)
    upscale_factor = args.upscale_factor if args.upscale_factor is not None else run_config.get("upscale_factor", 2)
    arch = run_config.get("arch", "unet")
    print(f"Loaded checkpoint {args.ckpt} (epoch {ckpt.get('epoch', '?')}, "
          f"val PSNR {ckpt.get('val_psnr', float('nan')):.2f} dB, arch={arch}, "
          f"upscale_factor={upscale_factor}) in {time.time()-t_load_model:.2f}s")
    if args.tta:
        print("Test-time augmentation ENABLED (4x forward passes per image)")

    filenames = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not filenames:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")
    print(f"Found {len(filenames)} input images in {args.input_dir}")

    n_done = 0
    t_pipeline_start = time.time()
    with torch.no_grad():
        for i in range(0, len(filenames), args.batch_size):
            batch_names = filenames[i:i + args.batch_size]
            batch_paths = [os.path.join(args.input_dir, f) for f in batch_names]

            # disk read + preprocessing (resize, if pre_upsample) happens inside load_batch
            batch_tensor = load_batch(batch_paths, upscale_factor, pre_upsample)

            # CPU -> GPU transfer + model execution
            batch_tensor = batch_tensor.to(device, non_blocking=True)
            if args.tta:
                restored = run_model_tta(model, batch_tensor)
            else:
                restored = model(batch_tensor)

            # GPU -> CPU transfer + post-processing + save
            restored_np = restored.cpu().numpy()
            for j, fname in enumerate(batch_names):
                out_arr = restored_np[j, 0].astype(np.float32)
                np.save(os.path.join(args.output_dir, fname), out_arr)

            n_done += len(batch_names)
            print(f"  {n_done}/{len(filenames)} restored", end="\r")

    t_end = time.time()
    pipeline_time = t_end - t_pipeline_start
    total_time = t_end - t_start

    print()
    print("=== End-to-end inference report ===")
    print(f"Images processed:        {n_done}")
    print(f"Batch size:              {args.batch_size}")
    print(f"Architecture:            {arch}{' (+ TTA 4x)' if args.tta else ''}")
    print(f"Device:                  {device} ({device_name})")
    print(f"Pipeline time (I/O + preprocessing + transfer + model + save): {pipeline_time:.2f}s")
    print(f"  -> {n_done / pipeline_time:.2f} images/sec")
    print(f"Total wall time (incl. model load): {total_time:.2f}s")
    print(f"Output written to: {args.output_dir}")


if __name__ == "__main__":
    main()
