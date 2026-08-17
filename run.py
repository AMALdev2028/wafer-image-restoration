"""
run.py
------
MANDATORY entry point per the organizers' final submission check.

Usage (positional arguments, exactly as required):
    python run.py <input-dir> <output-dir>
"""

import os
import sys
import time
import numpy as np
import torch
import cv2

from src.checkpoint_utils import load_checkpoint, needs_pre_upsample

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sirnet_best.pt")

BATCH_SIZE = 16


def load_batch(paths, upscale_factor, pre_upsample):
    arrays = []
    for p in paths:
        arr = np.load(p).astype(np.float32)
        arr = np.squeeze(arr)
        if pre_upsample and upscale_factor != 1:
            h, w = arr.shape
            arr = cv2.resize(arr, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_CUBIC)
        arrays.append(arr)
    batch = np.stack(arrays, axis=0)
    return torch.from_numpy(batch).unsqueeze(1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({device_name})")

    t0 = time.time()
    model, run_config, ckpt = load_checkpoint(DEFAULT_CKPT, device=device)
    pre_upsample = needs_pre_upsample(run_config)
    upscale_factor = run_config.get("upscale_factor", 2)
    print(f"Loaded model (arch={run_config.get('arch', 'unet')}, "
          f"upscale_factor={upscale_factor}) in {time.time()-t0:.2f}s")

    filenames = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))
    if not filenames:
        print(f"Error: no .npy files found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(filenames)} input files")

    n_done = 0
    n_fixed = 0
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(filenames), BATCH_SIZE):
            batch_names = filenames[i:i + BATCH_SIZE]
            batch_paths = [os.path.join(input_dir, f) for f in batch_names]

            batch_tensor = load_batch(batch_paths, upscale_factor, pre_upsample)
            batch_tensor = batch_tensor.to(device, non_blocking=True)
            restored = model(batch_tensor)
            restored_np = restored.cpu().numpy()

            for j, fname in enumerate(batch_names):
                out_arr = restored_np[j, 0].astype(np.float32)

                if not np.isfinite(out_arr).all():
                    out_arr = np.nan_to_num(out_arr, nan=0.0, posinf=1.0, neginf=0.0)
                    n_fixed += 1
                out_arr = np.clip(out_arr, 0.0, 1.0)

                np.save(os.path.join(output_dir, fname), out_arr)

            n_done += len(batch_names)
            print(f"  {n_done}/{len(filenames)} restored", end="\r")

    elapsed = time.time() - t_start
    print()
    print(f"Done: {n_done} files restored to {output_dir}")
    print(f"Time: {elapsed:.2f}s ({n_done/elapsed:.2f} images/sec)")
    if n_fixed:
        print(f"Note: {n_fixed} output(s) required NaN/Inf cleanup before saving.")


if __name__ == "__main__":
    main()