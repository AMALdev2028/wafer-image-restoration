# AI-Based Restoration of Degraded Images for Semiconductor Inspection
KLA Problem Statement — SEMICON India Hackathon 2026

A reproducible AI-based image-restoration pipeline that restores degraded
(speckle noise + additive Gaussian noise + downsampled) semiconductor
inspection imagery back toward ground-truth quality, using **SIRNet**, a
lightweight residual U-Net trained with PyTorch. Built to match the official
problem statement (`KLA_Problem_Statement_Studen_help_document.pdf`) and the
KLA webinar slides (`KLA_Problem_Statement_explanation.pptx`).

## Repository structure
```
README.md                  <- this file: setup, commands, contract, assumptions
requirements.txt
train.py                   <- reproducible training script (--arch, --resume, mixing, LPIPS loss)
inference.py                <- MANDATORY standalone inference script (input-dir -> output-dir)
evaluate.py                 <- PSNR/SSIM/LPIPS validation + baseline comparison
configs/
    default.yaml             <- documented default hyperparameters
src/
    data.py                   <- dataset loaders (official .npy pairs + synthetic generator)
    model.py                  <- SIRNet + SIRNetSR architectures
    checkpoint_utils.py        <- shared checkpoint loading (arch-aware), used by every script
webapp/                     <- optional local demo (not part of the submission)
weights/
    sirnet_best.pt             <- trained checkpoint (includes embedded run config)
results/
    comparison_grid.png        <- degraded / bicubic baseline / model / GT visual comparison
    run_config.json             <- exact hyperparameters + seed used for the saved checkpoint
    history.npy                  <- per-epoch loss/PSNR, for plotting training curves
solution_presentation.pptx  <- Phase 1 solution deck (separate file, see submission checklist)
```

## Environment setup
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Requires Python 3.9+. GPU (CUDA) is auto-detected and used automatically if
available; everything also runs on CPU (slower, especially at 256x256+).

## The official dataset (per Section 4A of the spec)
KLA provides paired `.npy` files, matched by filename (not by index — the
numbering has gaps, e.g. `000040.npy`, `000041.npy`, `000054.npy`...):
```
train/
    GT/          <- clean ground truth, always [0,1], ~256x256 or 512x512
        000040.npy
        ...
    NoisyLR/     <- degraded input, MAY exceed [0,1] intentionally, lower resolution
        000040.npy
        ...
Test_NoisyLR/    <- hidden-style held-out test set, NO ground truth
    ...
```
**Assumption used throughout this repo**: GT is 2x the resolution of
NoisyLR (e.g. GT 256x256 vs NoisyLR 128x128), based on the actual dataset
files inspected during development. If your download uses a different
ratio (the spec also mentions 512x512), pass `--upscale-factor` accordingly
to `inference.py` (and adjust `RealImageRestorationDataset`/training if the
train-time ratio also differs) — check with:
```bash
python3 -c "import numpy as np; print(np.load('train/GT/000040.npy').shape, np.load('train/NoisyLR/000040.npy').shape)"
```

Degradations are speckle noise + additive Gaussian noise + downsampling,
applied in an undisclosed order — per spec, the model does not need to
identify the order, only invert the net effect.

## Commands

**Train** on the official dataset:
```bashpython3 train.py --data-root train --epochs 5 --batch-size 16 --seed 42
python3 train.py --data-root train --epochs 15 --resume weights/sirnet_best.pt
```
This does a deterministic 90/10 train/val split by filename (`--val-fraction`,
same seed every run so results are comparable), applies flip/rotation
augmentation to the training split (`--no-augment` to disable), trains with
an L1+SSIM loss, and writes:
- `weights/sirnet_best.pt` — best checkpoint (by val PSNR), with the full
  run configuration (architecture, hyperparameters, seed) embedded inside it
- `results/run_config.json` — the same configuration, human-readable
- `results/history.npy` — per-epoch train/val loss and PSNR

**Continue training an existing checkpoint** rather than starting over
(`--epochs` is the TOTAL epoch count to reach, not additional epochs):
```bash
python3 train.py --data-root train --epochs 30 --resume weights/sirnet_best.pt
```
Architecture is read from the checkpoint itself and any conflicting
`--arch`/`--upscale-factor` flags are overridden to match, with a printed note.

**True super-resolution architecture** (SIRNetSR — takes the genuinely
lower-resolution NoisyLR directly and learns the upsampling itself via
PixelShuffle, instead of a fixed bicubic pre-upsample step):
```bash
python3 train.py --data-root train --arch unet_sr --upscale-factor 2 --epochs 20
```

**Mix in synthetic data** alongside the real pairs (spec explicitly allows
this: "you may create extra synthetic degraded pairs from the provided GT
images"):
```bash
python3 train.py --data-root train --synthetic-mix-len 200 --epochs 20
```

**Add a perceptual (LPIPS) loss term** during training, directly optimizing
for one of the metrics you're scored on (extra compute per step; requires
`pip install lpips` and internet access on first run):
```bash
python3 train.py --data-root train --lpips-weight 0.1 --epochs 20
```

**Evaluate** (PSNR, SSIM, LPIPS, vs. a bicubic baseline, per spec Section 4D):
```bash
python3 evaluate.py --ckpt weights/sirnet_best.pt --data-root train
```
Reports all three required metrics for both the trained model and a plain
bicubic-upsample baseline (no learned restoration), and saves a 4-column
visual comparison grid (degraded / bicubic baseline / model / ground
truth) to `results/comparison_grid.png`. Architecture is auto-detected from
the checkpoint. LPIPS needs to download pretrained AlexNet weights once
(standard `pip install lpips` behavior) — if you're offline, pass
`--no-lpips` and report PSNR/SSIM only.

**Run inference** (the mandatory standalone script, Section 4C) on any
folder of degraded `.npy` images — including KLA's own hidden test set,
unmodified:
```bash
python3 inference.py --input-dir Test_NoisyLR --output-dir results/predictions \
    --ckpt weights/sirnet_best.pt --batch-size 16
```
Writes one restored `.npy` per input (same filename, at GT resolution,
values clamped to [0,1] — KLA does not clip/renormalize on their end, so
this script owns that responsibility per spec). Architecture and upscale
factor are read from the checkpoint automatically. Prints an end-to-end
runtime report (I/O + preprocessing + CPU↔GPU transfer + model execution +
saving, plus images/sec throughput) — not just the model forward-pass time,
per the spec's runtime definition. No source-code edits needed; everything
is a CLI flag.

Add `--tta` for test-time augmentation (averages predictions over 4
flip/rotation variants of each input) — a small, usually-free quality bump
at ~4x the per-image compute cost; check the printed throughput before
deciding if that trade-off is worth it for your runtime score.

**Quick iteration without the real dataset**: all scripts fall back to a
procedurally generated synthetic semiconductor-image dataset if
`--data-root` isn't passed (see `src/data.py`) — useful for testing
architecture/loss changes before running a full real-data pass.

## Architecture
Two architectures are available (`--arch`), both residual-learning
encoder/decoder CNNs with skip connections:

**SIRNet** (`--arch unet`, default, ~467K params) — expects NoisyLR already
bicubic-resized to GT resolution before it reaches the network:
```
Input (1 x H x W, may exceed [0,1])
 -> Enc1 (32ch) -----------------------------\
     v pool                                   |  skip
 -> Enc2 (64ch) --------------------\         |
     v pool                          | skip   |
 -> Bottleneck (128ch)                |        |
     v upsample + concat -------------/        |
 -> Dec2 (64ch)                                |
     v upsample + concat -----------------------/
 -> Dec1 (32ch)
 -> 1x1 conv -> residual correction
 -> clamp(input + residual, 0, 1) -> Restored (1 x H x W)
```

**SIRNetSR** (`--arch unet_sr`, ~514K params) — takes the genuinely
lower-resolution NoisyLR directly; the same encoder/decoder body runs at
that native (smaller) resolution, then a PixelShuffle head produces the
upsampled residual directly, added to a bicubic-upsampled base of the raw
input (same residual/clamp rationale as SIRNet). This removes the fixed,
non-learned upsampling step — the network learns the upsampling itself,
usually sharper than feeding a same-resolution U-Net a pre-upsampled input.

Both predict a *correction* on top of the input rather than the final pixel
value directly through a sigmoid — more robust given the input isn't
guaranteed to lie in [0,1]. A U-Net body (vs. a plain denoising residual net
like DnCNN) was chosen because the degradation includes resolution loss,
not just additive noise — multi-scale skip connections handle that combined
restoration better.

`src/checkpoint_utils.py` is the single place that reads a checkpoint's
saved architecture and reconstructs the right model — `train.py --resume`,
`evaluate.py`, `inference.py`, and `webapp/app.py` all go through it, so
they can't disagree about what a checkpoint means.

## Loss function
`CombinedLoss` = `(1 - ssim_weight - lpips_weight) x L1 + ssim_weight x (1 - SSIM) [+ lpips_weight x LPIPS]`,
per the loss-functions survey KLA cited (Terven et al. 2025). Plain L1
alone tends to produce slightly blurry output since it averages over
plausible fine detail; the SSIM term rewards sharper, more structurally
faithful restorations. The optional LPIPS term (`--lpips-weight`, default 0)
directly optimizes for perceptual quality — one of the three metrics
you're scored on — at extra compute cost per training step.

## Baseline
`evaluate.py` always reports a **bicubic-upsample-only baseline** (no
learned model at all) alongside SIRNet's numbers, per spec Section 4D
("compare at least one baseline with the final method"). This is the floor
SIRNet needs to clear to justify training a model at all.

## Assumptions made (per spec Section 4A, "use the dimensions supplied in
the official dataset")
- GT:NoisyLR resolution ratio is 2x (confirmed against actual downloaded
  files during development: GT 256x256, NoisyLR 128x128). If evaluation
  uses 512x512 GT as the spec also mentions, re-check the ratio and adjust
  `--upscale-factor`.
- Grayscale, single-channel images throughout (matches the `.npy` shape
  `(H,W)` seen in the dataset).
- No clipping/renormalization is applied by KLA when scoring — `inference.py`
  and the model's residual+clamp design are responsible for producing
  already-correct [0,1] output.

## Interactive demo (optional, not part of the required submission)
A small local Flask app, same pattern as the wafer-defect-detection demo —
upload a degraded image, see the restored output side by side in a browser.
This is for visual sanity-checking during development; `inference.py` is
the actual required standalone script for the hackathon submission.
```bash
cd webapp
python3 app.py
```
Then open `http://127.0.0.1:5000`. Accepts either the official `.npy`
format (values may exceed [0,1], handled as-is) or a regular `.png`/`.jpg`
(converted to grayscale and normalized). Uses whatever checkpoint is at
`weights/sirnet_best.pt` — train one first if that file doesn't exist yet.

## External Resources

| Resource | Use | License |
|---|---|---|
| [pytorch-msssim](https://github.com/VainF/pytorch-msssim) | SSIM loss term during training | BSD-3-Clause |
| [lpips](https://github.com/richzhang/PerceptualSimilarity) (Zhang et al., 2018) | Perceptual quality metric (evaluation); optional training loss term | BSD-2-Clause |
| [scikit-image](https://scikit-image.org/) | PSNR and SSIM computation | BSD-3-Clause |
| KLA paired GT / NoisyLR training set | Training and validation data | Provided by organisers |

No other external datasets or pretrained model weights were used.

## Limitations

- Resolution is currently handled via bicubic pre-upsampling in the default
  architecture; a true learned super-resolution variant (`--arch unet_sr`,
  using PixelShuffle) is implemented but has not been directly compared
  against the default architecture on the real dataset yet.
- We tested a higher SSIM-loss weight (0.35 vs the default 0.2) over an
  additional 10 epochs; validation PSNR did not improve over the original
  configuration, so we kept the original weighting. Included here as an
  honest ablation, not a hidden negative result.
- LPIPS requires internet access on first run to download pretrained AlexNet
  weights.
- Validation is a 10% held-out split of the provided training data (fixed
  seed, no leakage) — it reflects in-distribution performance; true
  out-of-distribution behavior on KLA's hidden test set cannot be verified
  from our side.
- Restoration is noticeably weaker on scenes with dense fine texture against
  a highly reflective background (see failure case example) compared to
  other content types.
