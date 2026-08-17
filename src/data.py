"""
data.py
--------
Synthetic semiconductor / wafer-die image generator + degradation pipeline.

We don't have access to a licensed real-world SEM/wafer dataset inside this
environment, so this module procedurally generates clean "semiconductor-like"
grayscale images (die grid lines, circuit traces, contact pads, wafer-edge
shading) and then synthetically degrades them.

Degradation pipeline (aligned with the KLA "AI-Based Restoration of Degraded
Images" hackathon problem statement, f: x -> y):
    - down-sampling / resolution loss (random factor, then upsampled back
      to the original grid so shapes still line up for a fixed-size network)
    - optical defocus (Gaussian blur)
    - low contrast / uneven illumination
    - additive Gaussian (sensor/shot) noise
    - speckle (multiplicative) noise
    - dead / hot pixels (salt & pepper) -- kept as an extra realism knob,
      the hackathon's own f only lists the three above

The clean image is always kept as the paired ground-truth so the restoration
network is trained with (degraded -> clean) supervision.

IMPORTANT (per the problem-statement webinar): the ground-truth image is
always normalized to [0, 1], but the degraded/corrupted image is allowed to
take values *outside* [0, 1] -- e.g. Gaussian/speckle noise pushing pixels
below 0 or above 1. That's intentional, not a bug, so the degraded image is
NOT clipped to [0, 1] here (only the clean GT is). Downstream code (model,
loss, metrics) needs to expect that.

To use REAL wafer/SEM images instead: drop them (grayscale, any size) into
`real_images/`, and set `USE_REAL_DIR = "real_images"` when constructing the
Dataset -- the same degradation pipeline will be applied to them automatically.
"""

import numpy as np
import cv2
from torch.utils.data import Dataset
import torch


def _draw_die_grid(img, size, spacing):
    """Draw a die/reticle grid, like the streets between chips on a wafer."""
    for x in range(0, size, spacing):
        thickness = np.random.choice([1, 1, 2])
        cv2.line(img, (x, 0), (x, size), 1.0, thickness)
    for y in range(0, size, spacing):
        thickness = np.random.choice([1, 1, 2])
        cv2.line(img, (0, y), (size, y), 1.0, thickness)
    return img


def _draw_circuit_traces(img, size, n_traces=25):
    """Random rectilinear metal traces + contact pads, IC-layout style."""
    for _ in range(n_traces):
        x, y = np.random.randint(0, size, 2)
        length = np.random.randint(size // 12, size // 4)
        horizontal = np.random.rand() > 0.5
        thickness = np.random.randint(1, 3)
        val = np.random.uniform(0.5, 1.0)
        if horizontal:
            cv2.line(img, (x, y), (min(x + length, size - 1), y), val, thickness)
        else:
            cv2.line(img, (x, y), (x, min(y + length, size - 1)), val, thickness)
        if np.random.rand() > 0.7:
            r = np.random.randint(2, 5)
            cv2.circle(img, (x, y), r, val, -1)
    return img


def _wafer_illumination_shading(size):
    """Smooth radial illumination gradient, like uneven SEM/optical lighting."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size * np.random.uniform(0.3, 0.7), size * np.random.uniform(0.3, 0.7)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / dist.max()
    shading = 1.0 - 0.35 * dist
    return shading.astype(np.float32)


def generate_clean_image(size=128, seed=None):
    """Procedurally generate one clean synthetic semiconductor-die image."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)

    spacing = int(rng.integers(16, 32))
    img = _draw_die_grid(img, size, spacing)
    img = _draw_circuit_traces(img, size, n_traces=int(rng.integers(15, 35)))

    # base substrate reflectance + illumination shading
    base = rng.uniform(0.15, 0.25)
    shading = _wafer_illumination_shading(size)
    img = np.clip(img + base, 0, 1) * shading

    # mild smoothing so traces look like real deposited metal, not raster lines
    img = cv2.GaussianBlur(img, (3, 3), 0.4)
    img = np.clip(img, 0, 1).astype(np.float32)
    return img


def _downsample_upsample(img, rng, min_factor=1.5, max_factor=4.0):
    """
    Simulate resolution loss the way the hackathon slides show it: shrink by
    a random factor then resize back to the original grid with a soft
    (area/linear) interpolation, so fine detail is genuinely destroyed
    rather than just blurred. Kept at the original H x W so the rest of the
    pipeline and the fixed-size network don't need to change shape.
    """
    h, w = img.shape
    factor = rng.uniform(min_factor, max_factor)
    small_h, small_w = max(4, int(h / factor)), max(4, int(w / factor))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back.astype(np.float32)


def degrade_image(clean, rng=None):
    """
    Apply the hackathon's degradation function f: x (clean) -> y (corrupt):
    down-sampling + optical blur/contrast + additive Gaussian noise +
    speckle noise (+ optional salt & pepper for extra realism).

    NOTE: the returned image is intentionally NOT clipped to [0, 1] -- the
    hackathon spec explicitly allows the corrupted image to exceed that
    range. Only the clean ground truth is guaranteed to be in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    img = clean.copy()

    # 1) resolution loss / down-sampling (per hackathon f)
    if rng.random() < 0.9:
        img = _downsample_upsample(img, rng, min_factor=1.5, max_factor=4.0)

    # 2) optical defocus blur
    if rng.random() < 0.9:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.8, 2.2))

    # 3) low contrast / uneven brightness
    if rng.random() < 0.8:
        gain = rng.uniform(0.5, 0.85)
        bias = rng.uniform(-0.05, 0.15)
        img = img * gain + bias

    # 4) additive Gaussian (sensor/shot) noise -- per hackathon f
    if rng.random() < 0.95:
        sigma = rng.uniform(0.03, 0.12)
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)

    # 5) speckle (multiplicative) noise -- per hackathon f
    if rng.random() < 0.85:
        sigma = rng.uniform(0.05, 0.25)
        speckle = rng.normal(0, sigma, img.shape).astype(np.float32)
        img = img + img * speckle

    # 6) dead / hot pixels (salt & pepper) -- extra realism, not in the
    #    hackathon's f, but harmless and matches real fab sensor defects
    if rng.random() < 0.5:
        prob = rng.uniform(0.002, 0.02)
        mask = rng.random(img.shape)
        img[mask < prob / 2] = 0.0
        img[mask > 1 - prob / 2] = 1.0

    # Intentionally NOT clipped to [0,1] -- see module docstring.
    return img.astype(np.float32)


def degrade_image_lr(clean, rng=None, upscale_factor=2):
    """
    Same degradation stack as degrade_image(), but for training SIRNetSR
    (the true super-resolution architecture): instead of downsampling and
    then resizing back to the original grid, this stops after the ONE
    downsample step -- the returned array is genuinely smaller
    (H/upscale_factor x W/upscale_factor), matching how the real NoisyLR
    files relate to GT. Blur/contrast/noise are applied at full resolution
    first (so noise statistics look right), then downsampled once at the end.
    """
    if rng is None:
        rng = np.random.default_rng()
    img = clean.copy()

    if rng.random() < 0.9:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), rng.uniform(0.8, 2.2))

    if rng.random() < 0.8:
        gain = rng.uniform(0.5, 0.85)
        bias = rng.uniform(-0.05, 0.15)
        img = img * gain + bias

    if rng.random() < 0.95:
        sigma = rng.uniform(0.03, 0.12)
        img = img + rng.normal(0, sigma, img.shape).astype(np.float32)

    if rng.random() < 0.85:
        sigma = rng.uniform(0.05, 0.25)
        speckle = rng.normal(0, sigma, img.shape).astype(np.float32)
        img = img + img * speckle

    if rng.random() < 0.5:
        prob = rng.uniform(0.002, 0.02)
        mask = rng.random(img.shape)
        img[mask < prob / 2] = 0.0
        img[mask > 1 - prob / 2] = 1.0

    h, w = img.shape
    small_h, small_w = max(4, h // upscale_factor), max(4, w // upscale_factor)
    img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32)


def _augment_pair(a, b, rng):
    """Apply the SAME random flip/90-rotation to two arrays independently.
    Flips and 90-degree rotations commute with resizing, so this is safe to
    apply to a GT/NoisyLR pair even when they're at different resolutions --
    applying the same transform to each natively gives the same result as
    transforming-then-resizing."""
    if rng.random() < 0.5:
        a, b = np.fliplr(a).copy(), np.fliplr(b).copy()
    if rng.random() < 0.5:
        a, b = np.flipud(a).copy(), np.flipud(b).copy()
    k = int(rng.integers(0, 4))
    if k:
        a, b = np.rot90(a, k).copy(), np.rot90(b, k).copy()
    return a, b


class SemiconductorRestorationDataset(Dataset):
    """
    Paired (degraded, clean) synthetic semiconductor-image dataset.

    length:  number of samples generated on the fly per epoch
    size:    image side length in pixels (square), of the CLEAN/GT image
    seed_offset: lets train/val splits use disjoint random seeds
    lr_mode: if True, the degraded image is returned at its genuinely
        smaller native resolution (size / upscale_factor) instead of being
        resized back up -- use this when training SIRNetSR, which expects a
        real low-res input. Default False matches the original SIRNet
        (pre-upsampled, same-size input/output).
    upscale_factor: only used when lr_mode=True.
    """

    def __init__(self, length=400, size=128, seed_offset=0, lr_mode=False, upscale_factor=2):
        self.length = length
        self.size = size
        self.seed_offset = seed_offset
        self.lr_mode = lr_mode
        self.upscale_factor = upscale_factor

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seed = idx + self.seed_offset
        clean = generate_clean_image(self.size, seed=seed)
        rng = np.random.default_rng(seed * 7919 + 13)
        if self.lr_mode:
            degraded = degrade_image_lr(clean, rng=rng, upscale_factor=self.upscale_factor)
        else:
            degraded = degrade_image(clean, rng=rng)

        clean_t = torch.from_numpy(clean).unsqueeze(0)       # [1,H,W], always in [0,1]
        degraded_t = torch.from_numpy(degraded).unsqueeze(0)  # [1,h,w] (h,w = H,W or H/f,W/f), may exceed [0,1]
        return degraded_t, clean_t


class NpyPairedRestorationDataset(Dataset):
    """
    Loads the OFFICIAL hackathon dataset: paired (GT, NoisyLR) .npy files,
    matched by filename (NOT by index -- the numbering has gaps, e.g.
    000040, 000041, 000054... so a file only counts if the SAME filename
    exists in both the GT/ and NoisyLR/ folders).

    Expected layout (matches what the hackathon dataset download gives you):
        train/
            GT/
                000040.npy
                000041.npy
                ...
            NoisyLR/
                000040.npy
                000041.npy
                ...

    Each .npy file is a single grayscale image array (H, W), already
    normalized by the organizers -- GT is guaranteed [0,1], NoisyLR may
    exceed that range (per the webinar: "it is a feature not a bug"). No
    synthetic degradation is applied here since these ARE the real pairs.

    IMPORTANT: in the real dataset, GT and NoisyLR are genuinely different
    resolutions (e.g. GT 256x256, NoisyLR 128x128) -- the down-sampling in
    the hackathon's degradation function f is real, not simulated.

    pre_upsample: True (default) reproduces the original SIRNet behavior --
        NoisyLR is bicubic-resized up to GT's resolution here, so the
        network sees matching input/output sizes. Set to False when
        training SIRNetSR, which takes the genuine low-res input directly
        and does its own learned upsampling internally.
    augment: if True (default), applies a random flip/90-rotation to each
        pair (same transform to both GT and NoisyLR) -- cheap extra
        training variety given a comparatively small real dataset.

    Pass `filenames` to restrict this instance to a subset (e.g. for a
    train/val split) -- use `list_paired_filenames()` + a manual split, or
    `make_train_val_split()` below, rather than instantiating this twice
    over the same full folder.
    """

    def __init__(self, gt_dir, noisy_dir, filenames=None, pre_upsample=True, augment=True):
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.pre_upsample = pre_upsample
        self.augment = augment
        self.filenames = filenames if filenames is not None else list_paired_filenames(gt_dir, noisy_dir)
        if not self.filenames:
            raise FileNotFoundError(
                f"No matching .npy filenames found in both {gt_dir} and {noisy_dir}"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        import os
        fname = self.filenames[idx]
        clean = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        degraded = np.load(os.path.join(self.noisy_dir, fname)).astype(np.float32)

        # Defensive squeeze in case a file was saved with a stray channel
        # dim, e.g. shape (1,H,W) or (H,W,1) instead of plain (H,W).
        clean = np.squeeze(clean)
        degraded = np.squeeze(degraded)

        if self.augment:
            rng = np.random.default_rng()
            clean, degraded = _augment_pair(clean, degraded, rng)

        # Real dataset: NoisyLR is genuinely lower-resolution than GT.
        if self.pre_upsample and degraded.shape != clean.shape:
            h, w = clean.shape
            degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_CUBIC)

        clean_t = torch.from_numpy(clean.copy()).unsqueeze(0)       # [1,H,W]
        degraded_t = torch.from_numpy(degraded.copy()).unsqueeze(0)  # [1,h,w]
        return degraded_t, clean_t


class NpyTestDataset(Dataset):
    """
    Loads a held-out TEST folder that has NoisyLR .npy files only (no GT) --
    e.g. the hackathon's `Test_NoisyLR/` folder used for submission. Returns
    (degraded_tensor, filename) instead of (degraded, clean) since there's
    no ground truth to pair against.

    Since there's no GT here to read the target resolution from, pass
    `upscale_factor` to match whatever ratio you saw between GT and
    NoisyLR in the training set (e.g. GT 256x256 vs NoisyLR 128x128 ->
    upscale_factor=2). Set it to 1 if your test files are already at the
    same resolution SIRNet was trained on.
    """

    def __init__(self, noisy_dir, upscale_factor=2):
        import os
        self.noisy_dir = noisy_dir
        self.upscale_factor = upscale_factor
        self.filenames = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".npy"))
        if not self.filenames:
            raise FileNotFoundError(f"No .npy files found in {noisy_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        import os
        fname = self.filenames[idx]
        degraded = np.load(os.path.join(self.noisy_dir, fname)).astype(np.float32)
        degraded = np.squeeze(degraded)
        if self.upscale_factor != 1:
            h, w = degraded.shape
            degraded = cv2.resize(
                degraded, (w * self.upscale_factor, h * self.upscale_factor),
                interpolation=cv2.INTER_CUBIC,
            )
        degraded_t = torch.from_numpy(degraded).unsqueeze(0)
        return degraded_t, fname


def list_paired_filenames(gt_dir, noisy_dir):
    """Filenames present in BOTH gt_dir and noisy_dir, sorted for determinism."""
    import os
    gt_files = {f for f in os.listdir(gt_dir) if f.endswith(".npy")}
    noisy_files = {f for f in os.listdir(noisy_dir) if f.endswith(".npy")}
    return sorted(gt_files & noisy_files)


def make_train_val_split(gt_dir, noisy_dir, val_fraction=0.1, seed=42, pre_upsample=True, augment=True):
    """
    Splits the official dataset into train/val NpyPairedRestorationDataset
    instances, matched by filename and shuffled deterministically (same
    seed -> same split every run, so results are comparable across epochs).
    Augmentation is only applied to the TRAIN split -- validation should
    stay a fixed, un-augmented set for comparable metrics across runs.
    """
    filenames = list_paired_filenames(gt_dir, noisy_dir)
    rng = np.random.default_rng(seed)
    shuffled = list(filenames)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]
    train_ds = NpyPairedRestorationDataset(gt_dir, noisy_dir, filenames=train_files,
                                            pre_upsample=pre_upsample, augment=augment)
    val_ds = NpyPairedRestorationDataset(gt_dir, noisy_dir, filenames=val_files,
                                          pre_upsample=pre_upsample, augment=False)
    return train_ds, val_ds


class RealImageRestorationDataset(Dataset):
    """
    Paired (degraded, clean) dataset built from the user's OWN real image(s).

    Works even with a single image: each __getitem__ call takes a random
    crop of size `size` from a randomly chosen source image, applies a
    random flip/rotation for variety, then runs it through the same
    `degrade_image()` pipeline used for the synthetic data to build the
    (degraded, clean) training pair. This is why `length` (the number of
    samples "per epoch") can be set much higher than the number of source
    images -- every call produces a fresh crop + augmentation + degradation.

    image_dir: folder containing your image(s) (.png/.jpg/.jpeg/.bmp/.tif)
    length:    number of (random) samples to draw per epoch
    size:      training crop size in pixels (square)
    seed_offset: lets train/val splits draw different random crops
    """

    def __init__(self, image_dir, length=400, size=128, seed_offset=0):
        import os
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.paths = [
            os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(exts)
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        self.images = []
        for p in self.paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0
            # if the source image is smaller than our crop size, upscale it
            h, w = img.shape
            if h < size or w < size:
                scale = size / min(h, w) * 1.05
                img = cv2.resize(img, (int(w * scale) + 1, int(h * scale) + 1))
            self.images.append(img)

        if not self.images:
            raise FileNotFoundError(f"Found files but none were readable as images: {self.paths}")

        self.length = length
        self.size = size
        self.seed_offset = seed_offset

    def __len__(self):
        return self.length

    def _random_crop(self, img, rng):
        h, w = img.shape
        y = int(rng.integers(0, h - self.size + 1))
        x = int(rng.integers(0, w - self.size + 1))
        return img[y:y + self.size, x:x + self.size]

    def __getitem__(self, idx):
        seed = idx + self.seed_offset
        rng = np.random.default_rng(seed)

        img = self.images[int(rng.integers(0, len(self.images)))]
        crop = self._random_crop(img, rng).copy()

        # augmentation for variety, since we may only have 1-2 source images
        if rng.random() < 0.5:
            crop = np.fliplr(crop).copy()
        if rng.random() < 0.5:
            crop = np.flipud(crop).copy()
        k = int(rng.integers(0, 4))
        if k:
            crop = np.rot90(crop, k).copy()

        clean = np.clip(crop, 0, 1).astype(np.float32)
        degraded = degrade_image(clean, rng=np.random.default_rng(seed * 7919 + 13))

        clean_t = torch.from_numpy(clean).unsqueeze(0)
        degraded_t = torch.from_numpy(degraded).unsqueeze(0)
        return degraded_t, clean_t
