"""
model.py
--------
SIRNet: Semiconductor Image Restoration Network.

Two architectures are provided:

- SIRNet: a lightweight U-Net-style encoder/decoder with skip connections,
  operating at a FIXED resolution -- it expects NoisyLR to already be
  bicubic-upsampled to GT resolution before it sees it (a "pre-upsample"
  restoration baseline). This is the original/default architecture.

- SIRNetSR: takes the genuinely lower-resolution NoisyLR input DIRECTLY and
  learns the upsampling itself via PixelShuffle, jointly with restoration.
  See its docstring below for the rationale.

Encoder-decoder + skip connections were chosen over a plain denoising
residual net (e.g. DnCNN) because the degradation pipeline includes blur and
resolution loss, not just additive noise -- skip-connected multi-scale
features handle that combined restoration task better.

Residual learning: per the hackathon problem statement, the corrupted input
image is NOT guaranteed to lie in [0,1] (noise can push it slightly outside),
while the ground truth always does. Rather than squashing the output with a
sigmoid -- which assumes a bounded, well-behaved input -- both networks
instead predict a *correction* on top of the input and the result is
clamped to [0,1] to match the GT range. This residual formulation is
standard in image-restoration literature (the network only has to learn
"what changed", not reconstruct the whole image from scratch) and is more
robust to the input distribution shift called out in the webinar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SIRNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.bottleneck = conv_block(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        # No activation here -- this predicts a residual correction, not a
        # final pixel value, so it must be free to be positive or negative.
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # The raw degraded input can exceed [0,1] (see module docstring).
        # Clamp only what we feed the conv stack's numerics through
        # BatchNorm at a sane scale; the *skip* (identity) path below still
        # uses the original unclamped input so extreme pixels are visible
        # to the correction instead of being silently discarded.
        x_in = torch.clamp(x, -1.0, 2.0)

        e1 = self.enc1(x_in)            # [B,base,H,W]
        e2 = self.enc2(self.pool(e1))   # [B,base*2,H/2,W/2]
        b = self.bottleneck(self.pool(e2))  # [B,base*4,H/4,W/4]

        d2 = self.up2(b)                       # [B,base*2,H/2,W/2]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)                      # [B,base,H,W]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        residual = self.out_conv(d1)
        restored = torch.clamp(x + residual, 0.0, 1.0)  # final output in [0,1], matching GT
        return restored


class SIRNetSR(nn.Module):
    """
    Super-resolution variant of SIRNet: takes the genuinely lower-resolution
    NoisyLR input DIRECTLY (no bicubic pre-upsampling step outside the
    network) and learns the upsampling itself via PixelShuffle, jointly with
    denoising/deblurring.

    Why this exists: the plain SIRNet above operates on a NoisyLR that's
    already been bicubic-resized up to GT resolution before it ever reaches
    the network -- the network only ever denoises/deblurs, it never learns
    to upsample. That fixed, non-learned step is a bottleneck. Here, the
    encoder/decoder body runs at the LR resolution (cheaper too), and a
    PixelShuffle head produces the upsampled residual directly. A bicubic
    upsample of the raw input still forms the skip/base for residual
    learning (same [0,1]-clamped-output rationale as SIRNet), so training
    stays as stable as the original -- only the correction is learned
    end-to-end at full resolution instead of being handed a fixed baseline.

    upscale_factor must match the true GT:NoisyLR resolution ratio (e.g. 2
    for GT 256x256 vs NoisyLR 128x128).
    """

    def __init__(self, base=32, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.bottleneck = conv_block(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        # PixelShuffle upsampling head: base channels at LR resolution ->
        # base*(factor^2) channels -> PixelShuffle rearranges those extra
        # channels into spatial resolution, giving a learned upscale
        # instead of a fixed bicubic one.
        self.pre_shuffle = nn.Conv2d(base, base * (upscale_factor ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.post_shuffle = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 3, padding=1),
        )

    def forward(self, x):
        # x: [B,1,H,W] at NATIVE (low) resolution, may exceed [0,1].
        x_in = torch.clamp(x, -1.0, 2.0)

        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))          # [B,base,H,W]

        up_feat = self.pixel_shuffle(self.pre_shuffle(d1))   # [B,base,H*f,W*f]
        residual = self.post_shuffle(up_feat)                 # [B,1,H*f,W*f]

        x_up = F.interpolate(
            x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )
        restored = torch.clamp(x_up + residual, 0.0, 1.0)
        return restored


def build_model(arch="unet", upscale_factor=2, base=32):
    """Single place that knows how to construct either architecture, so
    train/evaluate/inference/webapp all agree on what a config means."""
    if arch == "unet":
        return SIRNet(base=base)
    elif arch == "unet_sr":
        return SIRNetSR(base=base, upscale_factor=upscale_factor)
    else:
        raise ValueError(f"Unknown arch '{arch}' (expected 'unet' or 'unet_sr')")


if __name__ == "__main__":
    for arch in ["unet", "unet_sr"]:
        model = build_model(arch, upscale_factor=2)
        if arch == "unet":
            x = torch.randn(2, 1, 128, 128)
        else:
            x = torch.randn(2, 1, 64, 64)
        y = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{arch}: input {tuple(x.shape)} -> output {tuple(y.shape)}, {n_params:,} params")
