"""
checkpoint_utils.py
--------------------
Single place that knows how to build the right model architecture from a
checkpoint's saved run_config, and whether that checkpoint expects its
input pre-upsampled (SIRNet) or wants the genuine low-res input directly
(SIRNetSR). Every script that loads a checkpoint (train.py --resume,
evaluate.py, inference.py, webapp/app.py) goes through this so they can't
disagree about what a checkpoint means.
"""

import torch

from src.model import build_model


def load_checkpoint(path, device=None):
    """
    Loads a checkpoint file and returns (model, run_config, checkpoint_dict).
    The model is already .to(device) and .eval()'d.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    run_config = ckpt.get("run_config", {})

    arch = run_config.get("arch", "unet")
    upscale_factor = run_config.get("upscale_factor", 2)

    model = build_model(arch=arch, upscale_factor=upscale_factor).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, run_config, ckpt


def needs_pre_upsample(run_config):
    """True if this checkpoint's architecture expects the input already
    resized to output resolution before it reaches the model (SIRNet),
    False if the model does its own learned upsampling (SIRNetSR)."""
    return run_config.get("arch", "unet") == "unet"
