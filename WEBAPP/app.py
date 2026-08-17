"""
webapp/app.py
-------------
A small local Flask app for trying SIRNet interactively: pick a checkpoint,
upload a degraded image (either the official .npy format, or a regular
.png/.jpg for convenience), see the restored output side by side.

This is a DEMO tool, not part of the mandatory hackathon submission
(inference.py is the required standalone script for that) -- it's here to
let you visually sanity-check models the same way you did for the
wafer-defect-detection Flask app.

Usage:
    cd webapp
    python3 app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import sys
import time
import uuid

import numpy as np
import cv2
import torch
from flask import Flask, request, render_template, url_for

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.checkpoint_utils import load_checkpoint, needs_pre_upsample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
WEIGHTS_DIR = os.path.join(BASE_DIR, "..", "weights")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB upload limit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache of loaded models, keyed by checkpoint filename -- so switching
# checkpoints in the UI doesn't reload from disk every single request.
_model_cache = {}


def list_checkpoints():
    """All .pt files in weights/, most recently modified first."""
    if not os.path.isdir(WEIGHTS_DIR):
        return []
    files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pt")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(WEIGHTS_DIR, f)), reverse=True)
    return files


def get_model(ckpt_name):
    """Load (or fetch from cache) the model + metadata for a given checkpoint filename."""
    if ckpt_name in _model_cache:
        return _model_cache[ckpt_name]

    ckpt_path = os.path.join(WEIGHTS_DIR, ckpt_name)
    m, run_config, ckpt = load_checkpoint(ckpt_path, device=device)

    info = {
        "name": ckpt_name,
        "epoch": ckpt.get("epoch", "?"),
        "val_psnr": ckpt.get("val_psnr", None),
        "data_source": run_config.get("data_source", "unknown"),
        "arch": run_config.get("arch", "unet"),
        "upscale_factor": run_config.get("upscale_factor", 2),
        "pre_upsample": needs_pre_upsample(run_config),
    }
    _model_cache[ckpt_name] = (m, info)
    return m, info


def load_input_array(filepath):
    """Load either a .npy array or a regular image file into a (H,W) float32
    array in roughly [0,1] (for images) or whatever range the .npy has
    (may exceed [0,1] for real NoisyLR files, per spec)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".npy":
        arr = np.load(filepath).astype(np.float32)
        arr = np.squeeze(arr)
        return arr, "npy"
    else:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image file.")
        arr = img.astype(np.float32) / 255.0
        return arr, "image"


def array_to_png(arr, path):
    """Save a float array (any range) as a viewable PNG, clipped to [0,1] for display."""
    disp = np.clip(arr, 0, 1)
    disp_u8 = (disp * 255).astype(np.uint8)
    cv2.imwrite(path, disp_u8)


@app.route("/", methods=["GET"])
def index():
    checkpoints = list_checkpoints()
    selected = request.args.get("ckpt") or (checkpoints[0] if checkpoints else None)
    ckpt_info = None
    error = None
    if selected:
        try:
            _, ckpt_info = get_model(selected)
        except Exception as e:
            error = f"Could not load checkpoint {selected}: {e}"
    return render_template("index.html", checkpoints=checkpoints, selected=selected,
                            ckpt_info=ckpt_info, result=None, error=error)


@app.route("/predict", methods=["POST"])
def predict():
    checkpoints = list_checkpoints()
    selected = request.form.get("ckpt")

    # Self-heal: if the submitted checkpoint is missing/invalid but there
    # IS a valid checkpoint available, fall back to the most recent one
    # rather than hard-failing -- avoids a confusing dead-end if the hidden
    # form field ever comes through empty (e.g. browser quirks, stale page).
    if selected not in checkpoints:
        if checkpoints:
            selected = checkpoints[0]
        else:
            return render_template("index.html", checkpoints=checkpoints, selected=None,
                                    ckpt_info=None, result=None,
                                    error="No checkpoints found in weights/ -- train a model first.")

    try:
        model, ckpt_info = get_model(selected)
    except Exception as e:
        return render_template("index.html", checkpoints=checkpoints, selected=selected,
                                ckpt_info=None, result=None, error=f"Could not load checkpoint: {e}")

    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", checkpoints=checkpoints, selected=selected,
                                ckpt_info=ckpt_info, result=None, error="Please choose a file.")

    upscale_factor = int(request.form.get("upscale_factor", 2))

    uid = uuid.uuid4().hex[:8]
    ext = os.path.splitext(file.filename)[1].lower()
    upload_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    file.save(upload_path)

    try:
        arr, kind = load_input_array(upload_path)
    except Exception as e:
        return render_template("index.html", checkpoints=checkpoints, selected=selected,
                                ckpt_info=ckpt_info, result=None, error=f"Could not read file: {e}")

    t0 = time.time()

    # Pre-upsample if this looks like a lower-resolution NoisyLR array AND
    # the loaded checkpoint's architecture expects that (SIRNet). SIRNetSR
    # checkpoints take the native low-res input directly and do their own
    # learned upsampling internally -- pre-upsampling would double it up.
    proc = arr.copy()
    if kind == "npy" and upscale_factor != 1 and ckpt_info.get("pre_upsample", True):
        h, w = proc.shape
        proc = cv2.resize(proc, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_CUBIC)

    input_tensor = torch.from_numpy(proc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        restored = model(input_tensor).cpu().squeeze(0).squeeze(0).numpy()

    elapsed = time.time() - t0

    input_png = f"{uid}_input.png"
    restored_png = f"{uid}_restored.png"
    array_to_png(proc, os.path.join(RESULT_DIR, input_png))
    array_to_png(restored, os.path.join(RESULT_DIR, restored_png))

    result = {
        "input_url": url_for("static", filename=f"results/{input_png}"),
        "restored_url": url_for("static", filename=f"results/{restored_png}"),
        "shape_in": f"{arr.shape[0]}\u00d7{arr.shape[1]}",
        "shape_out": f"{restored.shape[0]}\u00d7{restored.shape[1]}",
        "value_range_in": f"[{arr.min():.3f}, {arr.max():.3f}]",
        "elapsed_ms": f"{elapsed*1000:.1f}",
        "kind": kind,
    }
    return render_template("index.html", checkpoints=checkpoints, selected=selected,
                            ckpt_info=ckpt_info, result=result, error=None)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
