"""
inference.py -- everything the demo needs to turn an uploaded image into a prediction.
No Streamlit code in here, so you can also import it from a notebook to test.
Matches the final models (Version 15, LFAB v4: one learnable gain per frequency bin).
"""
import copy
import io
import os
import time
from itertools import combinations
from math import factorial

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps

from model_def import ProposedAIDetector, make_radial_band_masks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Filenames written by Notebook 2 (Model A) and Notebook 3 (Model B).
CKPT_NAMES = {
    "Model A (MobileNetV3-Small + LFAB)": ("mobilenet", "mobilenet_lfab_model.pth"),
    "Model B (EfficientNet-B0 + LFAB)": ("efficientnet", "efficientnet_lfab_model.pth"),
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Same as Notebook 4's eval_transform (images are resized to 256x256 first, see prepare_image).
EVAL_TRANSFORM = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def load_models(weights_dir):
    """Load whichever checkpoints exist in weights_dir. Returns ({label: model}, [missing labels]).
    Each model is warmed up once so the first real image is not slowed by one-time start-up costs."""
    models, missing = {}, []
    for label, (backbone, fname) in CKPT_NAMES.items():
        path = os.path.join(weights_dir, fname)
        if not os.path.isfile(path):
            missing.append(f"{label}  ->  {path}")
            continue
        m = ProposedAIDetector(embedding_dim=128, pretrained=False, backbone_type=backbone)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m = m.to(DEVICE).eval()
        with torch.no_grad():
            m(torch.zeros(1, 3, 224, 224, device=DEVICE))
        models[label] = m
    return models, missing


def benchmark_latency(model, runs=30, warmup=5):
    """Average forward-pass time in ms for ONE 224x224 image (batch size 1) on this machine.
    Same idea as the thesis latency test (it averaged 100 runs on a Tesla T4 GPU); fewer runs here so
    the demo starts quickly. Covers the model only, not image loading or preprocessing."""
    x = torch.randn(1, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _sync()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        _sync()
    return (time.perf_counter() - t0) / runs * 1000.0


def prepare_image(image_bytes, force_jpeg=True, already_processed=False):
    """Mimic Notebook 1's preprocessing so the demo sees what the models saw in training:
    RGB -> 256x256 bilinear resize -> JPEG re-save at PIL's default quality (75) ->
    then the 224 center crop + ImageNet normalisation done by EVAL_TRANSFORM.

    Notebook 1 saved each resized image under its original extension, and its dataset layout lists .jpg
    for every source, so the training images carry one round of JPEG compression at 256x256.
    force_jpeg=True (default) applies that to any upload (PNG/WebP too). With force_jpeg=False only
    uploads that were already JPEG are re-saved.

    already_processed=True is for images copied straight out of Notebook 1's resized_dataset folder (the
    sample gallery). They already went through the resize and the JPEG save once, so nothing is repeated.

    EXIF orientation is applied first so phone photos are not sideways.

    Returns a dict with crop (224x224x3 uint8, the exact pixels the models see), tensor (1x3x224x224
    float32 numpy array) and orig_size (width, height of the upload).
    """
    img = Image.open(io.BytesIO(image_bytes))
    was_jpeg = (img.format or "").upper() in ("JPEG", "MPO")
    img = ImageOps.exif_transpose(img)
    orig_size = img.size
    img = img.convert("RGB")
    if already_processed:
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BILINEAR)
    else:
        img = img.resize((256, 256), Image.BILINEAR)
        if force_jpeg or was_jpeg:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")  # PIL default quality, same as Notebook 1's .save(dest)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
    tensor = EVAL_TRANSFORM(img).unsqueeze(0).numpy().astype(np.float32)
    crop = np.array(img.crop((16, 16, 240, 240)))
    return {"crop": crop, "tensor": tensor, "orig_size": orig_size}


def _to_tensor(tensor_np):
    return torch.from_numpy(np.asarray(tensor_np, dtype=np.float32)).to(DEVICE)


@torch.no_grad()
def predict_prob(model, tensor_np):
    """P(AI-generated) for one preprocessed image (sigmoid of the model's logit). Uncalibrated."""
    logits, _ = model(_to_tensor(tensor_np))
    return float(torch.sigmoid(logits).item())


class _LogitOnly(nn.Module):
    """Grad-CAM needs a model that returns a single tensor, not (logit, embedding)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)[0]


def gradcam_overlay(model, tensor_np, target_class=1):
    """Grad-CAM heatmap blended onto the 224x224 crop (uint8 RGB).
    Same target layer as Notebook 4: the last backbone block (a coarse 7x7 map).
    target_class=1 explains the 'AI-generated' output (red = evidence for AI). target_class=0 explains
    the 'human-made' direction. The notebooks used the model's predicted class."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    x = _to_tensor(tensor_np).requires_grad_(True)
    with torch.enable_grad():
        cam = GradCAM(model=_LogitOnly(model).eval(), target_layers=[model.backbone[-1]])
        heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(int(target_class))])[0]
    rgb = np.asarray(tensor_np)[0].transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    return show_cam_on_image(rgb, heat, use_rgb=True, image_weight=0.5)


# =============================================================================================
# SHAP
# =============================================================================================
BAND_LABELS = ["Band 0 (lowest freq.)", "Band 1", "Band 2", "Band 3", "Band 4", "Band 5 (highest freq.)"]


@torch.no_grad()
def lfab_band_shap(model, tensor_np):
    """Exact Shapley values of LFAB's 6 radial frequency bands for ONE image (LFAB v4).

    Same idea as Notebook 2's run_shap_lfab_bands: a band that is 'dropped' has the gain of every
    frequency bin inside it forced to 0 (gain_logits = -1e4 -> 2*sigmoid(-1e5) = 0). With only 6
    players there are 2^6 = 64 coalitions, so all are enumerated to get exact Shapley values instead
    of a sampled KernelExplainer estimate.
    The signed values add up to  P(AI | all bands kept) - P(AI | all bands dropped);
    positive = that band pushes the score toward 'AI-generated'.
    Note this measures how much the prediction depends on each band. It is not the same as removing
    LFAB (the thesis alpha = 0 check) and it does not show that any frequency proves an image is AI.

    Works on a copy of the model so the shared, cached model is never modified.
    Returns (phi[6], P(all bands kept), P(all bands dropped)).
    """
    m = copy.deepcopy(model).eval()
    lfab = m.lfab
    orig = lfab.gain_logits.detach().clone()
    n = lfab.num_bands
    H, W = lfab.H, lfab.W
    # Bands over the STORED half-grid (rows 0..H//2), exactly as in Notebook 2.
    band_masks = [mk[: H // 2 + 1, :].to(DEVICE)
                  for mk in make_radial_band_masks(H, W // 2 + 1, n, "cpu")]
    x = _to_tensor(tensor_np)

    def value(keep):
        masked = orig.clone()
        for b in range(n):
            if b not in keep:
                masked[:, band_masks[b]] = -1e4
        lfab.gain_logits.copy_(masked)
        return torch.sigmoid(m(x)[0]).item()

    cache = {}

    def v(keep):
        key = frozenset(keep)
        if key not in cache:
            cache[key] = value(key)
        return cache[key]

    phi = np.zeros(n)
    for i in range(n):
        others = [b for b in range(n) if b != i]
        for k in range(n):
            weight = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(others, k):
                phi[i] += weight * (v(set(S) | {i}) - v(set(S)))
    return phi, v(set(range(n))), v(set())


def lfab_gains(model):
    """Trained mean gain per radial band, low -> high frequency. 1.0 = neutral (gain range is 0..2)."""
    return model.lfab.band_gains().detach().cpu().numpy()


def lfab_gain_map(model):
    """(H, W//2+1) learned gain map averaged over channels, for an imshow next to the SHAP bars."""
    return model.lfab.gain_map().detach().cpu().numpy()


def pixel_shap(model, crop_uint8, max_evals=300, blur=128):
    """Pixel-region SHAP for the 'AI-generated' probability (Partition explainer + blur masker, like
    Notebook 2). crop_uint8: (224, 224, 3) uint8 array. Returns a (224, 224) array where positive values
    push toward 'AI-generated' and negative toward 'human'."""
    import shap

    mean, std = np.array(MEAN), np.array(STD)

    def f(imgs):
        x = imgs.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(DEVICE)
        with torch.no_grad():
            return torch.sigmoid(model(x)[0]).cpu().numpy().flatten()

    masker = shap.maskers.Image(f"blur({blur},{blur})", crop_uint8.shape)
    explainer = shap.Explainer(f, masker)
    sv = explainer(crop_uint8[None], max_evals=max_evals, batch_size=50)
    return sv.values[0].sum(axis=-1)  # sum the 3 colour channels -> one value per pixel
