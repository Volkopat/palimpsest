"""Deterministic image corruptions for the degradation sweep (seed 0), shared by the capture
scripts so every engine sees identical input. Applied to the 8-bit display render, matching
the published methodology. Nine conditions: clean, blur r2/r4 (Gaussian blur radius in px),
noise 20/40 (additive Gaussian sigma on a 0..255 scale), contr 0.5/0.3 (contrast factor),
down 0.5/0.33 (bilinear downscale factor, then upsample back to the original size).

Identical to reproduce/degrade.py in the released harness; duplicated here so the internal
revision scripts have no dependency on the released tree.
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CONDS = ["clean", "blur_2", "blur_4", "noise_20", "noise_40",
         "contrast_0.5", "contrast_0.3", "downscale_0.5", "downscale_0.33"]


def degrade(img, cond):
    if cond == "clean":
        return img
    if cond.startswith("blur"):
        return img.filter(ImageFilter.GaussianBlur(radius=float(cond.split("_")[1])))
    if cond.startswith("noise"):
        s = float(cond.split("_")[1])
        a = np.asarray(img).astype(np.float32)
        a = np.clip(a + np.random.default_rng(0).normal(0, s, a.shape), 0, 255)
        return Image.fromarray(a.astype(np.uint8))
    if cond.startswith("contrast"):
        return ImageEnhance.Contrast(img).enhance(float(cond.split("_")[1]))
    if cond.startswith("downscale"):
        f = float(cond.split("_")[1])
        w, h = img.size
        return img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return img
