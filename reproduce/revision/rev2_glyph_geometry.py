"""Quantify the rendering of the burned-in text (reviewer 1: "unclear how the PHI varies in
location, font, orientation etc.").

Location and size are reported in Supplementary Table S2. This script measures the remaining
two properties directly from the pixels rather than asserting them: the polarity and contrast
of the rendering, the glyph height, an estimate of stroke width, and the skew angle of the text
baseline. Skew is estimated by the standard projection-profile method, rotating the binarized
crop over a range of angles and taking the angle that maximizes the variance of the horizontal
projection, which peaks when text lines are horizontal.

Everything is computed on the clean 8-bit display render inside the answer-key insertion box,
which is the same image the OCR sees.

Run in the florence-anon env after rev2_capture.py.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def otsu(g):
    hist, _ = np.histogram(g, bins=256, range=(0, 256))
    total = g.size
    sum_all = np.dot(np.arange(256), hist)
    sb, wB, best, thr = 0.0, 0.0, -1.0, 127
    sumB = 0.0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        sb = wB * wF * (mB - mF) ** 2
        if sb > best:
            best, thr = sb, i
    return thr


def rotate(img, deg):
    return np.asarray(Image.fromarray((img * 255).astype(np.uint8)).rotate(
        deg, resample=Image.BILINEAR, fillcolor=0)) > 127


def skew_angle(fg, span=8.0, step=0.5):
    best_a, best_v = 0.0, -1.0
    for a in np.arange(-span, span + 1e-9, step):
        r = rotate(fg, a) if abs(a) > 1e-9 else fg
        prof = r.sum(axis=1).astype(float)
        v = float(np.var(prof))
        if v > best_v:
            best_v, best_a = v, float(a)
    return best_a


def stroke_width(fg):
    """Mean run length of foreground pixels along rows, a standard cheap stroke-width proxy."""
    runs = []
    for row in fg:
        n = 0
        for v in row:
            if v:
                n += 1
            elif n:
                runs.append(n)
                n = 0
        if n:
            runs.append(n)
    return float(np.median(runs)) if runs else float("nan")


def glyph_height(fg):
    """Height of the tallest contiguous band of rows containing foreground, i.e. one text line."""
    rows = fg.any(axis=1)
    best = cur = 0
    for v in rows:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--pngs", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gate = json.load(open(args.gate))
    rows = []
    for r in gate[args.split]:
        if not r.get("file"):
            continue
        png = os.path.join(args.pngs, "%s__clean.png" % r["sop"][-16:])
        if not os.path.exists(png):
            continue
        b = r["gt"][0]
        x0, y0 = int(b["top_left"][0]), int(b["top_left"][1])
        x1, y1 = int(b["bottom_right"][0]), int(b["bottom_right"][1])
        g = np.asarray(Image.open(png).convert("L").crop((min(x0, x1), min(y0, y1),
                                                          max(x0, x1), max(y0, y1))), dtype=np.uint8)
        if g.size < 16:
            continue
        t = otsu(g)
        bright = g > t
        # polarity: text is the minority class
        frac = bright.mean()
        polarity = "light-on-dark" if frac < 0.5 else "dark-on-light"
        fg = bright if frac < 0.5 else ~bright
        mean_fg = float(g[fg].mean()) if fg.any() else float("nan")
        mean_bg = float(g[~fg].mean()) if (~fg).any() else float("nan")
        rows.append({
            "sop": r["sop"], "modality": r["modality"],
            "kind": "name_block" if "\n" in b["text"] else "short_marker",
            "polarity": polarity,
            "fg_fraction": round(float(fg.mean()), 4),
            "michelson_contrast": round(abs(mean_fg - mean_bg) / max(1.0, mean_fg + mean_bg), 3),
            "glyph_line_height_px": glyph_height(fg),
            "stroke_width_px": round(stroke_width(fg), 2),
            "skew_deg": skew_angle(fg),
        })

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "glyph_geometry.json")
    json.dump(rows, open(p, "w"), indent=1)

    import collections
    print("=== rendering of the burned-in text, measured on the clean display render (n=%d) ===" % len(rows))
    print("polarity:", dict(collections.Counter(x["polarity"] for x in rows)))
    for key, unit in (("michelson_contrast", ""), ("glyph_line_height_px", " px"),
                      ("stroke_width_px", " px"), ("skew_deg", " deg")):
        v = np.array([x[key] for x in rows], dtype=float)
        print("%-22s min %6.2f  median %6.2f  max %6.2f%s"
              % (key, np.nanmin(v), np.nanmedian(v), np.nanmax(v), unit))
    sk = np.array([abs(x["skew_deg"]) for x in rows])
    print("regions with |skew| <= 0.5 deg: %d/%d" % (int((sk <= 0.5).sum()), len(sk)))
    for kind in ("name_block", "short_marker"):
        k = [x for x in rows if x["kind"] == kind]
        if k:
            h = np.array([x["glyph_line_height_px"] for x in k], dtype=float)
            print("  %-13s n=%2d  glyph line height %d to %d px" % (kind, len(k), h.min(), h.max()))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
