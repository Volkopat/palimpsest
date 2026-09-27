"""Does EasyOCR's collapse under heavy blur survive parameter tuning? (reviewer 2, major
comment 4: "EasyOCR exposes mag_ratio, text_threshold and contrast parameters that materially
change behaviour on blurred input, and none are reported. If EasyOCR ran at defaults while
Florence-2 was used as intended, the collapse at blur r4 partly measures configuration rather
than architecture.")

Runs EasyOCR over the identical saved PNGs used for the published sweep, under its default
configuration and under configurations tuned specifically for blurred and low-contrast input,
and scores detection recall by the same geometric coverage against the answer-key insertion
coordinates. The default arm is included so the tuned arms are read against a reproduction of
the published number rather than against the published number itself.

Run in the midi-scorer env (which holds easyocr and pydicom 2.2.1).
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

THR = 0.50

ALL_CONFIGS = {
    "default": dict(),
    "tuned-sensitive": dict(text_threshold=0.5, low_text=0.3, link_threshold=0.3),
    "tuned-mag2": dict(mag_ratio=2.0, text_threshold=0.5, low_text=0.3, link_threshold=0.3),
    "tuned-mag2-contrast": dict(mag_ratio=2.0, text_threshold=0.5, low_text=0.3,
                                link_threshold=0.3, contrast_ths=0.3, adjust_contrast=0.7),
    "tuned-mag3": dict(mag_ratio=3.0, text_threshold=0.4, low_text=0.25, link_threshold=0.25),
}
# Magnification costs roughly the square of mag_ratio on CPU, so the full grid is expensive.
# --configs selects a subset; the decisive comparison is default vs tuned-mag2 at blur r4.
CONFIGS = dict(ALL_CONFIGS)


def gbox(info):
    b = info["gt"][0]
    tl, br = b["top_left"], b["bottom_right"]
    return [min(int(tl[0]), int(br[0])), min(int(tl[1]), int(br[1])),
            max(int(tl[0]), int(br[0])), max(int(tl[1]), int(br[1]))]


def cov(g, boxes, grid=200):
    if not boxes:
        return 0.0
    xs = np.linspace(g[0], g[2], grid)
    ys = np.linspace(g[1], g[3], grid)
    XX, YY = np.meshgrid(xs, ys)
    c = np.zeros(XX.shape, bool)
    for b in boxes:
        c |= (XX >= b[0]) & (XX <= b[2]) & (YY >= b[1]) & (YY <= b[3])
    return float(c.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--pngs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default="clean,blur_4")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--configs", default="all", help="'all' or comma-separated config names")
    ap.add_argument("--outfile", default="easyocr_tuned.json")
    ap.add_argument("--dump-regions", default=None,
                    help="also write per-region text+box captures here, for the removal sweep")
    args = ap.parse_args()

    global CONFIGS
    if args.configs != "all":
        CONFIGS = {k: ALL_CONFIGS[k] for k in args.configs.split(",")}

    import easyocr
    import torch
    print("[init] easyocr loaded, cuda=%s" % torch.cuda.is_available(), flush=True)
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    assert reader is not None, "INIT FAIL: easyocr reader not constructed"

    gate = json.load(open(args.gate))
    recs = [r for r in gate[args.split] if r.get("file")]
    conds = args.conds.split(",")

    results = {}
    regions = {}
    t0 = time.time()
    for cond in conds:
        for cname, kw in CONFIGS.items():
            hit = 0
            ndet = 0
            for r in recs:
                png = os.path.join(args.pngs, "%s__%s.png" % (r["sop"][-16:], cond))
                img = np.array(Image.open(png).convert("RGB"))
                boxes, texts = [], []
                for pts, _txt, conf in reader.readtext(img, detail=1, **kw):
                    if conf < args.conf:
                        continue
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    boxes.append([min(xs), min(ys), max(xs), max(ys)])
                    texts.append(_txt)
                ndet += len(boxes)
                hit += int(cov(gbox(r), boxes) >= THR)
                if args.dump_regions:
                    regions.setdefault(cond, {}).setdefault(cname, {})[r["sop"]] = [
                        {"text": t, "box": [int(b[0]), int(b[1]), int(b[2]), int(b[3])]}
                        for t, b in zip(texts, boxes)]
            results["%s/%s" % (cond, cname)] = {"detection_k": hit, "n": len(recs),
                                                "total_regions": ndet, "config": kw}
            print("[%-7s] %-20s detection %2d/%d  (%d regions, %.0fs)"
                  % (cond, cname, hit, len(recs), ndet, time.time() - t0), flush=True)
            # Persist after every configuration. Each arm costs minutes on CPU, so a run that
            # has to be stopped early should still leave usable partial results behind.
            os.makedirs(args.out, exist_ok=True)
            json.dump(results, open(os.path.join(args.out, args.outfile), "w"), indent=1)

    if args.dump_regions:
        json.dump(regions, open(os.path.join(args.out, args.dump_regions), "w"))
        print("wrote regions ->", os.path.join(args.out, args.dump_regions))
    print("\nwrote", os.path.join(args.out, args.outfile))


if __name__ == "__main__":
    main()
