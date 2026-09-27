"""Reviewer-response capture (2026-09-01). Regenerates the per-engine detection captures that
the original revision harness produced, after the session scratchpad holding them was lost to
cleanup. Renders each ground-truth burned-in instance through the PRODUCTION display renderer,
writes the nine deterministic corruptions to PNG so every engine sees byte-identical input,
then runs Florence-2 and records (text, box) per detected region.

Adds what the original capture did not keep: the recognized TEXT is retained per condition so
character/word error rates can be computed against the answer-key strings (reviewer 2, major
comment 2), and detection boxes are retained so classifier precision against the ground-truth
boxes can be computed (reviewer 1, decision-layer comment).

Inputs : gate_recall.json  (SOP -> resolved input path + answer-key boxes/text), produced by
         rev2_gate_recall.py
Outputs: <out>/degrade_pngs/<suffix>__<cond>.png
         <out>/florence_regions.json  {cond: {sop: [{text, box}]}}

Run in the florence-anon env. Asserts model + classifier + spatial prior are initialized
before any measurement, the safeguard adopted after an early diagnostic silently skipped
model initialization and produced a spurious result.
"""
import argparse
import io
import json
import os
import sys
import time
import warnings
from contextlib import redirect_stdout

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pydicom
import torch
from PIL import Image

from DicomLibrary import DicomAnonymizer
from degrade_conditions import CONDS, degrade


def to_display(arr, ds):
    """Production display rendering (mirrors DicomAnonymizer._to_display_image)."""
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    a = np.asarray(arr)
    if samples == 3:
        if a.ndim == 4:
            a = a[a.shape[0] // 2]
    else:
        if a.ndim == 3:
            a = a[a.shape[0] // 2]
        if "MONOCHROME1" in str(getattr(ds, "PhotometricInterpretation", "")):
            a = a.max() - a
    if a.dtype != np.uint8:
        lo, hi = float(a.min()), float(a.max())
        a = ((a - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)
    return Image.fromarray(a).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True, help="gate_recall.json from rev2_gate_recall.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--conds", default="all", help="'all' or comma-separated condition names")
    args = ap.parse_args()

    conds = CONDS if args.conds == "all" else args.conds.split(",")
    os.makedirs(os.path.join(args.out, "degrade_pngs"), exist_ok=True)
    gate = json.load(open(args.gate))
    recs = [r for r in gate[args.split] if r.get("file")]

    with redirect_stdout(io.StringIO()):
        anon = DicomAnonymizer(device="cuda")
        anon._ensure_model_loaded()
    assert anon.model is not None and anon.processor is not None, "INIT FAIL: model not loaded"
    assert anon.phi_detector is not None, "INIT FAIL: classifier not loaded"
    assert anon.phi_guidance is not None, "INIT FAIL: spatial prior not loaded"
    print("[init] model + classifier + spatial prior OK | cuda=%s" % torch.cuda.is_available(), flush=True)

    out = {c: {} for c in conds}
    t0 = time.time()
    for i, r in enumerate(recs, 1):
        ds = pydicom.dcmread(r["file"], force=True)
        base = to_display(ds.pixel_array, ds)
        suf = r["sop"][-16:]
        for c in conds:
            img = degrade(base, c)
            img.save(os.path.join(args.out, "degrade_pngs", "%s__%s.png" % (suf, c)))
            with redirect_stdout(io.StringIO()):
                res = anon._run_florence_analysis(img)
            regs = []
            for reg in res.get("text_regions", []):
                xs, ys = anon._extract_bbox_coords(reg.get("bbox", []))
                if xs and ys:
                    regs.append({"text": reg.get("text", ""),
                                 "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]})
            out[c][r["sop"]] = regs
        print("[%d/%d] %s %s  (%.0fs)" % (i, len(recs), r["modality"], suf, time.time() - t0), flush=True)

    path = os.path.join(args.out, "florence_regions.json")
    json.dump(out, open(path, "w"), indent=1)
    print("\n[done] %d instances x %d conditions -> %s (%.0fs)"
          % (len(recs), len(conds), path, time.time() - t0))


if __name__ == "__main__":
    main()
