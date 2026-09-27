"""The complementary arm of the controlled experiment (reviewer 2, major comment 1: "only one
arm of the experiment is run ... either run the second arm, or scope the claim").

The published experiment holds the decision layer fixed and swaps the perceptual engine. This
script does the reverse: it holds Florence-2's detections fixed and swaps the decision layer,
across five layers spanning the achievable range.

  null              redact nothing                            (floor)
  generic-PII       an independent generic PII rule set, of the family an off-the-shelf
                    PII detector implements (person-name shapes, dates, SSN, long numeric
                    IDs, phone, email). Written against generic PII categories, NOT against
                    this benchmark's marker strings, so it is a fair substitute rather than a
                    re-tuned copy of ours.
  ours-by_text      the shipped pattern classifier alone
  ours-union        the shipped additive union (classifier OR spatial OR margin)
  redact-all        redact every detected region                (ceiling)

The ceiling row is the load-bearing one: it is the maximum recall any decision layer could
reach from these detections, so the gap between it and a given layer is that layer's true
headroom. Also sweeps the coverage threshold, since the published 0.50 was queried as
permissive.

Run in the florence-anon env after rev2_capture.py.
"""
import argparse
import io
import json
import os
import re
import sys
import warnings
from contextlib import redirect_stdout

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pydicom

from DicomLibrary import DicomAnonymizer

_SPECIAL = re.compile(r"</?s>")

# ---- an independent generic PII rule set -------------------------------------------------
# Deliberately written from generic PII categories (the sort Presidio ships recognizers for),
# without reference to the benchmark's burned-in strings.
_PII = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("DATE", re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b|\b(?:19|20)\d{2}[01]\d[0-3]\d\b")),
    ("LONG_ID", re.compile(r"\b\d{6,}\b")),
    ("DOB_LABEL", re.compile(r"\b(?:dob|d\.o\.b|date of birth)\b", re.I)),
    ("MRN_LABEL", re.compile(r"\b(?:mrn|medical record|patient id|accession)\b", re.I)),
    ("PERSON", re.compile(r"\b[A-Z][a-z]{1,}\s+[A-Z][a-z]{1,}\b|\b[A-Z]{2,}\s*,\s*[A-Z]{2,}\b")),
]


def generic_pii(text):
    t = _SPECIAL.sub("", text or "").strip()
    if len(t) < 2:
        return False
    return any(rx.search(t) for _n, rx in _PII)


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


def reg(text, box):
    x0, y0, x1, y1 = box
    return {"text": text, "bbox": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "bbox_type": "quad"}


def intersects(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--cond", default="clean")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gate = json.load(open(args.gate))
    FL = json.load(open(args.regions))[args.cond]
    recs = {r["sop"]: r for r in gate[args.split] if r.get("file")}

    with redirect_stdout(io.StringIO()):
        anon = DicomAnonymizer(device="cpu")
    assert anon.phi_detector is not None and anon.phi_guidance is not None, "INIT FAIL"

    ctx = {}
    for sop, r in recs.items():
        ds = pydicom.dcmread(r["file"], stop_before_pixels=True, force=True)
        with redirect_stdout(io.StringIO()):
            meta = anon._extract_metadata(ds, str(ds.get("PatientID", "")), str(ds.get("PatientName", "")))
        meta["modality"] = r["modality"]
        ctx[sop] = {"g": gbox(r), "meta": meta, "W": r["Columns"], "H": r["Rows"]}

    THRS = [0.30, 0.50, 0.70, 0.90]
    LAYERS = ["null", "generic-PII", "ours-by_text", "ours-union", "redact-all"]
    hits = {L: {t: 0 for t in THRS} for L in LAYERS}
    prec = {L: {"flagged": 0, "tp": 0} for L in LAYERS}

    for sop, cx in ctx.items():
        regs = FL.get(sop, [])
        trs = [reg(r["text"], r["box"]) for r in regs]
        with redirect_stdout(io.StringIO()):
            union_boxes, dec = anon._select_redaction_boxes(trs, cx["meta"], cx["W"], cx["H"])
        sel = {
            "null": [],
            "generic-PII": [r["box"] for r in regs if generic_pii(r["text"])],
            "ours-by_text": [regs[i]["box"] for i in range(len(regs)) if dec[i]["by_text"]],
            "ours-union": union_boxes,
            "redact-all": [r["box"] for r in regs],
        }
        for L, boxes in sel.items():
            c = cov(cx["g"], boxes)
            for t in THRS:
                hits[L][t] += int(c >= t)
            for b in boxes:
                prec[L]["flagged"] += 1
                prec[L]["tp"] += int(intersects(b, cx["g"]))

    N = len(ctx)
    print("=== reverse arm: detections FIXED (Florence-2, %s), decision layer VARIED, N=%d ===" % (args.cond, N))
    print("%-14s %s   precision" % ("decision layer", "  ".join("cov>=%.2f" % t for t in THRS)))
    for L in LAYERS:
        row = "  ".join("%5d/%d" % (hits[L][t], N) for t in THRS)
        p = prec[L]
        pr = ("%.3f (%d/%d)" % (p["tp"] / p["flagged"], p["tp"], p["flagged"])) if p["flagged"] else "n/a"
        print("%-14s %s   %s" % (L, row, pr))

    out = {"condition": args.cond, "n": N, "thresholds": THRS,
           "recall": {L: {str(t): hits[L][t] for t in THRS} for L in LAYERS},
           "precision": prec}
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "reverse_arm.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
