"""Reviewer-response analysis (2026-09-01). Consumes rev2_capture.py output and produces the
three quantities the reviews asked for and the original submission did not report.

1. Transcription accuracy under degradation (reviewer 2, major comment 2). Coverage alone
   cannot distinguish reading from hallucinating, so we compute character and word error
   rates of the text Florence-2 actually returns over each ground-truth region, per
   corruption condition. Protocol: for a ground-truth box G, concatenate the text of every
   detected region whose box intersects G, ordered top-to-bottom then left-to-right, strip
   Florence special tokens, upper-case, collapse whitespace, and compare to the answer-key
   string normalized the same way. CER and WER are Levenshtein distance over characters and
   over whitespace-delimited tokens, normalized by reference length, capped at 1.0.

2. Decision-layer precision (reviewer 1, decision-layer comment). Over every detected region
   on the clean images, how many of those each signal flags actually fall on labelled PHI.
   Note the interpretation: MIDI-B labels only the injected PHI, so legitimate non-PHI text
   that we redact counts as a false positive here. That is the same convention as the
   scorer's pixels_retained penalty, and it is the conservative reading.

3. Removal recall per condition, recomputed through the production decision functions, so the
   published sweep is reproduced from a fresh capture rather than cited.

Run in the florence-anon env after rev2_capture.py.
"""
import argparse
import io
import json
import math
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
from degrade_conditions import CONDS

THR = 0.50
_SPECIAL = re.compile(r"</?s>")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1))


def lev(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_SEXMARK = re.compile(r"\[[A-Za-z]\]")


def norm(s):
    """Normalize exactly as the CBIIT scorer normalizes the answer-key string before its own
    text comparison (curation_validator.validate_pixels_hidden): newlines to spaces, the
    literal 'DOB:' label removed, and single-letter bracketed sex markers removed. Florence
    special tokens are stripped and case and whitespace are folded on top of that, so the
    reference and hypothesis are compared on the same footing."""
    s = _SPECIAL.sub(" ", s or "")
    s = s.replace("\n", " ").replace("DOB:", " ")
    s = _SEXMARK.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().upper()


def gbox(info):
    """Ground-truth box from a gate_recall.json record: the answer key stores one
    pixels_hidden action per region with top_left / bottom_right insertion coordinates."""
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


def intersects(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def reg(text, box):
    x0, y0, x1, y1 = box
    return {"text": text, "bbox": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "bbox_type": "quad"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gate = json.load(open(args.gate))
    FL = json.load(open(args.regions))
    recs = {r["sop"]: r for r in gate[args.split] if r.get("file")}

    with redirect_stdout(io.StringIO()):
        anon = DicomAnonymizer(device="cpu")
    assert anon.phi_detector is not None and anon.phi_guidance is not None, "INIT FAIL"
    pdet = anon.phi_detector

    ctx = {}
    for sop, r in recs.items():
        ds = pydicom.dcmread(r["file"], stop_before_pixels=True, force=True)
        with redirect_stdout(io.StringIO()):
            meta = anon._extract_metadata(ds, str(ds.get("PatientID", "")), str(ds.get("PatientName", "")))
        meta["modality"] = r["modality"]
        ctx[sop] = {"g": gbox(r), "meta": meta, "modality": r["modality"],
                    "W": r["Columns"], "H": r["Rows"],
                    "gt_text": norm(" ".join(b["text"] for b in r["gt"]))}

    out = {"transcription": [], "removal": [], "precision": {}}

    # ---- 1 + 3: per condition -------------------------------------------------------
    for c in CONDS:
        cers, wers, n_read = [], [], 0
        rem = det = 0
        for sop, cx in ctx.items():
            regs = FL.get(c, {}).get(sop, [])
            raw = [r["box"] for r in regs]
            det += int(cov(cx["g"], raw) >= THR)
            trs = [reg(r["text"], r["box"]) for r in regs]
            with redirect_stdout(io.StringIO()):
                boxes, _dec = anon._select_redaction_boxes(trs, cx["meta"], cx["W"], cx["H"])
            rem += int(cov(cx["g"], boxes) >= THR)

            hit = [r for r in regs if intersects(r["box"], cx["g"])]
            hit.sort(key=lambda r: (r["box"][1], r["box"][0]))
            hyp = norm(" ".join(r["text"] for r in hit))
            ref = cx["gt_text"]
            if not ref:
                continue
            n_read += 1 if hyp else 0
            cers.append(min(1.0, lev(hyp, ref) / max(1, len(ref))))
            rw, hw = ref.split(), hyp.split()
            wers.append(min(1.0, lev(hw, rw) / max(1, len(rw))))
        out["transcription"].append({
            "condition": c, "n": len(cers),
            "regions_with_any_text": n_read,
            "CER_mean": round(float(np.mean(cers)), 4),
            "CER_median": round(float(np.median(cers)), 4),
            "WER_mean": round(float(np.mean(wers)), 4),
            "WER_median": round(float(np.median(wers)), 4)})
        lo, hi = wilson(rem, len(ctx))
        out["removal"].append({"condition": c, "detection_k": det, "removal_k": rem,
                               "n": len(ctx), "wilson_lo": lo, "wilson_hi": hi})
        print("[%-14s] det %2d  rem %2d  CER %.3f  WER %.3f  (text on %d/%d GT regions)"
              % (c, det, rem, out["transcription"][-1]["CER_mean"],
                 out["transcription"][-1]["WER_mean"], n_read, len(cers)), flush=True)

    # ---- 2: decision-layer precision on clean images ---------------------------------
    tally = {k: {"flagged": 0, "tp": 0} for k in ("by_text", "by_spatial", "by_margin", "union")}
    total_regions = 0
    for sop, cx in ctx.items():
        regs = FL.get("clean", {}).get(sop, [])
        total_regions += len(regs)
        trs = [reg(r["text"], r["box"]) for r in regs]
        with redirect_stdout(io.StringIO()):
            _b, dec = anon._select_redaction_boxes(trs, cx["meta"], cx["W"], cx["H"])
        for r, d in zip(regs, dec):
            tp = intersects(r["box"], cx["g"])
            for key, flag in (("by_text", d["by_text"]), ("by_spatial", d["by_spatial"]),
                              ("by_margin", d["by_margin"]), ("union", d["redacted"])):
                if flag:
                    tally[key]["flagged"] += 1
                    tally[key]["tp"] += int(tp)
    for k, v in tally.items():
        f, t = v["flagged"], v["tp"]
        v["precision"] = round(t / f, 4) if f else None
        v["wilson"] = wilson(t, f) if f else None
    out["precision"] = {"total_detected_regions": total_regions,
                        "n_instances": len(ctx), "signals": tally}
    print("\n=== decision-layer precision on clean images (%d regions over %d instances) ==="
          % (total_regions, len(ctx)))
    for k, v in tally.items():
        print("  %-11s flagged %4d  on labelled PHI %3d  precision %s"
              % (k, v["flagged"], v["tp"], v["precision"]))

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "rev2_analysis.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
