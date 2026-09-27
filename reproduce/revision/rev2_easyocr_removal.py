"""Removal recall for the tuned EasyOCR arm.

Supplementary Table S10 reports *detection* recall for the tuned configurations, but the
pipeline is detection-gated, so the number that matters for de-identification is how many
ground-truth regions survive the decision layer. Table S1's EasyOCR removal column is
default-config only. This pushes the tuned engine's captured boxes through the same shipped
decision layer used everywhere else, so the tuned engine is compared to Florence-2 on removal
and not only on detection.

Reads the region dump written by rev2_easyocr_tuned.py --dump-regions.
Run in the florence-anon env.
"""
import argparse
import io
import json
import os
import sys
import warnings
from contextlib import redirect_stdout

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pydicom

from DicomLibrary import DicomAnonymizer

THR = 0.50


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--regions", required=True, help="easyocr_best_regions.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gate = json.load(open(args.gate))
    dump = json.load(open(args.regions))
    recs = {r["sop"]: r for r in gate[args.split] if r.get("file")}

    with redirect_stdout(io.StringIO()):
        anon = DicomAnonymizer(device="cpu")
    assert anon.phi_detector is not None and anon.phi_guidance is not None, "INIT FAIL"

    ctx = {}
    for sop, r in recs.items():
        ds = pydicom.dcmread(r["file"], stop_before_pixels=True, force=True)
        with redirect_stdout(io.StringIO()):
            meta = anon._extract_metadata(ds, str(ds.get("PatientID", "")),
                                          str(ds.get("PatientName", "")))
        meta["modality"] = r["modality"]
        ctx[sop] = {"g": gbox(r), "meta": meta, "W": r["Columns"], "H": r["Rows"]}

    out = {}
    for cond, bycfg in dump.items():
        for cname, persop in bycfg.items():
            det = rem = 0
            for sop, cx in ctx.items():
                regs = persop.get(sop, [])
                det += int(cov(cx["g"], [r["box"] for r in regs]) >= THR)
                trs = [reg(r["text"], r["box"]) for r in regs]
                with redirect_stdout(io.StringIO()):
                    boxes, _dec = anon._select_redaction_boxes(trs, cx["meta"], cx["W"], cx["H"])
                rem += int(cov(cx["g"], boxes) >= THR)
            key = "%s/%s" % (cond, cname)
            out[key] = {"detection_k": det, "removal_k": rem, "n": len(ctx)}
            print("[%-7s] %-20s detection %2d/%d  removal %2d/%d"
                  % (cond, cname, det, len(ctx), rem, len(ctx)), flush=True)

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "easyocr_tuned_removal.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
