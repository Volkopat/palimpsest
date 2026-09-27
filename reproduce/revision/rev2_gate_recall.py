"""Policy-gate recall over every ground-truth burned-in instance (reviewer 1: "the paper would
benefit from evidence that the policy gate itself is safe").

The pixel path is reached only through `should_pixel_pass`, so a gate false negative is an
unrecoverable leak that no downstream layer can correct. This script reads the CBIIT answer
key, selects every instance carrying a `pixels_hidden` action, resolves it to its input DICOM,
and runs the production gate over the input header.

Also emits, per instance, the answer-key burned-in boxes and text, the BurnedInAnnotation value
actually present in the input, and device/geometry fields. That output file is the input to
rev2_capture.py and rev2_analyze.py, so this script runs first.

Usage (florence-anon env):
    python rev2_gate_recall.py --root <Datasets/MIDI-B> --out <dir>
"""
import argparse
import collections
import io
import json
import math
import os
import sys
import warnings
from contextlib import redirect_stdout

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import sqlite3

import pydicom

from DicomLibrary import DicomAnonymizer

SALT = b"midi-b-validation-v1"
SPLITS = {"val": ("Validation", "MIDI-B-Synthetic-Validation"),
          "test": ("Test", "MIDI-B-Synthetic-Test")}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Datasets/MIDI-B containing _answer_keys and the synthetic trees")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with redirect_stdout(io.StringIO()):
        anon = DicomAnonymizer(device="cpu", salt=SALT)
    assert anon.phi_detector is not None, "INIT FAIL: pipeline not constructed"

    out = {}
    for split, (keydir, tree) in SPLITS.items():
        db = os.path.join(args.root, "_answer_keys", keydir, "MIDI-B-Answer-Key-%s.db" % keydir)
        cur = sqlite3.connect(db).cursor()
        cols = [d[1] for d in cur.execute("PRAGMA table_info(answer_data)")]
        recs = []
        for row in cur.execute("SELECT * FROM answer_data"):
            d = dict(zip(cols, row))
            ad = json.loads(d["AnswerData"])
            hits = [v for v in ad.values() if v.get("action") == "<pixels_hidden>"]
            if hits:
                recs.append((d, hits))
        out[split] = []
        for d, hits in recs:
            pid, st = str(d["PatientID"]), str(d["StudyInstanceUID"])
            se, sop = str(d["SeriesInstanceUID"]), str(d["SOPInstanceUID"])
            sdir = os.path.join(args.root, tree, pid, st, se)
            path = None
            if os.path.isdir(sdir):
                for fn in sorted(os.listdir(sdir)):
                    p = os.path.join(sdir, fn)
                    try:
                        h = pydicom.dcmread(p, specific_tags=[0x00080018],
                                            stop_before_pixels=True, force=True)
                        if str(h.get("SOPInstanceUID", "")) == sop:
                            path = p
                            break
                    except Exception:
                        pass
            rec = {"split": split, "sop": sop, "modality": str(d["Modality"]), "patient": pid,
                   "study": st, "series": se, "file": path, "n_boxes": len(hits)}
            if path:
                hdr = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                rec.update(gate=bool(anon.should_pixel_pass(hdr)),
                           BurnedInAnnotation=str(hdr.get("BurnedInAnnotation", "") or ""),
                           SOPClassUID=str(hdr.get("SOPClassUID", "") or ""),
                           BitsAllocated=int(hdr.get("BitsAllocated", 0) or 0),
                           Manufacturer=str(hdr.get("Manufacturer", "") or ""),
                           Model=str(hdr.get("ManufacturerModelName", "") or ""),
                           Rows=int(hdr.get("Rows", 0) or 0),
                           Columns=int(hdr.get("Columns", 0) or 0))
                texts = []
                for h in hits:
                    try:
                        texts.append(json.loads(h["action_text"].strip("<>")))
                    except Exception:
                        pass
                rec["gt"] = texts
            out[split].append(rec)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "gate_recall.json")
    json.dump(out, open(path, "w"), indent=1)

    tf = tg = 0
    for split in out:
        found = [x for x in out[split] if x.get("file")]
        gated = [x for x in found if x.get("gate")]
        tf += len(found)
        tg += len(gated)
        bc = collections.Counter(x["BurnedInAnnotation"] or "<empty>" for x in found)
        print("[%s] burned-in instances %d, resolved %d, GATE FIRED %d/%d"
              % (split, len(out[split]), len(found), len(gated), len(found)))
        print("     modality: %s" % dict(collections.Counter(x["modality"] for x in found)))
        print("     BurnedInAnnotation: %s" % dict(bc))
        miss = [x for x in found if not x.get("gate")]
        print("     NOT GATED: %s" % ([(x["modality"], x["sop"][-12:]) for x in miss] or "none"))
    print("\nCOMBINED GATE RECALL %d/%d  Wilson 95%% CI %s" % (tg, tf, wilson(tg, tf)))
    print("wrote", path)


if __name__ == "__main__":
    main()
