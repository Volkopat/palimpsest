"""DICOM IOD conformance check of the de-identified output (reviewer 2, minor comment:
"No DICOM conformance check is reported, for a system that rewrites pixel data and emits
compressed input uncompressed").

Runs the dicom3tools `dciodvfy` validator, which the CBIIT harness already bundles, over a
deterministic random sample of de-identified instances from each profile, and aggregates the
Error and Warning classes. Reports strict and retain separately, because the interesting
question is whether over-redaction under the Safe-Harbor profile costs IOD conformance.

Usage (florence-anon env, Windows):
    python rev2_conformance.py --root <_deid_out> --runs val_margin_strict val_full_retain \
        --n 200 --out <dir>
"""
import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DCIODVFY = os.path.join(REPO, "MIDI-B", "MIDI_validation_script", "software",
                        "dicom3tools_winexe_1.00.snapshot.20250128115421", "dciodvfy.exe")

# Normalize a message to its class by stripping the instance-specific element/module/value.
_STRIP = [
    (re.compile(r"Element=<[^>]*>"), "Element=<..>"),
    (re.compile(r"Module=<[^>]*>"), "Module=<..>"),
    (re.compile(r"\(0x[0-9a-fA-F]{4},0x[0-9a-fA-F]{4}\)[^-]*"), "(tag) "),
    (re.compile(r"\s+"), " "),
]


def classify(line):
    s = line.strip()
    for rx, rep in _STRIP:
        s = rx.sub(rep, s)
    return s.strip()


def run_one(path):
    # dciodvfy is invoked with cwd set to its own directory (it needs the cygwin DLLs
    # beside it), so the file argument MUST be absolute. Passing a relative path makes
    # every call fail to open the file and silently report zero errors, which is the
    # "harness produced a clean result because it never ran" failure this project
    # explicitly guards against.
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None, [("Runner", "input missing")]
    try:
        p = subprocess.run([DCIODVFY, path], capture_output=True, timeout=120,
                           cwd=os.path.dirname(DCIODVFY))
    except Exception as e:
        return None, [("Runner", "harness error: %s" % type(e).__name__)]
    txt = (p.stderr or b"").decode("utf-8", "replace") + (p.stdout or b"").decode("utf-8", "replace")
    out = []
    iod = None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Error"):
            out.append(("Error", classify(line)))
        elif line.startswith("Warning"):
            out.append(("Warning", classify(line)))
        elif line.startswith("Abort"):
            out.append(("Error", "Abort: " + classify(line)[:60]))
        elif iod is None:
            iod = line
    # Guard: a run that produced no output at all did not actually validate anything.
    if not out and iod is None:
        return None, [("Runner", "no validator output")]
    return iod, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(DCIODVFY):
        sys.exit("dciodvfy not found at %s" % DCIODVFY)

    summary = {}
    for run in args.runs:
        base = os.path.join(args.root, run)
        files = []
        for dirpath, _d, names in os.walk(base):
            for n in names:
                if n.lower().endswith(".dcm"):
                    files.append(os.path.join(dirpath, n))
        files.sort()
        rng = random.Random(args.seed)
        sample = files if len(files) <= args.n else rng.sample(files, args.n)
        sample.sort()
        print("[%s] %d files, sampling %d" % (run, len(files), len(sample)), flush=True)

        errs, warns, iods = collections.Counter(), collections.Counter(), collections.Counter()
        clean = 0
        for i, f in enumerate(sample, 1):
            iod, msgs = run_one(f)
            iods[iod or "?"] += 1
            e = [m for k, m in msgs if k == "Error"]
            w = [m for k, m in msgs if k == "Warning"]
            errs.update(e)
            warns.update(w)
            if not e:
                clean += 1
            if i % 50 == 0:
                print("   %d/%d" % (i, len(sample)), flush=True)
        summary[run] = {"n_total": len(files), "n_sampled": len(sample),
                        "instances_error_free": clean,
                        "iods": iods.most_common(),
                        "errors": errs.most_common(),
                        "warnings": warns.most_common(20)}
        print("[%s] error-free instances: %d/%d" % (run, clean, len(sample)), flush=True)
        for m, c in errs.most_common(10):
            print("    ERROR x%-5d %s" % (c, m[:110]), flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "conformance.json")
    json.dump(summary, open(path, "w"), indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
