"""Host memory footprint of the two passes (reviewer 1: the deployment claim "would want
inference time, throughput, hardware requirements, GPU/CPU memory usage, and perhaps DICOMs
processed per minute").

VRAM, latency and throughput were already measured. This adds the host-side figure: peak
resident set size of a single worker for the metadata pass, and for the gated pixel pass with
Florence-2 resident, so a deployer can size a worker pool.

Uses only the standard library plus the Windows working-set counter exposed through ctypes, so
it does not add a dependency on psutil.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import glob
import io
import os
import sys
import time
import warnings
from contextlib import redirect_stdout

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pydicom

from DicomLibrary import DicomAnonymizer

SALT = b"midi-b-validation-v1"


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def mem():
    """(current, peak) working set in MB for this process.

    Modern Windows exports the counter as kernel32!K32GetProcessMemoryInfo; the older
    psapi!GetProcessMemoryInfo entry point is not always resolvable, so try both.
    """
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p   # a 64-bit handle must not be truncated
    h = k32.GetCurrentProcess()
    for lib, name in ((k32, "K32GetProcessMemoryInfo"),
                      (ctypes.windll.psapi, "GetProcessMemoryInfo")):
        try:
            fn = getattr(lib, name)
        except AttributeError:
            continue
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), wt.DWORD]
        fn.restype = wt.BOOL
        if fn(h, ctypes.byref(c), c.cb):
            return (c.WorkingSetSize / 1048576.0, c.PeakWorkingSetSize / 1048576.0)
    return (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="a MIDI-B synthetic split")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pixel-n", type=int, default=8)
    args = ap.parse_args()

    print("baseline (interpreter only):        %6.1f MB current, %6.1f MB peak" % mem())

    files = sorted(glob.glob(os.path.join(args.input, "**", "*.dcm"), recursive=True))
    step = max(1, len(files) // args.n)
    sample = files[::step][:args.n]

    with redirect_stdout(io.StringIO()):
        anon = DicomAnonymizer(device="cpu", salt=SALT)
    print("after constructing the pipeline:    %6.1f MB current, %6.1f MB peak" % mem())

    t0 = time.time()
    for f in sample:
        ds = pydicom.dcmread(f, force=True)
        with redirect_stdout(io.StringIO()):
            anon.metadata_anonymization(ds)
    dt = time.time() - t0
    cur, peak = mem()
    print("after %d metadata instances:       %6.1f MB current, %6.1f MB peak" % (len(sample), cur, peak))
    print("   metadata: %.1f ms per instance, %.0f instances/s in one worker"
          % (1000 * dt / len(sample), len(sample) / dt))

    gated = []
    for f in files:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if anon.should_pixel_pass(h):
            gated.append(f)
        if len(gated) >= args.pixel_n:
            break
    if gated:
        with redirect_stdout(io.StringIO()):
            anon._ensure_model_loaded()
        print("after loading Florence-2 (CPU):     %6.1f MB current, %6.1f MB peak" % mem())
        t0 = time.time()
        for f in gated:
            ds = pydicom.dcmread(f, force=True)
            with redirect_stdout(io.StringIO()):
                anon.image_anonymization(ds)
        dt = time.time() - t0
        cur, peak = mem()
        print("after %d gated pixel instances:      %6.1f MB current, %6.1f MB peak" % (len(gated), cur, peak))
        print("   pixel pass on CPU: %.0f ms per image (GPU figure is 699 ms; CPU is the fallback path)"
              % (1000 * dt / len(gated)))
    else:
        print("no gated instances found in the sample")


if __name__ == "__main__":
    main()
