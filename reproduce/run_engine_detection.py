"""Standalone detection reproduction for EasyOCR or Tesseract (no proprietary import).
Renders each clean GT DICOM to an 8-bit display image (paper's renderer) and runs the chosen
engine, scoring detection recall by geometric coverage of the GT insertion coordinates.
EasyOCR needs the `easyocr` package; Tesseract needs `pytesseract` + a Tesseract 5 binary."""
import os, sys, json, argparse
import numpy as np
from PIL import Image
import pydicom
THR = 0.50


def to_display(arr, ds):
    samples = int(getattr(ds, 'SamplesPerPixel', 1) or 1)
    a = np.asarray(arr)
    if samples == 3:
        if a.ndim == 4:
            a = a[a.shape[0] // 2]
    else:
        if a.ndim == 3:
            a = a[a.shape[0] // 2]
        if 'MONOCHROME1' in str(getattr(ds, 'PhotometricInterpretation', '')):
            a = a.max() - a
    if a.dtype != np.uint8:
        lo, hi = float(a.min()), float(a.max())
        a = ((a - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8)
    return Image.fromarray(a).convert('RGB')


def gbox(info):
    b = info["boxes"][0]
    return [min(b["tl"][0], b["br"][0]), min(b["tl"][1], b["br"][1]),
            max(b["tl"][0], b["br"][0]), max(b["tl"][1], b["br"][1])]


def cov(g, boxes, grid=200):
    if not boxes:
        return 0.0
    xs = np.linspace(g[0], g[2], grid); ys = np.linspace(g[1], g[3], grid)
    XX, YY = np.meshgrid(xs, ys); c = np.zeros(XX.shape, bool)
    for b in boxes:
        c |= (XX >= b[0]) & (XX <= b[2]) & (YY >= b[1]) & (YY <= b[3])
    return float(c.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--engine", required=True, choices=["easyocr", "tesseract"])
    ap.add_argument("--tesseract-cmd", default=None)
    ap.add_argument("--tessdata", default=None)
    args = ap.parse_args()
    gt = json.load(open(args.gt))

    if args.engine == "easyocr":
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True, verbose=False)

        def detect(img):
            out = []
            for pts, txt, conf in reader.readtext(np.array(img), detail=1):
                if conf < 0.2:
                    continue
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                out.append([min(xs), min(ys), max(xs), max(ys)])
            return out
    else:
        import pytesseract
        if args.tessdata:
            os.environ.setdefault("TESSDATA_PREFIX", args.tessdata)
        if args.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

        def detect(img):
            d = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
            out = []
            for i in range(len(d["text"])):
                if d["text"][i].strip():
                    try:
                        conf = float(d["conf"][i])
                    except Exception:
                        conf = -1
                    if conf > 20:
                        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                        out.append([x, y, x + w, y + h])
            return out

    N = len(gt); hits = 0
    for i, (sop, info) in enumerate(gt.items()):
        ds = pydicom.dcmread(info["file"], force=True)
        img = to_display(ds.pixel_array, ds)
        boxes = detect(img)
        hits += int(cov(gbox(info), boxes) >= THR)
        print(f"[{i+1}/{N}] {info['modality']} dets={len(boxes)}", flush=True)
    paper = {"easyocr": 33, "tesseract": 11}[args.engine]
    print(f"\n=== STANDALONE {args.engine} detection: {hits}/{N}   (paper {paper}) ===")


if __name__ == "__main__":
    main()
