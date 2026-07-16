"""Standalone degradation-removal sweep (no proprietary import). For each of the 35 clean GT
images: render, apply the 9 deterministic corruptions, run the engine, and carry each through
the clean-room decision layer to a redaction box; report removal recall (geometric coverage of
the GT region, threshold 0.50) per condition. Reproduces the Fig. 2 / Table S1 removal rows.

--engine florence uses Florence-2-base (this file). EasyOCR/Tesseract removal use the same
decision layer on their detections; run them in their own environments with the analogous
detection front-end (see run_engine_detection.py) feeding decision.select_redaction."""
import os, sys, json, argparse
import numpy as np
from PIL import Image
import torch
import pydicom
from transformers import AutoProcessor, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__))
import phi_classifier as cr
import decision as dc
from degrade import degrade, CONDS

OCR_TASK = "<OCR_WITH_REGION>"
GEN = dict(max_new_tokens=512, num_beams=1, do_sample=False, use_cache=False, early_stopping=False)
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


def meta_of(ds):
    m = {"manufacturer": str(ds.get("Manufacturer", "")), "modality": str(ds.get("Modality", "")),
         "dicom_phi_values": {}}
    for tag in cr.DICOM_PHI_TAGS:
        v = ds.get(tag, "")
        if v not in (None, ""):
            m["dicom_phi_values"][tag] = str(v)
    m["dicom_phi_values"]["PatientName"] = str(ds.get("PatientName", ""))
    m["dicom_phi_values"]["PatientID"] = str(ds.get("PatientID", ""))
    return m


def florence_regions(model, processor, device, img):
    inputs = processor(text=OCR_TASK, images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                             pad_token_id=processor.tokenizer.pad_token_id, **GEN)
    txt = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(txt, task=OCR_TASK, image_size=(img.width, img.height))
    regs = []
    for box, label in zip(parsed.get(OCR_TASK, {}).get('quad_boxes', []),
                          parsed.get(OCR_TASK, {}).get('labels', [])):
        xs = box[0::2]; ys = box[1::2]
        regs.append({"text": label, "bbox": [[min(xs), min(ys)], [max(xs), min(ys)],
                                             [max(xs), max(ys)], [min(xs), max(ys)]]})
    return regs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--engine", default="florence", choices=["florence"])
    ap.add_argument("--model", default="microsoft/Florence-2-base")
    args = ap.parse_args()
    gt = json.load(open(args.gt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device).eval()
    is_tech = lambda t: cr._is_technical_term(t)
    N = len(gt)
    rem = {c: 0 for c in CONDS}
    for i, (sop, info) in enumerate(gt.items()):
        ds = pydicom.dcmread(info["file"], force=True)
        base = to_display(ds.pixel_array, ds)
        g = gbox(info); m = meta_of(ds); w, h = base.width, base.height
        for c in CONDS:
            regs = florence_regions(model, processor, device, degrade(base, c))
            boxes, _ = dc.select_redaction(regs, m, w, h, cr.sensitivity_reason, is_tech, margin_active=True)
            rem[c] += int(cov(g, boxes) >= THR)
        print(f"[{i+1}/{N}] done", flush=True)
    print("\n=== STANDALONE %s degradation removal (/%d) ===" % (args.engine, N))
    for c in CONDS:
        print(f"  {c:15s}: {rem[c]}/{N}")
    print("  (paper Florence removal: flat 31 to 33 across all conditions)")


if __name__ == "__main__":
    main()
