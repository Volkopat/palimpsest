"""Standalone (no proprietary import) reproduction of the Florence-2 side of the clean
detector-swap, from raw public DICOM: render an 8-bit display image exactly as the paper
describes, run Florence-2 <OCR_WITH_REGION>, then apply the clean-room classifier + spatial
fallback + margin. Reproduces detection / +classifier / spatial(fallback) / union / removal.

Dependencies: torch, transformers, pydicom, numpy, pillow (+ the flash_attn import shim that
Florence-2-base requires). Model weights: public microsoft/Florence-2-base."""
import os, sys, json, argparse
import numpy as np
from PIL import Image
import torch
import pydicom
from transformers import AutoProcessor, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__))
import phi_classifier as cr
import decision as dc

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
    data = parsed.get(OCR_TASK, {})
    for box, label in zip(data.get('quad_boxes', []), data.get('labels', [])):
        xs = box[0::2]; ys = box[1::2]
        regs.append({"text": label, "bbox": [[min(xs), min(ys)], [max(xs), min(ys)],
                                             [max(xs), max(ys)], [min(xs), max(ys)]],
                     "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]})
    return regs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="val_burnedin_gt.json (GT insertion coords + file paths)")
    ap.add_argument("--model", default="microsoft/Florence-2-base")
    args = ap.parse_args()
    gt = json.load(open(args.gt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device).eval()
    is_tech = lambda t: cr._is_technical_term(t)
    tot = dict(det=0, clf=0, spat=0, union=0, rem=0)
    N = len(gt)
    for i, (sop, info) in enumerate(gt.items()):
        ds = pydicom.dcmread(info["file"], force=True)
        img = to_display(ds.pixel_array, ds)
        w, h = img.width, img.height
        regs = florence_regions(model, processor, device, img)
        g = gbox(info); m = meta_of(ds)
        raw = [r["box"] for r in regs]
        clf = [r["box"] for r in regs if cr.sensitivity_reason(r["text"], r, m)]
        spat = [r["box"] for r in regs if dc.by_spatial(r["text"], r, w, h)]
        union = [r["box"] for r in regs if cr.sensitivity_reason(r["text"], r, m) or dc.by_spatial(r["text"], r, w, h)]
        rem, _ = dc.select_redaction(regs, m, w, h, cr.sensitivity_reason, is_tech, margin_active=True)
        tot["det"] += int(cov(g, raw) >= THR)
        tot["clf"] += int(cov(g, clf) >= THR)
        tot["spat"] += int(cov(g, spat) >= THR)
        tot["union"] += int(cov(g, union) >= THR)
        tot["rem"] += int(cov(g, rem) >= THR)
        print(f"[{i+1}/{N}] {info['modality']} regions={len(regs)}", flush=True)
    print("\n=== STANDALONE clean detector-swap (Florence side), N=%d ===" % N)
    print(f"  Florence-2 detection : {tot['det']}/{N}   (paper 34)")
    print(f"  + clean-room classifier: {tot['clf']}/{N}   (paper 10)")
    print(f"  spatial (fallback)   : {tot['spat']}/{N}   (paper 27 w/ catalogue)")
    print(f"  layered union        : {tot['union']}/{N}   (paper 30 w/ catalogue)")
    print(f"  full-pipeline removal: {tot['rem']}/{N}   (paper 33)")


if __name__ == "__main__":
    main()
