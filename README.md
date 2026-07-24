# MIDI-B burned-in PHI reproduction harness

Reproduction harness for the manuscript **"An On-Premise, Open-Weights Vision-Language
Pipeline for Burned-In PHI Removal: Detection, Decision, and Robustness on MIDI-B."** The
manuscript is currently in submission; a preprint or DOI link will be added here when
available.

This repository contains the scoring and experiment scripts, which operate on the public
MIDI-B data, under the Apache-2.0 license. The de-identification pipeline evaluated in the
paper is a commercial product of aycan Medical Systems LLC and is **not** included here; its
complete rule set, technical-term veto lists, spatial prior with generic image-margin
fallback, and margin rule are published in **Supplementary Material A** of the paper. This
harness implements that published specification in `reproduce/` (files `phi_classifier.py`
and `decision.py`), so the reported pixel-path results reproduce from the public data plus
the published spec, with no proprietary code.

The proprietary manufacturer zone catalogue is not disclosed. On this benchmark it
contributes only 3 of the spatial prior's 27 flagged regions; the generic image-margin
fallback that supplies the other 24 is implemented here in full, so the reported removal
results reproduce without it (see "What reproduces" below).

## What reproduces (and what needs the proprietary pipeline)

Reproduces standalone from public MIDI-B + public Florence-2-base weights + this harness:

| Result | Paper | Harness | Script |
|---|---|---|---|
| Florence-2 detection | 34/35 | 34/35 | `run_clean_detector_swap_florence.py` |
| EasyOCR detection | 33/35 | 33/35 | `run_engine_detection.py --engine easyocr` |
| Tesseract detection | 11/35 | 11/35 | `run_engine_detection.py --engine tesseract` |
| + classifier (either modern engine) | 10/35 | 10/35 | `run_clean_detector_swap_florence.py` |
| Full-pipeline removal, clean | 33/35 | 33/35 | `run_clean_detector_swap_florence.py` |
| Degradation removal (Florence flat 31-33; EasyOCR collapse to 18 at blur r4) | see Table S1 | reproduces | `run_degradation_removal.py` |

Two intermediate ablation rows in the paper's Table 1 are computed **with** the proprietary
catalogue and therefore differ by one region when the catalogue is withheld: the "Spatial
prior (margin plus catalogue)" row is 27/35 in the paper and **28/35** here (fallback only),
and the "Layered union" row is 30/35 in the paper and **31/35** here. The generic fallback is
marginally more aggressive than the catalogue on the single Philips image the catalogue
handles. The headline detection, classifier, and removal numbers are unaffected (the extra
region is already covered by the classifier or margin rule, so removal stays 33/35).

Needs the full proprietary pipeline (metadata de-identification + CBIIT scorer), not
reproducible from this harness alone: the field-standard **series-accuracy frontier**
(retain 97.58% / strict 86.69% / CTP-default 46.75%) and the shipped scorer-legibility
figures (32-33/35). These depend on the metadata pipeline, which is not part of Supplementary
Material A. The scoring denominator is 117,669 validation checks.

## Data

- MIDI-B validation/test: The Cancer Imaging Archive, DOI 10.7937/cf2p-aw56 (public).
- Independent corroboration set (TCIA Pseudo-PHI-DICOM): DOI 10.7937/s17z-r072 (public).
- `data/val_burnedin_gt.json`: the 35 validation burned-in ground-truth regions (answer-key
  insertion coordinates, OCR-independent) with their DICOM file paths. Point the `file`
  fields at your local MIDI-B download, or regenerate from the public answer key.

## Environments

The three OCR engines have conflicting `torch` requirements; use one virtual environment
per engine.

- **Florence-2** (`florence`): `pip install -r requirements.txt`, then torch/torchvision for
  your GPU (this work used `--index-url https://download.pytorch.org/whl/cu128`).
- **EasyOCR** (`easyocr`): `pip install pydicom==2.2.1 numpy pillow easyocr`.
- **Tesseract** (`tesseract`): `pip install pydicom==2.2.1 numpy pillow pytesseract` plus a
  system Tesseract 5 install.

### Florence-2 setup

Florence-2-base imports `flash_attn` unconditionally even when it is unused. If `flash_attn`
is not installed, create a stub package exposing the names it imports (`flash_attn_func`,
`flash_attn_varlen_func`, etc.) that raise if actually called; they are not called on the
CPU/GPU inference path used here. Model weights are the public `microsoft/Florence-2-base`.

## Run

```
# Florence side: detection + clean-room classifier + spatial fallback + margin + removal
python reproduce/run_clean_detector_swap_florence.py --gt data/val_burnedin_gt.json \
    --model microsoft/Florence-2-base

# EasyOCR / Tesseract detection (each in its own environment)
python reproduce/run_engine_detection.py --gt data/val_burnedin_gt.json --engine easyocr
python reproduce/run_engine_detection.py --gt data/val_burnedin_gt.json --engine tesseract \
    --tesseract-cmd /path/to/tesseract --tessdata /path/to/tessdata

# Degradation removal sweep (per engine)
python reproduce/run_degradation_removal.py --gt data/val_burnedin_gt.json --engine florence

# Figures from the results table
python reproduce/make_figures.py
```

`results/revision_results.json` holds the full numbers (seed 0, coverage grid 200 at
threshold 0.50, strict profile) for cross-checking. Note: this file is the log of the
shipped (proprietary, catalogue-included) pipeline run, so its `spatial_prior` (27) and
`union` (30) are the paper's Table 1 values; the clean-room harness in this repo, with the
catalogue withheld, produces 28 and 31 for those two rows (see the table above). Detection,
classifier, and removal match exactly in both.

## Method notes

- Removal is measured by geometric coverage (threshold 0.50) of the ground-truth region by
  the pipeline's redaction box, deliberately not the scorer's OCR-legibility, because
  corruption can render text illegible on its own and would falsely credit the pipeline. On
  the clean condition the two metrics agree to within one region (legibility 32/35, geometry
  33/35).
- Corruptions are deterministic (seed 0), applied to the 8-bit display render.
- All measurements assert model/classifier initialization before running (an early diagnostic
  that skipped model initialization produced a spurious result; this guard prevents that
  class of error).

## License

Apache-2.0 (see `LICENSE`). The de-identification pipeline itself is proprietary and not
included; see Supplementary Material A of the paper for its specification.
