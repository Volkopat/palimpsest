# Revision scripts (R2)

The scripts that produced the measurements added in the R2 revision, copied here
from the private working tree by `scripts/port_revision_to_harness.py`. Result files
are in `../../results/revision/`.

## What runs against public inputs alone

| script | notes |
|---|---|
| `rev2_easyocr_tuned.py` | EasyOCR only; runs against the saved PNGs |
| `rev2_glyph_geometry.py` | pixel statistics only |
| `rev2_conformance.py` | drives dciodvfy; needs no model and no pipeline |
| `degrade_conditions.py` | the nine corruption conditions, deterministic |
| `ctp/CtpRun.java` | drives RSNA CTP's own classes; needs no pipeline |

## What needs the de-identification pipeline, which is not released

These are shipped **as they were run**, not adapted. The pipeline is a commercial
product and is withheld; its rule set, technical-term veto lists, spatial prior and
margin rule are specified in reproduction-complete detail in Supplementary Material A,
and `reproduce/decision.py` and `reproduce/phi_classifier.py` in this harness are a
clean-room implementation of that specification which reproduces the paper's clean
detection, classifier and removal figures. The scripts below were not rewritten
against that clean-room module, because an unverified rewrite that silently produced
different numbers would be worse than shipping none. Every number they produced is in
`../../results/revision/`, so each can be checked without the package.

| script | what it calls into |
|---|---|
| `rev2_gate_recall.py` | should_pixel_pass, the policy gate |
| `rev2_capture.py` | the pipeline's Florence-2 wrapper |
| `rev2_analyze.py` | the PHI classifier and metadata extraction |
| `rev2_reverse_arm.py` | _select_redaction_boxes, the layered decision union |
| `rev2_easyocr_removal.py` | _select_redaction_boxes, the layered decision union |
| `rev2_footprint.py` | the whole pipeline, since it measures its resident memory |

## Paths

Result files record DICOM paths relative to the dataset directory, for example
`MIDI-B-Synthetic-Validation/<study>/<series>/00000001.dcm`. Point the scripts at your
own copy of the public data; nothing here depends on the layout of the machine that
produced the numbers.
