"""Clean-room decision layer implementing the PUBLISHED spatial-prior fallback and margin
rule from Supplementary Material A (Section A.9) and the paper. Imports no proprietary
product code and does NOT use the proprietary manufacturer zone catalogue (withheld); it
implements only the generic image-margin fallback, which the paper states supplies 24 of
the spatial prior's 27 flagged regions. Together with phi_classifier.py this reproduces the
layered union and the shipped removal recall from public data.

NOTE on fidelity: the generic-strip fallback is reproduced exactly as the shipped pipeline
computes it. The fallback rectangles are emitted as (x, y, w, h) tuples that the scoring
step reads positionally as (x1, y1, x2, y2); this makes the top and left strips valid
rectangles while the bottom and right strips are malformed and contribute only via the
distance-to-center tier. This quirk is preserved so the harness matches the paper numbers.
"""
import re

MARGIN_FRACTION = 0.12
MARGIN_MIN_ALNUM = 2
ANONYMIZATION_PADDING = 5
MARGIN_MODALITIES = frozenset({'CR', 'DX', 'MG'})
_FLORENCE_SPECIAL = re.compile(r'</?s>')

RADIOGRAPHIC_TECHNIQUE = frozenset({
    'ap', 'pa', 'lat', 'lateral', 'oblique', 'obl', 'decub', 'decubitus',
    'portable', 'mobile', 'bedside', 'stretcher', 'wheelchair', 'grid',
    'erect', 'semierect', 'semi', 'upright', 'supine', 'prone', 'standing',
    'sitting', 'recumbent', 'lordotic', 'axial', 'tangential', 'swimmers',
    'expiration', 'inspiration', 'exp', 'insp', 'inhale', 'exhale',
    'weightbearing', 'bearing', 'flexion', 'extension', 'stress',
    'left', 'right', 'bilat', 'bilateral', 'superior', 'inferior',
    'anterior', 'posterior', 'medial', 'proximal', 'distal',
    'chest', 'abdomen', 'abd', 'pelvis', 'spine', 'kub', 'cxr', 'view',
    'projection', 'proj', 'intra', 'op', 'stat', 'wet', 'read', 'comparison',
    'prior', 'horizontal', 'vertical', 'cross', 'table', 'rotation',
    'internal', 'external', 'sid', 'fov', 'xr', 'dr',
})


def _bbox_to_coords(bbox):
    try:
        if isinstance(bbox[0], (list, tuple)):
            xs = [p[0] for p in bbox if len(p) >= 2]
            ys = [p[1] for p in bbox if len(p) >= 2]
            if xs and ys:
                return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        elif len(bbox) >= 4:
            xs = [bbox[i] for i in range(0, len(bbox), 2)]
            ys = [bbox[i] for i in range(1, len(bbox), 2)]
            if xs and ys:
                return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    except Exception:
        pass
    return None


def _default_focus_regions(width, height):
    # exact shipped tuples (x, y, w, h), read positionally as (x1, y1, x2, y2) downstream
    return [
        (0, 0, width, int(height * 0.15)),
        (0, int(height * 0.85), width, int(height * 0.15)),
        (0, 0, int(width * 0.2), height),
        (int(width * 0.8), 0, int(width * 0.2), height),
    ]


def should_prioritize_region(region, width, height):
    """Content-blind positional score in {0.3, 0.5, 0.6, 0.8, 1.0}. Catalogue withheld, so
    focus regions are the generic default strips only (match_type 'default')."""
    bbox = region.get('bbox', [])
    if not bbox:
        return 0.5
    rc = _bbox_to_coords(bbox)
    if not rc:
        return 0.5
    x1, y1, x2, y2 = rc
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    focus = _default_focus_regions(width, height)
    for fx1, fy1, fx2, fy2 in focus:
        if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
            return 1.0
    mind = float('inf')
    for fx1, fy1, fx2, fy2 in focus:
        fcx = (fx1 + fx2) / 2
        fcy = (fy1 + fy2) / 2
        mind = min(mind, ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5)
    if mind < 100:
        return 0.8
    if mind < 200:
        return 0.6
    return 0.3


def by_spatial(text, region, width, height):
    score = should_prioritize_region(region, width, height)
    return (score is not None and score >= 0.8 and len(text) > 1
            and any(c.isupper() or c.isdigit() for c in text))


def _is_radiographic_technique(text, is_technical_term):
    toks = re.findall(r'[A-Za-z]+', str(text).lower())
    if not toks:
        return True
    return all(t in RADIOGRAPHIC_TECHNIQUE or is_technical_term(t) for t in toks)


def by_margin(text, bbox, width, height, is_technical_term):
    clean = _FLORENCE_SPECIAL.sub('', text or '')
    alnum = ''.join(c for c in clean if c.isalnum())
    if len(alnum) < MARGIN_MIN_ALNUM:
        return False
    rc = _bbox_to_coords(bbox)
    if not rc:
        return False
    x1, y1, x2, y2 = rc
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    mf = MARGIN_FRACTION
    in_margin = (cx < mf * width or cx > (1 - mf) * width
                 or cy < mf * height or cy > (1 - mf) * height)
    if not in_margin:
        return False
    return not _is_radiographic_technique(clean, is_technical_term)


def select_redaction(regions, metadata, width, height, sensitivity_reason, is_technical_term,
                     margin_active=True):
    """Mirror of the shipped _select_redaction_boxes: redact if by_text OR by_spatial OR
    by_margin (margin gated behind the other two and CR/DX/MG + strict profile). Returns
    padded, clipped (x0,y0,x1,y1) boxes plus per-region provenance."""
    boxes, decisions = [], []
    pad = ANONYMIZATION_PADDING
    modality = str((metadata or {}).get("modality", "")).upper()
    for region in regions:
        text = (region.get("text", "") or "").strip()
        bbox = region.get("bbox", [])
        bt = sensitivity_reason(text, region, metadata) is not None
        bs = by_spatial(text, region, width, height) if metadata else False
        bm = False
        if margin_active and not (bt or bs) and modality in MARGIN_MODALITIES:
            bm = by_margin(text, bbox, width, height, is_technical_term)
        sensitive = bt or bs or bm
        rc = _bbox_to_coords(bbox) if bbox else None
        if sensitive and rc:
            x0 = max(0, rc[0] - pad); y0 = max(0, rc[1] - pad)
            x1 = min(width, rc[2] + pad); y1 = min(height, rc[3] + pad)
            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1, y1))
        decisions.append({"text": text, "by_text": bt, "by_spatial": bs, "by_margin": bm,
                          "redacted": bool(sensitive)})
    return boxes, decisions
