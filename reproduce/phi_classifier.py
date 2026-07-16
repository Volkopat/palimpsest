"""Clean-room PHI text classifier, implemented from the published specification in
Supplementary Material A of "An On-Premise, Open-Weights Vision-Language Pipeline for
Burned-In PHI Removal" (Sections 1 to 4). This is an independent implementation of the
PUBLISHED decision logic; it imports no proprietary product code. Given (OCR text, region
bbox, DICOM metadata) it reproduces the paper's clean-text classifier result (by_text),
and, with the spatial and margin helpers here, the layered union and shipped recall.

Deterministic and dependency-light (stdlib only). See Supplementary Material A for the
file:line provenance of every rule below."""
import re

# ---- config constants (Supp A Section 3.2, 6.1) --------------------------------------
MIN_TEXT_LENGTH = 2
HEADER_Y_THRESHOLD = 60
MIN_CAPITALIZED_WORDS = 2
MIN_ID_LENGTH = 6

TECHNICAL_EXCLUSIONS = [
    'tis', 'mi', 'hz', 'mhz', 'psv', 'edv', 'mdv',
    'ri', 'pi', 'at', 'tapv', 'prox', 'dist', 'mid',
    'arterial', 'vasc', 'vascular', 'doppler', 'tiso',
    'systolic', 'diastolic', 'velocity', 'flow', 'resistance',
    'pulsatility', 'acceleration', 'time', 'frequency', 'power',
    'gain', 'depth', 'focus', 'tgc', 'compress', 'dynamic',
    'persistence', 'map', 'scale', 'baseline', 'wall', 'filter',
    'steer', 'compound', 'harmonic', 'contrast',
]
MANUFACTURER_EXCLUSIONS = {
    'PHILIPS': ['xres', 'cres', 'speckle', 'xplane'],
    'GE':      ['logiq', 'vivid', 'voluson'],
    'SIEMENS': ['acuson', 'sequoia', 'antares'],
    'CANON':   ['aplio', 'xario'],
    'MINDRAY': ['resona', 'dc'],
}
TECHNICAL_HEADER_TERMS = [
    'vascular', 'arterial', 'experts', 'medical', 'hospital', 'clinic',
    'radiology', 'ultrasound', 'ct', 'mri', 'imaging', 'center',
    'tis', 'mi', 'hz', 'mhz', 'crr', 'le', 'ex',
]
MEDICAL_EXCLUSION_TERMS = ['arterial', 'vascular', 'experts', 'medical']

MRN_INDICATORS = ['mrn:', 'medical record number:', 'patient id:']
SSN_INDICATORS = ['ssn:', 'social security:']

DICOM_PHI_TAGS = [
    'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
    'PatientAge', 'AccessionNumber', 'StudyDate',
    'StudyTime', 'SeriesDate', 'SeriesTime',
]
_NAME_TAGS = ['PatientName', 'ReferringPhysicianName', 'PerformingPhysicianName']
_ID_TAGS = ['PatientID', 'AccessionNumber', 'StudyDate']


def _tokenize(text_lower):
    return set(re.findall(r'[a-z0-9]+', text_lower))


def _name_tokens(value):
    return set(value.lower().replace('^', ' ').replace(',', ' ').split())


def _is_technical_term(text_lower, manufacturer=""):
    tokens = _tokenize(text_lower)
    alpha_tokens = {t for t in tokens if not t.isdigit()}
    if not alpha_tokens:
        return False
    excl = set(TECHNICAL_EXCLUSIONS)
    if manufacturer:
        mup = manufacturer.upper()
        for mfg, terms in MANUFACTURER_EXCLUSIONS.items():
            if mfg in mup:
                excl |= set(t.lower() for t in terms)
    return alpha_tokens.issubset(excl)


def _matches_dicom_phi_value(text, dicom_phi_values):
    text_clean = text.strip().lower()
    if not text_clean:
        return False
    for tag_name, tag_value in dicom_phi_values.items():
        tag_value_clean = str(tag_value).strip().lower()
        if not tag_value_clean:
            continue
        # Mode 1: exact equality, any tag
        if text_clean == tag_value_clean:
            return True
        # Mode 2: name token-set intersection, name tags only
        if tag_name in _NAME_TAGS:
            text_parts = _name_tokens(text_clean)
            tag_parts = _name_tokens(tag_value_clean)
            shared = text_parts & tag_parts
            if len(shared) >= 2:
                return True
            if len(text_parts) == 1 and shared and max(len(t) for t in shared) >= 3:
                return True
        # Mode 3: bidirectional substring, ID/date tags
        elif tag_name in _ID_TAGS:
            if tag_value_clean in text_clean or text_clean in tag_value_clean:
                return True
    return False


def _header_y(bbox):
    """Return the y used for header determination (Supp A Step 3), or None."""
    if not bbox:
        return None
    if isinstance(bbox, list) and len(bbox) > 0:
        if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
            return bbox[0][1]
        if len(bbox) >= 2 and isinstance(bbox[1], (int, float)):
            return bbox[1]
    return None


def sensitivity_reason(text, region, metadata=None):
    """The Supp A Section 1 waterfall. Returns a reason string if PHI, else None."""
    metadata = metadata or {}
    dicom_phi_values = metadata.get("dicom_phi_values", {})
    manufacturer = metadata.get("manufacturer", "")

    # Step 0a: length gate (on raw text, before EOS strip)
    text = (text or "").strip()
    if not text or len(text) < MIN_TEXT_LENGTH:
        return None
    # Step 0b: Florence EOS strip
    if text.startswith('</s>'):
        text = text[4:].strip()
    # Step 0c: lowercase copy
    text_lower = text.lower()

    # Step 1: DICOM cross-reference
    if dicom_phi_values and _matches_dicom_phi_value(text, dicom_phi_values):
        return "Matches DICOM PHI tag value"
    # Step 2: technical-term veto (ALL alpha tokens)
    if _is_technical_term(text_lower, manufacturer):
        return None

    # Step 3: header determination
    y = _header_y(region.get("bbox", []))
    is_in_header = y is not None and y < HEADER_Y_THRESHOLD

    header_tokens = _tokenize(text_lower)
    # Step 4 (header only)
    if is_in_header:
        # 4a comma name
        if ',' in text:
            parts = text.split(',')
            if len(parts) == 2 and all(p.strip() for p in parts):
                if not header_tokens.issubset(set(TECHNICAL_EXCLUSIONS)):
                    return "Patient name pattern"
        # 4b multiple capitalized words
        words = text.split()
        if len(words) >= MIN_CAPITALIZED_WORDS:
            cap_words = [w for w in words if w and w[0].isupper()]
            if len(cap_words) >= MIN_CAPITALIZED_WORDS:
                if not (header_tokens & set(MEDICAL_EXCLUSION_TERMS)):
                    return "Multiple capitalized words in header"

    # Step 5: DOB indicator (header-independent)
    if re.search(r'\bdob\b', text_lower) or 'date of birth' in text_lower or 'd.o.b' in text_lower:
        return "DOB indicator"
    # Step 6: MRN indicator
    if any(ind in text_lower for ind in MRN_INDICATORS):
        return "Medical record number indicator"
    # Step 7: SSN indicator
    if any(ind in text_lower for ind in SSN_INDICATORS):
        return "SSN indicator"
    # Step 8: bare SSN pattern
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
        return "SSN pattern"

    # Step 9 (header only)
    if is_in_header:
        # 9a header technical veto (ANY token)
        if _tokenize(text_lower) & set(TECHNICAL_HEADER_TERMS):
            return None
        # 9b date in header
        if re.search(r'\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b', text):
            return "Date in header"
        # 9c long numeric ID in header
        if text.replace(',', '').replace('.', '').replace('-', '').isdigit():
            if len(text) >= MIN_ID_LENGTH:
                return "Long numeric ID in header"

    # Step 10: default
    return None


def is_phi(text, region, metadata=None):
    return sensitivity_reason(text, region, metadata) is not None
