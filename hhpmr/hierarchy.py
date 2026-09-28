import json
import re

import numpy as np

ICD9_MAX_DEPTH = 5
ICD10_MAX_DEPTH = 7

ICD9_NUMERIC = re.compile(r"^\d{3,5}$")
ICD9_V = re.compile(r"^V\d{2,4}$")
ICD9_E = re.compile(r"^E\d{3,4}$")
ICD10_CODE = re.compile(r"^[A-Z]\d{2}[A-Z0-9]{0,4}$")
VERSION_PREFIX = re.compile(r"^(9|10):(.+)$")


def normalize_code(code):
    return str(code).strip().upper().replace(".", "")


def parse_diagnosis_entry(entry, default_version):
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        return normalize_code(entry[0]), int(entry[1])
    text = str(entry)
    match = VERSION_PREFIX.match(text)
    if match:
        return normalize_code(match.group(2)), int(match.group(1))
    return normalize_code(text), default_version


def icd9_depth(code):
    if ICD9_NUMERIC.match(code) or ICD9_V.match(code):
        return len(code)
    if ICD9_E.match(code):
        return len(code) - 1
    return None


def icd10_depth(code):
    if ICD10_CODE.match(code):
        return len(code)
    return None


def load_depth_maps(path):
    with open(path, "r") as handle:
        raw = json.load(handle)
    maps = {}
    for version, table in raw.items():
        normalized = {normalize_code(code): int(depth) for code, depth in table.items()}
        maps[int(version)] = (normalized, max(normalized.values()))
    return maps


def specificity_value(code, version, depth_maps=None):
    if depth_maps is not None and version in depth_maps:
        table, max_depth = depth_maps[version]
        depth = table.get(code)
    else:
        depth = icd9_depth(code) if version == 9 else icd10_depth(code)
        max_depth = ICD9_MAX_DEPTH if version == 9 else ICD10_MAX_DEPTH
    if depth is None or depth <= 0:
        return None
    return min(depth / max_depth, 1.0)


def build_specificity_vector(words, default_version=9, depth_maps=None):
    values = np.zeros(len(words) + 1, dtype=np.float32)
    mapped = 0
    for idx, entry in enumerate(words):
        code, version = parse_diagnosis_entry(entry, default_version)
        value = specificity_value(code, version, depth_maps)
        if value is not None:
            values[idx] = value
            mapped += 1
    coverage = {
        "mapped": mapped,
        "total": len(words),
        "fraction": mapped / max(len(words), 1),
    }
    return values, coverage
