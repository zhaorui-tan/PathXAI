from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Tuple


def assign_split(key: str, train: float = 0.7, val: float = 0.15) -> str:
    if train <= 0 or val < 0 or train + val >= 1:
        raise ValueError("Expected train > 0, val >= 0, and train + val < 1")
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def add_splits(items: Iterable[dict], train: float = 0.7, val: float = 0.15) -> List[dict]:
    result = []
    for item in items:
        out = dict(item)
        key = f"{out.get('dataset_name')}::{out.get('case_id')}::{out.get('path')}"
        out["split"] = assign_split(key, train=train, val=val)
        result.append(out)
    return result


def split_counts(items: Iterable[dict]) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0, "unsplit": 0}
    for item in items:
        split = item.get("split") or "unsplit"
        counts[split] = counts.get(split, 0) + 1
    return counts
