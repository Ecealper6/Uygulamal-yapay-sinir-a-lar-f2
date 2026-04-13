from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .preprocessing import CATEGORY_COLUMN, NUMERIC_COLUMNS


DEFAULT_SAMPLE = {
    "Yas": 67,
    "Nabiz": 112,
    "Ates": 38.4,
    "Sistolik_Tansiyon": 95,
    "Oksijen_Saturasyonu": 91.0,
    "Solunum_Sayisi": 24,
    "Sikayet": "NefesDarligi",
}


TRIAGE_LABELS = {
    2: "Acil - çok hızlı müdahale gerekli",
    3: "Öncelikli - kısa sürede değerlendirme gerekli",
    4: "Standart - orta öncelik",
    5: "Düşük öncelik - stabil hasta",
}


def build_demo_message(best_model_name: str, prediction: int, sample: Dict[str, object]) -> str:
    lines = [
        "Demo Patient Input:",
        *(f"- {k}: {v}" for k, v in sample.items()),
        f"Predicted Triage Level ({best_model_name}): {prediction}",
    ]
    return "\n".join(lines)


def transform_sample(sample: Dict[str, object], reference_csv: str, metadata: dict) -> np.ndarray:
    pd.read_csv(reference_csv)  # lightweight verification of file availability
    row = {col: sample[col] for col in NUMERIC_COLUMNS}
    temp = pd.DataFrame([row])

    categories = metadata.get("categories") or []
    if metadata.get("encoding") == "onehot":
        for category in categories:
            temp[f"Sikayet_{category}"] = 1.0 if sample[CATEGORY_COLUMN] == category else 0.0
    else:
        mapping = {name: idx for idx, name in enumerate(categories)}
        temp["Sikayet_Label"] = float(mapping[sample[CATEGORY_COLUMN]])

    ordered = metadata["feature_names"]
    for col in ordered:
        if col not in temp.columns:
            temp[col] = 0.0
    temp = temp[ordered]

    scale_stats = metadata.get("scale_stats", {})
    scaling = metadata.get("scaling")
    if scaling == "zscore":
        for col in NUMERIC_COLUMNS:
            stats = scale_stats.get(col, {})
            mean = float(stats.get("mean", 0.0))
            std = float(stats.get("std", 1.0)) or 1.0
            temp[col] = (temp[col].astype(float) - mean) / std
    elif scaling == "minmax":
        for col in NUMERIC_COLUMNS:
            stats = scale_stats.get(col, {})
            min_val = float(stats.get("min", 0.0))
            denom = float(stats.get("denom", 1.0)) or 1.0
            temp[col] = (temp[col].astype(float) - min_val) / denom

    return temp.values
