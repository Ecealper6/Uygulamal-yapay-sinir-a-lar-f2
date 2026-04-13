from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessConfig:
    test_size: float = 0.2
    random_state: int = 42
    encoding: str = "onehot"  # onehot or label
    scaling: str = "zscore"  # zscore, minmax, none
    clip_z_threshold: float = 0.0


NUMERIC_COLUMNS: List[str] = [
    "Yas",
    "Nabiz",
    "Ates",
    "Sistolik_Tansiyon",
    "Oksijen_Saturasyonu",
    "Solunum_Sayisi",
]
TARGET_COLUMN = "Triyaj_Seviyesi"
CATEGORY_COLUMN = "Sikayet"


def _clip_outliers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if threshold <= 0:
        return df
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        mean = out[col].mean()
        std = out[col].std(ddof=0)
        if std < 1e-9:
            std = 1.0
        low = mean - threshold * std
        high = mean + threshold * std
        out[col] = out[col].clip(lower=low, upper=high)
    return out


def _encode_features(df: pd.DataFrame, encoding: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    base = df[NUMERIC_COLUMNS].copy()
    categories = sorted(df[CATEGORY_COLUMN].dropna().unique().tolist())

    if encoding == "onehot":
        cat = pd.get_dummies(df[CATEGORY_COLUMN], prefix="Sikayet")
        X = pd.concat([base, cat], axis=1)
        return X, list(X.columns), categories

    if encoding == "label":
        mapping = {name: idx for idx, name in enumerate(categories)}
        base["Sikayet_Label"] = df[CATEGORY_COLUMN].map(mapping).astype(float)
        return base, list(base.columns), categories

    raise ValueError(f"Unsupported encoding: {encoding}")


def _scale_features(X: pd.DataFrame, scaling: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    out = X.copy()
    stats: Dict[str, Dict[str, float]] = {}
    target_cols = NUMERIC_COLUMNS.copy()

    if scaling == "none":
        for col in target_cols:
            if col in out.columns:
                stats[col] = {}
        return out, stats

    for col in target_cols:
        if col not in out.columns:
            continue
        series = out[col].astype(float)
        if scaling == "zscore":
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            if std < 1e-9:
                std = 1.0
            out[col] = (series - mean) / std
            stats[col] = {"mean": mean, "std": std}
        elif scaling == "minmax":
            min_val = float(series.min())
            max_val = float(series.max())
            denom = max_val - min_val
            if abs(denom) < 1e-9:
                denom = 1.0
            out[col] = (series - min_val) / denom
            stats[col] = {"min": min_val, "max": max_val, "denom": float(denom)}
        else:
            raise ValueError(f"Unsupported scaling: {scaling}")
    return out, stats


def load_and_preprocess(csv_path: str, config: PreprocessConfig | None = None):
    config = config or PreprocessConfig()
    df = pd.read_csv(csv_path)

    required = set(NUMERIC_COLUMNS + [CATEGORY_COLUMN, TARGET_COLUMN])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = _clip_outliers(df, config.clip_z_threshold)
    X_df, feature_names, categories = _encode_features(df, config.encoding)
    X_df, scale_stats = _scale_features(X_df, config.scaling)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values,
        y.values,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
    )

    metadata = {
        "feature_names": feature_names,
        "class_distribution": y.value_counts().sort_index().to_dict(),
        "sample_count": int(len(df)),
        "encoding": config.encoding,
        "scaling": config.scaling,
        "categories": categories,
        "scale_stats": scale_stats,
        "numeric_columns": NUMERIC_COLUMNS.copy(),
    }

    return X_train, X_test, y_train, y_test, metadata
