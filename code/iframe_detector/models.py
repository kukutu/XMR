from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support


def train_lgbm_small(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 64,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    class_weight: str | dict | None = "balanced",
) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=min(2**max_depth, 31),
        learning_rate=learning_rate,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight=class_weight,
        random_state=20260821,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, thresholds: list[float]) -> list[dict[str, float]]:
    rows = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        rows.append({"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f1)})
    return rows


def choose_threshold(rows: list[dict[str, float]], min_recall: float = 0.8) -> float:
    feasible = [r for r in rows if r["recall"] >= min_recall]
    if feasible:
        return max(feasible, key=lambda r: (r["f1"], r["precision"], r["threshold"]))["threshold"]
    return max(rows, key=lambda r: (r["f1"], r["precision"], r["threshold"]))["threshold"] if rows else 0.5


def save_model(model, feature_names: list[str], metadata: dict, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root / "model.pkl")
    payload = dict(metadata)
    payload["feature_names"] = feature_names
    (root / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(root: Path):
    model = joblib.load(root / "model.pkl")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata
