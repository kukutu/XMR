from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iframe_detector.models import load_model, threshold_metrics
from iframe_detector.packets import load_dataframe, save_dataframe, write_json
from iframe_detector.splits import DEFAULT_SPLIT_CONFIG_PATH, canonicalize_capture_columns, load_split_config


def binary_metrics(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    if df.empty:
        return {
            "threshold": threshold,
            "true_positive_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "predicted_positive_count": 0,
            "true_keyframe_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    y = df["label_iframe"].astype(int).to_numpy()
    pred = df["iframe_score"].ge(threshold).astype(int).to_numpy()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pred_count = int(pred.sum())
    truth_count = int(y.sum())
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / truth_count if truth_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "predicted_positive_count": pred_count,
        "true_keyframe_count": truth_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, default=Path("artifacts/rebuild_models"))
    ap.add_argument("--table-path", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/rebuild_iframe_oracle_eval"))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    args = ap.parse_args()

    split_config = load_split_config(args.split_config)
    table_path = args.table_path or (args.model_root / "iframe_training_table.csv.gz")
    table = load_dataframe(table_path) if table_path.exists() else pd.DataFrame()
    table = canonicalize_capture_columns(table, split_config, drop_unassigned=bool(split_config))
    if table.empty:
        raise SystemExit(f"I-frame training table is empty or missing: {table_path}")

    model, meta = load_model(args.model_root / "iframe_model")
    feature_cols = list(meta["feature_names"])
    scores = model.predict_proba(table.reindex(columns=feature_cols, fill_value=0))[:, 1]
    scored = table.copy()
    scored["iframe_score"] = scores
    threshold = float(args.threshold if args.threshold is not None else meta.get("threshold", 0.5))
    scored["predicted_iframe"] = scored["iframe_score"].ge(threshold).astype(int)
    save_dataframe(scored, args.output_root / "oracle_iframe_scores.csv.gz")

    search = threshold_metrics(
        scored["label_iframe"].astype(int).to_numpy(),
        np.asarray(scores),
        [x / 100 for x in range(5, 96, 5)],
    )
    summary = {
        "threshold": threshold,
        "threshold_search": search,
        "aggregate": binary_metrics(scored, threshold),
        "by_application": {},
        "by_split_role": {},
    }
    for app in sorted(scored["application"].astype(str).unique()):
        summary["by_application"][app] = binary_metrics(scored[scored["application"].astype(str).eq(app)], threshold)
    for role in sorted(scored["split_role"].astype(str).unique()):
        summary["by_split_role"][role] = binary_metrics(scored[scored["split_role"].astype(str).eq(role)], threshold)
    write_json(summary, args.output_root / "oracle_iframe_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
