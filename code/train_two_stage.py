from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from iframe_detector.features import (
    build_frame_training_table,
    numeric_feature_columns,
    pairwise_gap_features,
)
from iframe_detector.models import choose_threshold, save_model, threshold_metrics, train_lgbm_small
from iframe_detector.packets import load_dataframe, save_dataframe, write_json
from iframe_detector.splits import (
    DEFAULT_SPLIT_CONFIG_PATH,
    canonicalize_capture_columns,
    load_split_config,
    split_masks,
    summarize_split_table,
)


def load_all_packets(packet_root: Path) -> pd.DataFrame:
    frames = [load_dataframe(p) for p in sorted((packet_root / "packets").glob("*.packets.csv.gz"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_by_ood_app(df: pd.DataFrame, ood_app: str) -> tuple[pd.Series, pd.Series]:
    if "application" not in df.columns or df.empty:
        return pd.Series(dtype=bool), pd.Series(dtype=bool)
    test = df["application"].astype(str).str.lower().eq(ood_app.lower())
    if test.any() and (~test).any():
        return ~test, test
    captures = sorted(df["capture_id"].astype(str).unique())
    holdout = set(captures[-max(1, len(captures) // 4) :])
    test = df["capture_id"].astype(str).isin(holdout)
    return ~test, test


def split_training_table(table: pd.DataFrame, split_config: dict | None, ood_app: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    if split_config:
        table = canonicalize_capture_columns(table, split_config, drop_unassigned=True)
        train_mask, test_mask = split_masks(table)
        summary = {
            "strategy": "split_config",
            "split_id": split_config.get("split_id"),
            "train_roles": ["development"],
            "test_roles": ["final_app_ood"],
            "table": summarize_split_table(table),
        }
        return table, train_mask, test_mask, summary
    train_mask, test_mask = split_by_ood_app(table, ood_app)
    return table, train_mask, test_mask, {"strategy": "ood_app_fallback", "ood_app": ood_app}


def train_boundary(packet_df: pd.DataFrame, label_df: pd.DataFrame, output_root: Path, ood_app: str, split_config: dict | None) -> dict:
    tables = []
    if not packet_df.empty:
        packet_df = canonicalize_capture_columns(packet_df, split_config, drop_unassigned=bool(split_config))
    if not label_df.empty:
        label_df = canonicalize_capture_columns(label_df, split_config, drop_unassigned=bool(split_config))
    if packet_df.empty or label_df.empty or "flow_key" not in packet_df.columns or "flow_key" not in label_df.columns:
        table = pd.DataFrame()
        save_dataframe(table, output_root / "boundary_training_table.csv.gz")
        return {"trained": False, "reason": "not enough labeled pairwise boundary examples", "row_count": 0}
    group_cols = ["capture_id", "flow_key"] if "capture_id" in packet_df.columns else ["flow_key"]
    labels_by_flow = {k: g.copy() for k, g in label_df.groupby(group_cols)} if not label_df.empty else {}
    for key, flow_df in packet_df.groupby(group_cols):
        lbl = labels_by_flow.get(key)
        if lbl is None or lbl.empty:
            continue
        feat = pairwise_gap_features(flow_df, lbl)
        if not feat.empty:
            tables.append(feat)
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    save_dataframe(table, output_root / "boundary_training_table.csv.gz")
    if table.empty or table["label_boundary"].nunique() < 2:
        return {"trained": False, "reason": "not enough labeled pairwise boundary examples", "row_count": int(len(table))}
    table = table[table["label_known"].eq(1)].copy()
    table, train_mask, test_mask, split_summary = split_training_table(table, split_config, ood_app)
    feature_cols = numeric_feature_columns(table, label_cols=["label_boundary"])
    train = table[train_mask]
    test = table[test_mask] if test_mask.any() else table[~train_mask]
    if train.empty or test.empty or train["label_boundary"].nunique() < 2:
        return {
            "trained": False,
            "reason": "not enough labeled pairwise boundary examples after recovered split",
            "row_count": int(len(table)),
            "split": split_summary,
        }
    model = train_lgbm_small(train[feature_cols], train["label_boundary"], n_estimators=96, max_depth=5)
    scores = model.predict_proba(test[feature_cols])[:, 1] if len(test) else np.array([])
    search = threshold_metrics(test["label_boundary"].to_numpy(), scores, [x / 100 for x in range(5, 96, 5)]) if len(test) else []
    threshold = choose_threshold(search, min_recall=0.85) if search else 0.5
    save_model(
        model,
        feature_cols,
        {
            "model_name": "rebuilt_pairwise_gap_lgbm96_d5",
            "threshold": threshold,
            "threshold_search": search,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "split": split_summary,
        },
        output_root / "boundary_model",
    )
    return {"trained": True, "feature_count": len(feature_cols), "threshold": threshold, "threshold_search": search, "split": split_summary}


def train_iframe(packet_df: pd.DataFrame, label_df: pd.DataFrame, output_root: Path, ood_app: str, split_config: dict | None) -> dict:
    if not packet_df.empty:
        packet_df = canonicalize_capture_columns(packet_df, split_config, drop_unassigned=bool(split_config))
    if not label_df.empty:
        label_df = canonicalize_capture_columns(label_df, split_config, drop_unassigned=bool(split_config))
    if packet_df.empty or label_df.empty or "flow_key" not in packet_df.columns or "flow_key" not in label_df.columns:
        table = pd.DataFrame()
        save_dataframe(table, output_root / "iframe_training_table.csv.gz")
        return {"trained": False, "reason": "not enough labeled frame examples", "row_count": 0}
    table = build_frame_training_table(packet_df, label_df)
    save_dataframe(table, output_root / "iframe_training_table.csv.gz")
    if table.empty or table["label_iframe"].nunique() < 2:
        return {"trained": False, "reason": "not enough labeled frame examples", "row_count": int(len(table))}
    table, train_mask, test_mask, split_summary = split_training_table(table, split_config, ood_app)
    feature_cols = numeric_feature_columns(table, label_cols=["label_iframe"])
    train = table[train_mask]
    test = table[test_mask] if test_mask.any() else table[~train_mask]
    if train.empty or test.empty or train["label_iframe"].nunique() < 2:
        return {
            "trained": False,
            "reason": "not enough labeled I-frame examples after recovered split",
            "row_count": int(len(table)),
            "split": split_summary,
        }
    model = train_lgbm_small(train[feature_cols], train["label_iframe"], n_estimators=24, max_depth=4)
    scores = model.predict_proba(test[feature_cols])[:, 1] if len(test) else np.array([])
    search = threshold_metrics(test["label_iframe"].to_numpy(), scores, [x / 100 for x in range(5, 96, 5)]) if len(test) else []
    threshold = choose_threshold(search, min_recall=0.75) if search else 0.5
    save_model(
        model,
        feature_cols,
        {
            "model_name": "rebuilt_frame_iframe_lgbm24_d4",
            "threshold": threshold,
            "threshold_search": search,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "split": split_summary,
        },
        output_root / "iframe_model",
    )
    return {"trained": True, "feature_count": len(feature_cols), "threshold": threshold, "threshold_search": search, "split": split_summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-root", type=Path, default=Path("artifacts/rebuild_packets"))
    ap.add_argument("--label-root", type=Path, default=Path("artifacts/rebuild_labels"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/rebuild_models"))
    ap.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    ap.add_argument("--ood-app", default="xhs")
    args = ap.parse_args()

    split_config = load_split_config(args.split_config)
    packets = load_all_packets(args.packet_root)
    labels = load_dataframe(args.label_root / "all_frames.csv.gz") if (args.label_root / "all_frames.csv.gz").exists() else pd.DataFrame()
    packets = canonicalize_capture_columns(packets, split_config, drop_unassigned=bool(split_config))
    labels = canonicalize_capture_columns(labels, split_config, drop_unassigned=bool(split_config))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "split_id": split_config.get("split_id") if split_config else None,
        "packet_split": summarize_split_table(packets),
        "label_split": summarize_split_table(labels),
        "packet_rows": int(len(packets)),
        "label_rows": int(len(labels)),
        "keyframe_rows": int(labels["is_keyframe"].sum()) if not labels.empty else 0,
        "boundary": train_boundary(packets, labels, args.output_root, args.ood_app, split_config),
        "iframe": train_iframe(packets, labels, args.output_root, args.ood_app, split_config),
    }
    write_json(summary, args.output_root / "training_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
