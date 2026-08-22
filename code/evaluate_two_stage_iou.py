from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iframe_detector.metrics import match_packet_sets_grouped
from iframe_detector.packets import load_dataframe, write_json
from iframe_detector.splits import DEFAULT_SPLIT_CONFIG_PATH, canonicalize_capture_columns, load_split_config
from iframe_detector.video import parse_packet_numbers


def packet_set_rows(df: pd.DataFrame, key_col: str = "frame_index") -> list[tuple[str, list[int]]]:
    rows = []
    if df.empty:
        return rows
    for row in df.itertuples(index=False):
        capture = getattr(row, "capture_id", "")
        flow = getattr(row, "flow_key", "")
        idx = getattr(row, key_col, len(rows))
        rows.append((f"{capture}:{flow}:{idx}", parse_packet_numbers(getattr(row, "packet_numbers"))))
    return rows


def grouped_packet_set_rows(df: pd.DataFrame, key_col: str = "frame_index") -> list[tuple[str, str, list[int]]]:
    rows = []
    if df.empty:
        return rows
    for row in df.itertuples(index=False):
        capture = getattr(row, "capture_id", "")
        flow = getattr(row, "flow_key", "")
        idx = getattr(row, key_col, len(rows))
        group_key = f"{capture}:{flow}"
        item_key = f"{group_key}:{idx}"
        rows.append((group_key, item_key, parse_packet_numbers(getattr(row, "packet_numbers"))))
    return rows


def known_packets_by_flow(label_df: pd.DataFrame) -> dict[tuple[str, str], set[int]]:
    known: dict[tuple[str, str], set[int]] = {}
    if label_df.empty:
        return known
    for row in label_df.itertuples(index=False):
        key = (str(getattr(row, "capture_id", "")), str(getattr(row, "flow_key", "")))
        known.setdefault(key, set()).update(parse_packet_numbers(getattr(row, "packet_numbers")))
    return known


def intersect_known_packets(pred_df: pd.DataFrame, support_label_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()
    known = known_packets_by_flow(support_label_df)
    rows = []
    for row in pred_df.itertuples(index=False):
        key = (str(getattr(row, "capture_id", "")), str(getattr(row, "flow_key", "")))
        packets = [p for p in parse_packet_numbers(getattr(row, "packet_numbers")) if p in known.get(key, set())]
        if not packets:
            continue
        payload = row._asdict()
        payload["packet_numbers"] = "|".join(str(p) for p in packets)
        rows.append(payload)
    return pd.DataFrame(rows)


def load_frame_type_labels(label_root: Path) -> pd.DataFrame:
    stage_path = label_root / "frame_type_labels.csv.gz"
    if stage_path.exists():
        return load_dataframe(stage_path)
    fallback_path = label_root / "all_frames.csv.gz"
    return load_dataframe(fallback_path) if fallback_path.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-root", type=Path, default=Path("artifacts/rebuild_predictions"))
    ap.add_argument("--label-root", type=Path, default=Path("artifacts/rebuild_labels"))
    ap.add_argument("--minimum-packet-iou", type=float, default=0.9)
    ap.add_argument("--output-path", type=Path, default=Path("artifacts/rebuild_predictions/iframe_iou90_summary.json"))
    ap.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    ap.add_argument("--known-packet-policy", choices=["full", "intersect"], default="full")
    args = ap.parse_args()

    split_config = load_split_config(args.split_config)
    pred_path = args.prediction_root / "positives" / "predicted_iframes.csv.gz"
    pred_df = load_dataframe(pred_path) if pred_path.exists() else pd.DataFrame()
    truth_df = load_frame_type_labels(args.label_root)
    pred_df = canonicalize_capture_columns(pred_df, split_config, drop_unassigned=bool(split_config))
    truth_df = canonicalize_capture_columns(truth_df, split_config, drop_unassigned=bool(split_config))
    support_df = truth_df.copy()
    if args.known_packet_policy == "intersect":
        pred_df = intersect_known_packets(pred_df, support_df)
    if not truth_df.empty:
        truth_df = truth_df[truth_df["is_keyframe"].astype(int).eq(1)].copy()
    by_application = {}
    by_split_role = {}
    aggregate = match_packet_sets_grouped(grouped_packet_set_rows(pred_df), grouped_packet_set_rows(truth_df), args.minimum_packet_iou)
    if not truth_df.empty and "application" in truth_df.columns:
        for app in sorted(truth_df["application"].astype(str).unique()):
            t_app = truth_df[truth_df["application"].astype(str).eq(app)]
            p_app = pred_df[pred_df["application"].astype(str).eq(app)] if not pred_df.empty else pred_df
            by_application[app] = match_packet_sets_grouped(grouped_packet_set_rows(p_app), grouped_packet_set_rows(t_app), args.minimum_packet_iou)
    if not truth_df.empty and "split_role" in truth_df.columns:
        for role in sorted(truth_df["split_role"].astype(str).unique()):
            t_role = truth_df[truth_df["split_role"].astype(str).eq(role)]
            p_role = pred_df[pred_df["split_role"].astype(str).eq(role)] if not pred_df.empty else pred_df
            by_split_role[role] = match_packet_sets_grouped(grouped_packet_set_rows(p_role), grouped_packet_set_rows(t_role), args.minimum_packet_iou)
    summary = {
        "split_id": split_config.get("split_id") if split_config else None,
        "minimum_packet_iou": args.minimum_packet_iou,
        "known_packet_policy": args.known_packet_policy,
        "aggregate": aggregate,
        "by_application": by_application,
        "by_split_role": by_split_role,
    }
    write_json(summary, args.output_path)
    print(summary)


if __name__ == "__main__":
    main()
