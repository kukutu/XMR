from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iframe_detector.metrics import match_packet_sets
from iframe_detector.packets import load_dataframe, write_json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-root", type=Path, default=Path("artifacts/rebuild_predictions"))
    ap.add_argument("--label-root", type=Path, default=Path("artifacts/rebuild_labels"))
    ap.add_argument("--minimum-packet-iou", type=float, default=0.9)
    ap.add_argument("--output-path", type=Path, default=Path("artifacts/rebuild_predictions/iframe_iou90_summary.json"))
    args = ap.parse_args()

    pred_path = args.prediction_root / "positives" / "predicted_iframes.csv.gz"
    truth_path = args.label_root / "all_frames.csv.gz"
    pred_df = load_dataframe(pred_path) if pred_path.exists() else pd.DataFrame()
    truth_df = load_dataframe(truth_path) if truth_path.exists() else pd.DataFrame()
    if not truth_df.empty:
        truth_df = truth_df[truth_df["is_keyframe"].astype(int).eq(1)].copy()
    by_role = {}
    aggregate = match_packet_sets(packet_set_rows(pred_df), packet_set_rows(truth_df), args.minimum_packet_iou)
    if not truth_df.empty and "application" in truth_df.columns:
        for app in sorted(truth_df["application"].astype(str).unique()):
            t_app = truth_df[truth_df["application"].astype(str).eq(app)]
            p_app = pred_df[pred_df["application"].astype(str).eq(app)] if not pred_df.empty else pred_df
            by_role[app] = match_packet_sets(packet_set_rows(p_app), packet_set_rows(t_app), args.minimum_packet_iou)
    summary = {
        "minimum_packet_iou": args.minimum_packet_iou,
        "aggregate": aggregate,
        "by_application": by_role,
    }
    write_json(summary, args.output_path)
    print(summary)


if __name__ == "__main__":
    main()

