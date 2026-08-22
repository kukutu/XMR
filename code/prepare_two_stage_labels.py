from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iframe_detector.packets import load_dataframe, save_dataframe, write_json
from iframe_detector.splits import DEFAULT_SPLIT_CONFIG_PATH, canonicalize_capture_columns, load_split_config, summarize_split_table


FRAME_PACKET_COLUMNS = [
    "capture_id",
    "application",
    "flow_key",
    "stream_id",
    "frame_index",
    "packet_numbers",
    "packet_count",
    "label_source",
]

FRAME_TYPE_COLUMNS = [
    "capture_id",
    "application",
    "flow_key",
    "stream_id",
    "frame_index",
    "packet_numbers",
    "packet_count",
    "is_keyframe",
    "codec",
    "timestamp_ms",
    "label_source",
]


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def filter_label_sources(
    df: pd.DataFrame,
    allow_sources: list[str] | None,
    exclude_sources: list[str] | None,
) -> pd.DataFrame:
    if df.empty or "label_source" not in df.columns:
        return df.copy()
    out = df.copy()
    source = out["label_source"].astype(str).str.lower()
    if allow_sources:
        allowed = {str(x).lower() for x in allow_sources}
        out = out[source.isin(allowed)].copy()
        source = out["label_source"].astype(str).str.lower()
    if exclude_sources:
        excluded = {str(x).lower() for x in exclude_sources}
        out = out[~source.isin(excluded)].copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-label-root", type=Path, default=Path("artifacts/rebuild_labels"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/rebuild_two_stage_labels"))
    ap.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    ap.add_argument("--allow-label-source", action="append", default=None)
    ap.add_argument("--exclude-label-source", action="append", default=None)
    args = ap.parse_args()

    split_config = load_split_config(args.split_config)
    input_path = args.input_label_root / "all_frames.csv.gz"
    labels = load_dataframe(input_path) if input_path.exists() else pd.DataFrame()
    labels = canonicalize_capture_columns(labels, split_config, drop_unassigned=bool(split_config))
    filtered = filter_label_sources(labels, args.allow_label_source, args.exclude_label_source)

    frame_packet = filtered[existing_columns(filtered, FRAME_PACKET_COLUMNS)].copy() if not filtered.empty else pd.DataFrame()
    frame_type = filtered[existing_columns(filtered, FRAME_TYPE_COLUMNS)].copy() if not filtered.empty else pd.DataFrame()

    save_dataframe(filtered, args.output_root / "all_frames.csv.gz")
    save_dataframe(frame_packet, args.output_root / "frame_packet_labels.csv.gz")
    save_dataframe(frame_type, args.output_root / "frame_type_labels.csv.gz")

    summary = {
        "input_label_root": str(args.input_label_root),
        "output_root": str(args.output_root),
        "allow_label_source": args.allow_label_source,
        "exclude_label_source": args.exclude_label_source,
        "input_rows": int(len(labels)),
        "filtered_rows": int(len(filtered)),
        "frame_packet_label_rows": int(len(frame_packet)),
        "frame_type_label_rows": int(len(frame_type)),
        "keyframe_rows": int(frame_type["is_keyframe"].sum()) if not frame_type.empty and "is_keyframe" in frame_type.columns else 0,
        "split": summarize_split_table(filtered),
        "by_label_source": {
            str(k): int(v)
            for k, v in filtered["label_source"].value_counts(dropna=False).sort_index().items()
        }
        if not filtered.empty and "label_source" in filtered.columns
        else {},
    }
    write_json(summary, args.output_root / "two_stage_label_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
