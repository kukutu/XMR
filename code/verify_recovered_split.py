from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iframe_detector.packets import load_dataframe, write_json
from iframe_detector.splits import (
    DEFAULT_SPLIT_CONFIG_PATH,
    canonicalize_capture_columns,
    load_split_config,
    resolve_capture_record,
    summarize_split_table,
)


def pcap_presence(data_root: Path, config: dict) -> tuple[list[dict], list[dict]]:
    configured = []
    seen_names = set()
    for row in config.get("captures", []):
        path = data_root / str(row["pcap_filename"])
        seen_names.add(path.name.lower())
        actual_size = path.stat().st_size if path.exists() else None
        expected_size = row.get("expected_size_bytes")
        configured.append(
            {
                "capture_id": row["capture_id"],
                "application": row["application"],
                "role": row["role"],
                "pcap_filename": row["pcap_filename"],
                "present": path.exists(),
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "size_matches": bool(path.exists() and (expected_size is None or actual_size == expected_size)),
            }
        )
    extras = []
    for path in sorted(data_root.glob("*.pcapng")):
        if path.name.lower() in seen_names:
            continue
        record = resolve_capture_record(path.name, config)
        extras.append(
            {
                "pcap_filename": path.name,
                "size_bytes": path.stat().st_size,
                "mapped_capture_id": record.get("capture_id") if record else None,
                "included_in_recovered_split": bool(record),
            }
        )
    return configured, extras


def packet_table_summary(packet_root: Path, config: dict) -> dict:
    frames = []
    for path in sorted((packet_root / "packets").glob("*.packets.csv.gz")):
        df = load_dataframe(path)
        if not df.empty:
            frames.append(df)
    packet_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    packet_df = canonicalize_capture_columns(packet_df, config, drop_unassigned=False)
    return summarize_split_table(packet_df)


def label_table_summary(label_root: Path, config: dict) -> dict:
    path = label_root / "all_frames.csv.gz"
    label_df = load_dataframe(path) if path.exists() else pd.DataFrame()
    label_df = canonicalize_capture_columns(label_df, config, drop_unassigned=False)
    summary = summarize_split_table(label_df)
    if not label_df.empty and "is_keyframe" in label_df.columns:
        summary["keyframe_rows"] = int(label_df["is_keyframe"].astype(int).sum())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--packet-root", type=Path, default=Path("artifacts/rebuild_packets"))
    ap.add_argument("--label-root", type=Path, default=Path("artifacts/rebuild_labels"))
    ap.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    ap.add_argument("--output-path", type=Path, default=Path("artifacts/recovered_split_verification.json"))
    ap.add_argument("--fail-on-missing", action="store_true")
    args = ap.parse_args()

    config = load_split_config(args.split_config)
    if not config:
        raise SystemExit(f"split config not found: {args.split_config}")
    configured, extras = pcap_presence(args.data_root, config)
    missing = [row for row in configured if not row["present"] or not row["size_matches"]]
    summary = {
        "split_id": config.get("split_id"),
        "data_root": str(args.data_root),
        "configured_captures": configured,
        "extra_pcaps_not_in_recovered_primary_split": extras,
        "expected_prepared_summaries": config.get("expected_prepared_summaries", {}),
        "packet_tables": packet_table_summary(args.packet_root, config) if (args.packet_root / "packets").exists() else None,
        "label_tables": label_table_summary(args.label_root, config) if (args.label_root / "all_frames.csv.gz").exists() else None,
    }
    write_json(summary, args.output_path)
    print(f"split_id={summary['split_id']}")
    for row in configured:
        status = "ok" if row["present"] and row["size_matches"] else "missing_or_size_mismatch"
        print(f"{status}\t{row['role']}\t{row['capture_id']}\t{row['pcap_filename']}\t{row['actual_size_bytes']}")
    if extras:
        print("extra_pcaps_not_in_recovered_primary_split:")
        for row in extras:
            print(f"excluded\t{row['pcap_filename']}\t{row['size_bytes']}")
    print(f"wrote {args.output_path}")
    if args.fail_on_missing and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
