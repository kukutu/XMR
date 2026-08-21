from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iframe_detector.features import select_media_flows
from iframe_detector.packets import load_dataframe, save_dataframe, write_json
from iframe_detector.splits import DEFAULT_SPLIT_CONFIG_PATH, canonicalize_capture_columns, load_split_config
from iframe_detector.video import derive_labels_for_flow, labels_to_dataframe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-root", type=Path, default=Path("artifacts/rebuild_packets"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/rebuild_labels"))
    ap.add_argument("--top-flows", type=int, default=5)
    ap.add_argument("--min-downlink-bytes", type=int, default=64_000)
    ap.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    ap.add_argument(
        "--allow-weak-annexb",
        action="store_true",
        help="Allow raw Annex-B start-code labels. This can false-trigger on encrypted/random payloads.",
    )
    args = ap.parse_args()

    split_config = load_split_config(args.split_config)
    packet_paths = sorted((args.packet_root / "packets").glob("*.packets.csv.gz"))
    all_label_frames = []
    summaries = []
    for packet_path in packet_paths:
        packets = canonicalize_capture_columns(load_dataframe(packet_path), split_config, drop_unassigned=True)
        if packets.empty:
            continue
        capture_id = str(packets["capture_id"].iloc[0])
        application = str(packets["application"].iloc[0])
        media = select_media_flows(packets, args.top_flows, args.min_downlink_bytes)
        capture_labels = []
        for flow in media.itertuples(index=False):
            flow_df = packets[packets["flow_key"].eq(flow.flow_key)].copy()
            stream_id = f"{capture_id}__{flow.flow_key}"
            labels = derive_labels_for_flow(flow_df, stream_id, allow_weak_annexb=args.allow_weak_annexb)
            label_df = labels_to_dataframe(labels)
            if not label_df.empty:
                label_df["capture_id"] = capture_id
                label_df["application"] = application
                label_df["flow_key"] = flow.flow_key
                capture_labels.append(label_df)
        out_df = pd.concat(capture_labels, ignore_index=True) if capture_labels else pd.DataFrame()
        out_path = args.output_root / "frames" / f"{capture_id}.frames.csv.gz"
        save_dataframe(out_df, out_path)
        if not out_df.empty:
            all_label_frames.append(out_df)
        summaries.append(
            {
                "capture_id": capture_id,
                "application": application,
                "candidate_flow_count": int(len(media)),
                "labeled_frame_count": int(len(out_df)),
                "keyframe_count": int(out_df["is_keyframe"].sum()) if not out_df.empty else 0,
                "label_path": str(out_path),
                "split_role": str(packets["split_role"].iloc[0]) if "split_role" in packets.columns else None,
            }
        )
        print(f"labels {capture_id}: {len(out_df)} frames")
    all_df = pd.concat(all_label_frames, ignore_index=True) if all_label_frames else pd.DataFrame()
    save_dataframe(all_df, args.output_root / "all_frames.csv.gz")
    write_json({"split_id": split_config.get("split_id") if split_config else None, "captures": summaries}, args.output_root / "label_summary.json")


if __name__ == "__main__":
    main()
