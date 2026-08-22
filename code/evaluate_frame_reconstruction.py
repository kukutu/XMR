from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iframe_detector.metrics import match_packet_sets_grouped
from iframe_detector.models import load_model
from iframe_detector.packets import load_dataframe, load_packet_feature_dataframe, save_dataframe, write_json
from iframe_detector.splits import DEFAULT_SPLIT_CONFIG_PATH, canonicalize_capture_columns, load_split_config
from iframe_detector.video import parse_packet_numbers
from predict_two_stage import reconstruct_frames


def load_all_packets(packet_root: Path) -> pd.DataFrame:
    frames = [load_packet_feature_dataframe(p) for p in sorted((packet_root / "packets").glob("*.packets.csv.gz"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_frame_packet_labels(label_root: Path) -> pd.DataFrame:
    stage_path = label_root / "frame_packet_labels.csv.gz"
    if stage_path.exists():
        return load_dataframe(stage_path)
    fallback_path = label_root / "all_frames.csv.gz"
    return load_dataframe(fallback_path) if fallback_path.exists() else pd.DataFrame()


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
        key = (str(getattr(row, "capture_id")), str(getattr(row, "flow_key")))
        known.setdefault(key, set()).update(parse_packet_numbers(getattr(row, "packet_numbers")))
    return known


def intersect_known_packets(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()
    known = known_packets_by_flow(label_df)
    rows = []
    for row in pred_df.itertuples(index=False):
        key = (str(getattr(row, "capture_id")), str(getattr(row, "flow_key")))
        packets = [p for p in parse_packet_numbers(getattr(row, "packet_numbers")) if p in known.get(key, set())]
        if not packets:
            continue
        payload = row._asdict()
        payload["packet_numbers"] = "|".join(str(p) for p in packets)
        rows.append(payload)
    return pd.DataFrame(rows)


def add_frame_count_alias(metrics: dict[str, float]) -> dict[str, float]:
    out = dict(metrics)
    out["true_frame_count"] = out.get("true_keyframe_count", 0)
    return out


def metrics_for(pred_df: pd.DataFrame, truth_df: pd.DataFrame, minimum_iou: float) -> dict[str, float]:
    return add_frame_count_alias(
        match_packet_sets_grouped(
            grouped_packet_set_rows(pred_df),
            grouped_packet_set_rows(truth_df),
            minimum_iou,
        )
    )


def reconstruct_all_labeled_flows(
    packets: pd.DataFrame,
    truth: pd.DataFrame,
    boundary_model,
    boundary_features: list[str],
    boundary_threshold: float,
    boundary_policy: str,
    lookahead_packets: int,
    max_packets_per_frame: int,
    max_frame_duration_s: float,
) -> pd.DataFrame:
    packet_lookup = {k: g.copy() for k, g in packets.groupby(["capture_id", "flow_key"])}
    pred_rows = []
    for (capture_id, flow_key), _ in truth.groupby(["capture_id", "flow_key"]):
        flow = packet_lookup.get((capture_id, flow_key))
        if flow is None or flow.empty:
            continue
        frames = reconstruct_frames(
            flow,
            boundary_model,
            boundary_features,
            boundary_threshold,
            max_packets_per_frame=max_packets_per_frame,
            max_frame_duration_s=max_frame_duration_s,
            boundary_policy=boundary_policy,
            lookahead_packets=lookahead_packets,
        )
        for idx, packet_numbers in enumerate(frames):
            pred_rows.append(
                {
                    "capture_id": capture_id,
                    "application": str(flow["application"].iloc[0]),
                    "split_role": str(flow["split_role"].iloc[0]) if "split_role" in flow.columns else "",
                    "flow_key": flow_key,
                    "frame_index": idx,
                    "packet_numbers": "|".join(str(x) for x in packet_numbers),
                }
            )
    return pd.DataFrame(pred_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-root", type=Path, default=Path("artifacts/rebuild_packets"))
    ap.add_argument("--label-root", type=Path, default=Path("artifacts/rebuild_labels"))
    ap.add_argument("--model-root", type=Path, default=Path("artifacts/rebuild_models"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/rebuild_frame_eval"))
    ap.add_argument("--minimum-packet-iou", type=float, default=0.9)
    ap.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    ap.add_argument("--boundary-threshold", type=float, default=None)
    ap.add_argument("--boundary-policy", choices=["threshold", "lookahead_peak"], default=None)
    ap.add_argument("--lookahead-packets", type=int, default=None)
    ap.add_argument("--max-packets-per-frame", type=int, default=96)
    ap.add_argument("--max-frame-duration-s", type=float, default=0.16)
    ap.add_argument("--known-packet-policy", choices=["full", "intersect"], default="full")
    ap.add_argument("--search-thresholds", action="store_true")
    args = ap.parse_args()

    split_config = load_split_config(args.split_config)
    packets = canonicalize_capture_columns(load_all_packets(args.packet_root), split_config, drop_unassigned=bool(split_config))
    truth = load_frame_packet_labels(args.label_root)
    truth = canonicalize_capture_columns(truth, split_config, drop_unassigned=bool(split_config))
    if packets.empty or truth.empty:
        raise SystemExit("packet table or frame labels are empty")

    boundary_model, boundary_meta = load_model(args.model_root / "boundary_model")
    boundary_features = list(boundary_meta["feature_names"])
    boundary_threshold = float(args.boundary_threshold if args.boundary_threshold is not None else boundary_meta.get("threshold", 0.5))
    boundary_policy = str(args.boundary_policy or boundary_meta.get("boundary_policy", "lookahead_peak"))
    lookahead_packets = int(args.lookahead_packets if args.lookahead_packets is not None else boundary_meta.get("lookahead_packets", 1))

    pred = reconstruct_all_labeled_flows(
        packets,
        truth,
        boundary_model,
        boundary_features,
        boundary_threshold,
        boundary_policy,
        lookahead_packets,
        args.max_packets_per_frame,
        args.max_frame_duration_s,
    )
    eval_pred = intersect_known_packets(pred, truth) if args.known_packet_policy == "intersect" else pred
    save_dataframe(pred, args.output_root / "reconstructed_frames.csv.gz")
    if args.known_packet_policy == "intersect":
        save_dataframe(eval_pred, args.output_root / "reconstructed_frames_known_intersect.csv.gz")

    threshold_search = []
    if args.search_thresholds:
        for threshold in [x / 100 for x in range(5, 96, 5)]:
            candidate = reconstruct_all_labeled_flows(
                packets,
                truth,
                boundary_model,
                boundary_features,
                threshold,
                boundary_policy,
                lookahead_packets,
                args.max_packets_per_frame,
                args.max_frame_duration_s,
            )
            eval_candidate = intersect_known_packets(candidate, truth) if args.known_packet_policy == "intersect" else candidate
            row = metrics_for(eval_candidate, truth, args.minimum_packet_iou)
            row["threshold"] = threshold
            threshold_search.append(row)

    summary = {
        "minimum_packet_iou": args.minimum_packet_iou,
        "boundary_threshold": boundary_threshold,
        "boundary_policy": boundary_policy,
        "lookahead_packets": lookahead_packets,
        "max_packets_per_frame": args.max_packets_per_frame,
        "max_frame_duration_s": args.max_frame_duration_s,
        "known_packet_policy": args.known_packet_policy,
        "threshold_search": threshold_search,
        "aggregate": metrics_for(eval_pred, truth, args.minimum_packet_iou),
        "by_application": {},
        "by_split_role": {},
    }
    for app in sorted(truth["application"].astype(str).unique()):
        summary["by_application"][app] = metrics_for(
            eval_pred[eval_pred["application"].astype(str).eq(app)] if not eval_pred.empty else eval_pred,
            truth[truth["application"].astype(str).eq(app)],
            args.minimum_packet_iou,
        )
    for role in sorted(truth["split_role"].astype(str).unique()):
        summary["by_split_role"][role] = metrics_for(
            eval_pred[eval_pred["split_role"].astype(str).eq(role)] if not eval_pred.empty else eval_pred,
            truth[truth["split_role"].astype(str).eq(role)],
            args.minimum_packet_iou,
        )
    write_json(summary, args.output_root / "frame_reconstruction_iou_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
