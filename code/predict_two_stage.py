from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iframe_detector.features import frame_features_from_packets, pairwise_gap_features
from iframe_detector.models import load_model
from iframe_detector.packets import load_packet_feature_dataframe, save_dataframe, write_json
from iframe_detector.splits import DEFAULT_SPLIT_CONFIG_PATH, canonicalize_capture_columns, load_split_config


def reconstruct_frames(
    flow_df: pd.DataFrame,
    boundary_model,
    boundary_features: list[str],
    threshold: float,
    max_packets_per_frame: int = 96,
    max_frame_duration_s: float = 0.16,
    boundary_policy: str = "lookahead_peak",
    lookahead_packets: int = 1,
) -> list[list[int]]:
    down = flow_df[flow_df["is_downlink"].eq(True)].sort_values("time_epoch")
    if down.empty:
        return []
    pairs = pairwise_gap_features(flow_df, None)
    scores_by_cur: dict[int, float] = {}
    if not pairs.empty:
        X = pairs.reindex(columns=boundary_features, fill_value=0)
        scores = boundary_model.predict_proba(X)[:, 1]
        cut_flags = scores >= threshold
        if boundary_policy == "lookahead_peak":
            cut_flags = cut_flags.copy()
            for i, score in enumerate(scores):
                if score < threshold:
                    cut_flags[i] = False
                    continue
                left = max(0, i - lookahead_packets)
                right = min(len(scores), i + lookahead_packets + 1)
                cut_flags[i] = bool(score >= scores[left:right].max())
        for pkt, score, flag in zip(pairs["cur_packet_number"], scores, cut_flags):
            scores_by_cur[int(pkt)] = float(score) if flag else 0.0
    frames: list[list[int]] = []
    current: list[int] = []
    start_time = None
    for row in down.itertuples(index=False):
        pkt = int(row.packet_number)
        t = float(row.time_epoch)
        if not current:
            start_time = t
        current.append(pkt)
        duration = 0.0 if start_time is None else t - start_time
        should_cut = scores_by_cur.get(pkt, 0.0) >= threshold
        if len(current) >= max_packets_per_frame or duration >= max_frame_duration_s or should_cut:
            frames.append(current)
            current = []
            start_time = None
    if current:
        frames.append(current)
    return frames


def _adaptive_radius_for_flow(group: pd.DataFrame, threshold: float, min_frames: int, max_frames: int) -> int:
    candidates = group[group["iframe_score"].ge(threshold)].sort_values("frame_index")
    if len(candidates) < 3:
        return min_frames
    strong_cutoff = max(threshold, float(candidates["iframe_score"].quantile(0.75)))
    strong = candidates[candidates["iframe_score"].ge(strong_cutoff)]["frame_index"].astype(int).to_numpy()
    if len(strong) < 3:
        strong = candidates["frame_index"].astype(int).to_numpy()
    gaps = pd.Series(strong).diff().dropna()
    gaps = gaps[gaps.gt(0)]
    if gaps.empty:
        return min_frames
    estimated_period = int(gaps.median())
    return max(min_frames, min(max_frames, int(round(estimated_period * 0.45))))


def apply_iframe_postprocess(
    frames_df: pd.DataFrame,
    threshold: float,
    policy: str = "raw",
    gop_min_frames: int = 8,
    gop_max_frames: int = 90,
) -> pd.DataFrame:
    if frames_df.empty:
        out = frames_df.copy()
        out["predicted_iframe"] = []
        return out
    out = frames_df.copy()
    out["predicted_iframe"] = 0
    if policy == "raw":
        out.loc[out["iframe_score"].ge(threshold), "predicted_iframe"] = 1
        return out
    if policy != "adaptive_gop_nms":
        raise ValueError(f"unknown I-frame postprocess policy: {policy}")
    group_cols = ["capture_id", "flow_key"]
    for _, group in out.groupby(group_cols, sort=False):
        candidates = group[group["iframe_score"].ge(threshold)].copy()
        if candidates.empty:
            continue
        radius = _adaptive_radius_for_flow(group, threshold, gop_min_frames, gop_max_frames)
        selected: list[int] = []
        for row in candidates.sort_values(["iframe_score", "frame_index"], ascending=[False, True]).itertuples():
            frame_idx = int(row.frame_index)
            if all(abs(frame_idx - keep) > radius for keep in selected):
                selected.append(frame_idx)
        mask = out.index.isin(group[group["frame_index"].isin(selected)].index)
        out.loc[mask, "predicted_iframe"] = 1
    return out


def score_reconstructed_frames(
    packet_df: pd.DataFrame,
    model_root: Path,
    output_root: Path,
    iframe_threshold: float | None = None,
    boundary_threshold_override: float | None = None,
    boundary_policy: str | None = None,
    lookahead_packets: int | None = None,
    max_packets_per_frame: int = 96,
    max_frame_duration_s: float = 0.16,
    iframe_postprocess_policy: str = "raw",
    gop_min_frames: int = 8,
    gop_max_frames: int = 90,
) -> dict:
    boundary_model, boundary_meta = load_model(model_root / "boundary_model")
    iframe_model, iframe_meta = load_model(model_root / "iframe_model")
    boundary_features = list(boundary_meta["feature_names"])
    iframe_features = list(iframe_meta["feature_names"])
    boundary_threshold = float(boundary_threshold_override if boundary_threshold_override is not None else boundary_meta.get("threshold", 0.5))
    boundary_policy = str(boundary_policy or boundary_meta.get("boundary_policy", "lookahead_peak"))
    lookahead_packets = int(lookahead_packets if lookahead_packets is not None else boundary_meta.get("lookahead_packets", 1))
    iframe_threshold = float(iframe_threshold if iframe_threshold is not None else iframe_meta.get("threshold", 0.5))

    frame_rows = []
    group_cols = ["capture_id", "flow_key"] if "capture_id" in packet_df.columns else ["flow_key"]
    for group_key, flow_df in packet_df.groupby(group_cols):
        flow_key = group_key[1] if isinstance(group_key, tuple) else group_key
        flow_indexed = flow_df.set_index("packet_number", drop=False)
        frames = reconstruct_frames(
            flow_df,
            boundary_model,
            boundary_features,
            boundary_threshold,
            max_packets_per_frame=max_packets_per_frame,
            max_frame_duration_s=max_frame_duration_s,
            boundary_policy=boundary_policy,
            lookahead_packets=lookahead_packets,
        )
        for idx, packets in enumerate(frames):
            feat = frame_features_from_packets(flow_indexed, packets)
            if not feat:
                continue
            existing = [p for p in dict.fromkeys(int(x) for x in packets) if p in flow_indexed.index]
            frame_packets = flow_indexed.loc[existing] if existing else pd.DataFrame()
            row = {
                "capture_id": str(flow_df["capture_id"].iloc[0]),
                "application": str(flow_df["application"].iloc[0]),
                "split_role": str(flow_df["split_role"].iloc[0]) if "split_role" in flow_df.columns else "",
                "flow_key": flow_key,
                "frame_index": idx,
                "packet_numbers": "|".join(str(p) for p in packets),
                "frame_start_time_epoch": float(frame_packets["time_epoch"].min()) if not frame_packets.empty else 0.0,
                "frame_end_time_epoch": float(frame_packets["time_epoch"].max()) if not frame_packets.empty else 0.0,
                **feat,
            }
            X = pd.DataFrame([row]).reindex(columns=iframe_features, fill_value=0)
            score = float(iframe_model.predict_proba(X)[:, 1][0])
            row["iframe_score"] = score
            frame_rows.append(row)
    frames_df = pd.DataFrame(frame_rows)
    frames_df = apply_iframe_postprocess(
        frames_df,
        iframe_threshold,
        policy=iframe_postprocess_policy,
        gop_min_frames=gop_min_frames,
        gop_max_frames=gop_max_frames,
    )
    positive_rows = frames_df[frames_df["predicted_iframe"].astype(int).eq(1)].copy() if not frames_df.empty else pd.DataFrame()
    positives_df = pd.DataFrame(positive_rows)
    save_dataframe(frames_df, output_root / "frames" / "scored_frames.csv.gz")
    save_dataframe(positives_df, output_root / "positives" / "predicted_iframes.csv.gz")
    summary = {
        "frame_count": int(len(frames_df)),
        "predicted_positive_count": int(len(positives_df)),
        "boundary_threshold": boundary_threshold,
        "boundary_policy": boundary_policy,
        "lookahead_packets": lookahead_packets,
        "max_packets_per_frame": max_packets_per_frame,
        "max_frame_duration_s": max_frame_duration_s,
        "iframe_threshold": iframe_threshold,
        "iframe_postprocess_policy": iframe_postprocess_policy,
        "gop_min_frames": gop_min_frames,
        "gop_max_frames": gop_max_frames,
    }
    write_json(summary, output_root / "prediction_summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-root", type=Path, default=Path("artifacts/rebuild_packets"))
    ap.add_argument("--model-root", type=Path, default=Path("artifacts/rebuild_models"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/rebuild_predictions"))
    ap.add_argument("--capture-id", default=None)
    ap.add_argument("--boundary-threshold", type=float, default=None)
    ap.add_argument("--iframe-threshold", type=float, default=None)
    ap.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG_PATH)
    ap.add_argument("--boundary-policy", choices=["threshold", "lookahead_peak"], default=None)
    ap.add_argument("--lookahead-packets", type=int, default=None)
    ap.add_argument("--max-packets-per-frame", type=int, default=96)
    ap.add_argument("--max-frame-duration-s", type=float, default=0.16)
    ap.add_argument("--iframe-postprocess-policy", choices=["raw", "adaptive_gop_nms"], default="raw")
    ap.add_argument("--gop-min-frames", type=int, default=8)
    ap.add_argument("--gop-max-frames", type=int, default=90)
    args = ap.parse_args()

    split_config = load_split_config(args.split_config)
    packet_paths = sorted((args.packet_root / "packets").glob("*.packets.csv.gz"))
    frames = []
    for path in packet_paths:
        if args.capture_id and not path.name.startswith(args.capture_id):
            continue
        frames.append(load_packet_feature_dataframe(path))
    packets = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    packets = canonicalize_capture_columns(packets, split_config, drop_unassigned=bool(split_config))
    if packets.empty:
        raise SystemExit("no packet tables found")
    summary = score_reconstructed_frames(
        packets,
        args.model_root,
        args.output_root,
        args.iframe_threshold,
        boundary_threshold_override=args.boundary_threshold,
        boundary_policy=args.boundary_policy,
        lookahead_packets=args.lookahead_packets,
        max_packets_per_frame=args.max_packets_per_frame,
        max_frame_duration_s=args.max_frame_duration_s,
        iframe_postprocess_policy=args.iframe_postprocess_policy,
        gop_min_frames=args.gop_min_frames,
        gop_max_frames=args.gop_max_frames,
    )
    print(summary)


if __name__ == "__main__":
    main()
