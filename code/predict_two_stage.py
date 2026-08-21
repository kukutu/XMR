from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iframe_detector.features import frame_features_from_packets, pairwise_gap_features
from iframe_detector.models import load_model
from iframe_detector.packets import load_dataframe, save_dataframe, write_json


def reconstruct_frames(
    flow_df: pd.DataFrame,
    boundary_model,
    boundary_features: list[str],
    threshold: float,
    max_packets_per_frame: int = 96,
    max_frame_duration_s: float = 0.16,
) -> list[list[int]]:
    down = flow_df[flow_df["is_downlink"].eq(True)].sort_values("time_epoch")
    if down.empty:
        return []
    pairs = pairwise_gap_features(flow_df, None)
    scores_by_cur: dict[int, float] = {}
    if not pairs.empty:
        X = pairs.reindex(columns=boundary_features, fill_value=0)
        scores = boundary_model.predict_proba(X)[:, 1]
        for pkt, score in zip(pairs["cur_packet_number"], scores):
            scores_by_cur[int(pkt)] = float(score)
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


def score_reconstructed_frames(
    packet_df: pd.DataFrame,
    model_root: Path,
    output_root: Path,
    iframe_threshold: float | None = None,
) -> dict:
    boundary_model, boundary_meta = load_model(model_root / "boundary_model")
    iframe_model, iframe_meta = load_model(model_root / "iframe_model")
    boundary_features = list(boundary_meta["feature_names"])
    iframe_features = list(iframe_meta["feature_names"])
    boundary_threshold = float(boundary_meta.get("threshold", 0.5))
    iframe_threshold = float(iframe_threshold if iframe_threshold is not None else iframe_meta.get("threshold", 0.5))

    frame_rows = []
    positive_rows = []
    for flow_key, flow_df in packet_df.groupby("flow_key"):
        frames = reconstruct_frames(flow_df, boundary_model, boundary_features, boundary_threshold)
        for idx, packets in enumerate(frames):
            feat = frame_features_from_packets(flow_df, packets)
            if not feat:
                continue
            row = {
                "capture_id": str(flow_df["capture_id"].iloc[0]),
                "application": str(flow_df["application"].iloc[0]),
                "flow_key": flow_key,
                "frame_index": idx,
                "packet_numbers": "|".join(str(p) for p in packets),
                **feat,
            }
            X = pd.DataFrame([row]).reindex(columns=iframe_features, fill_value=0)
            score = float(iframe_model.predict_proba(X)[:, 1][0])
            row["iframe_score"] = score
            row["predicted_iframe"] = int(score >= iframe_threshold)
            frame_rows.append(row)
            if row["predicted_iframe"]:
                positive_rows.append(row)
    frames_df = pd.DataFrame(frame_rows)
    positives_df = pd.DataFrame(positive_rows)
    save_dataframe(frames_df, output_root / "frames" / "scored_frames.csv.gz")
    save_dataframe(positives_df, output_root / "positives" / "predicted_iframes.csv.gz")
    summary = {
        "frame_count": int(len(frames_df)),
        "predicted_positive_count": int(len(positives_df)),
        "boundary_threshold": boundary_threshold,
        "iframe_threshold": iframe_threshold,
    }
    write_json(summary, output_root / "prediction_summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet-root", type=Path, default=Path("artifacts/rebuild_packets"))
    ap.add_argument("--model-root", type=Path, default=Path("artifacts/rebuild_models"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/rebuild_predictions"))
    ap.add_argument("--capture-id", default=None)
    ap.add_argument("--iframe-threshold", type=float, default=None)
    args = ap.parse_args()

    packet_paths = sorted((args.packet_root / "packets").glob("*.packets.csv.gz"))
    frames = []
    for path in packet_paths:
        if args.capture_id and not path.name.startswith(args.capture_id):
            continue
        frames.append(load_dataframe(path))
    packets = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if packets.empty:
        raise SystemExit("no packet tables found")
    summary = score_reconstructed_frames(packets, args.model_root, args.output_root, args.iframe_threshold)
    print(summary)


if __name__ == "__main__":
    main()

