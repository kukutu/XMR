from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .video import parse_packet_numbers


ROLL_WINDOWS = (3, 5, 10, 20, 50)


def select_media_flows(packet_df: pd.DataFrame, top_n: int = 3, min_downlink_bytes: int = 64_000) -> pd.DataFrame:
    down = packet_df[packet_df["is_downlink"].eq(True)]
    if down.empty:
        return pd.DataFrame(columns=["flow_key", "downlink_bytes", "packet_count"])
    grouped = (
        down.groupby("flow_key")
        .agg(downlink_bytes=("payload_len", "sum"), packet_count=("packet_number", "count"))
        .reset_index()
        .sort_values("downlink_bytes", ascending=False)
    )
    grouped = grouped[grouped["downlink_bytes"] >= min_downlink_bytes]
    return grouped.head(top_n)


def add_packet_context(flow_df: pd.DataFrame) -> pd.DataFrame:
    df = flow_df.sort_values("time_epoch").copy()
    df["prev_time"] = df["time_epoch"].shift(1)
    df["iat_us"] = ((df["time_epoch"] - df["prev_time"]) * 1_000_000).fillna(0).clip(lower=0)
    down = df[df["is_downlink"].eq(True)].copy()
    down["prev_down_time"] = down["time_epoch"].shift(1)
    down["prev_down_payload"] = down["payload_len"].shift(1).fillna(0)
    down["down_iat_us"] = ((down["time_epoch"] - down["prev_down_time"]) * 1_000_000).fillna(0).clip(lower=0)
    for w in ROLL_WINDOWS:
        down[f"roll{w}_payload_mean"] = down["payload_len"].rolling(w, min_periods=1).mean()
        down[f"roll{w}_payload_max"] = down["payload_len"].rolling(w, min_periods=1).max()
        down[f"roll{w}_payload_std"] = down["payload_len"].rolling(w, min_periods=1).std().fillna(0)
        down[f"roll{w}_iat_mean_us"] = down["down_iat_us"].rolling(w, min_periods=1).mean()
        down[f"roll{w}_iat_max_us"] = down["down_iat_us"].rolling(w, min_periods=1).max()
    return down


def build_packet_group_map(label_df: pd.DataFrame) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if label_df.empty:
        return mapping
    for group_id, row in enumerate(label_df.sort_values(["stream_id", "frame_index"]).itertuples(index=False), start=1):
        for pkt in parse_packet_numbers(getattr(row, "packet_numbers")):
            mapping[pkt] = group_id
    return mapping


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def pairwise_gap_features(flow_df: pd.DataFrame, label_df: pd.DataFrame | None = None) -> pd.DataFrame:
    down = add_packet_context(flow_df).reset_index(drop=True)
    if len(down) < 2:
        return pd.DataFrame()
    rows = []
    group_map = build_packet_group_map(label_df) if label_df is not None else {}
    all_df = flow_df.sort_values("time_epoch").reset_index(drop=True)
    all_times = all_df["time_epoch"].to_numpy()
    uplink_flags = all_df["is_uplink"].astype(int).to_numpy()
    payloads = all_df["payload_len"].astype(float).to_numpy()
    uplink_packet_cumsum = np.concatenate([[0], np.cumsum(uplink_flags)])
    uplink_byte_cumsum = np.concatenate([[0.0], np.cumsum(payloads * uplink_flags)])
    bidi_packet_cumsum = np.arange(len(all_df) + 1)
    bidi_byte_cumsum = np.concatenate([[0.0], np.cumsum(payloads)])
    for i in range(len(down) - 1):
        cur = down.iloc[i]
        nxt = down.iloc[i + 1]
        left = int(np.searchsorted(all_times, float(cur["time_epoch"]), side="right"))
        right = int(np.searchsorted(all_times, float(nxt["time_epoch"]), side="right"))
        uplink_packets = int(uplink_packet_cumsum[right] - uplink_packet_cumsum[left])
        uplink_bytes = float(uplink_byte_cumsum[right] - uplink_byte_cumsum[left])
        bidi_packets = int(bidi_packet_cumsum[right] - bidi_packet_cumsum[left])
        bidi_bytes = float(bidi_byte_cumsum[right] - bidi_byte_cumsum[left])
        row = {
            "capture_id": cur.get("capture_id", ""),
            "application": cur.get("application", ""),
            "flow_key": cur["flow_key"],
            "cur_packet_number": int(cur["packet_number"]),
            "next_packet_number": int(nxt["packet_number"]),
            "cur_payload": float(cur["payload_len"]),
            "next_payload": float(nxt["payload_len"]),
            "prev_payload": float(cur["prev_down_payload"]),
            "gap_us": float((nxt["time_epoch"] - cur["time_epoch"]) * 1_000_000),
            "next_to_cur_payload_ratio": _safe_ratio(nxt["payload_len"], cur["payload_len"]),
            "cur_to_prev_payload_ratio": _safe_ratio(cur["payload_len"], cur["prev_down_payload"]),
            "payload_delta": float(nxt["payload_len"] - cur["payload_len"]),
            "payload_delta_abs": float(abs(nxt["payload_len"] - cur["payload_len"])),
            "uplink_packets_between": uplink_packets,
            "uplink_bytes_between": int(uplink_bytes),
            "bidi_packets_between": bidi_packets,
            "bidi_bytes_between": int(bidi_bytes),
            "uplink_to_downlink_bytes_ratio": _safe_ratio(uplink_bytes, cur["payload_len"] + nxt["payload_len"]),
        }
        for w in ROLL_WINDOWS:
            mean = float(cur[f"roll{w}_payload_mean"])
            iat_mean = float(cur[f"roll{w}_iat_mean_us"])
            row[f"roll{w}_payload_mean"] = mean
            row[f"roll{w}_payload_max"] = float(cur[f"roll{w}_payload_max"])
            row[f"roll{w}_payload_std"] = float(cur[f"roll{w}_payload_std"])
            row[f"roll{w}_iat_mean_us"] = iat_mean
            row[f"roll{w}_iat_max_us"] = float(cur[f"roll{w}_iat_max_us"])
            row[f"cur_to_roll{w}_payload_mean_ratio"] = _safe_ratio(cur["payload_len"], mean)
            row[f"next_to_roll{w}_payload_mean_ratio"] = _safe_ratio(nxt["payload_len"], mean)
            row[f"gap_to_roll{w}_iat_mean_ratio"] = _safe_ratio(row["gap_us"], iat_mean)
        if group_map:
            g1 = group_map.get(int(cur["packet_number"]), 0)
            g2 = group_map.get(int(nxt["packet_number"]), 0)
            row["label_boundary"] = int(g1 > 0 and g2 > 0 and g1 != g2)
            row["label_known"] = int(g1 > 0 and g2 > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def frame_features_from_packets(flow_df: pd.DataFrame, packet_numbers: list[int], prefix: str = "") -> dict[str, float]:
    pkt = flow_df[flow_df["packet_number"].isin(packet_numbers)].sort_values("time_epoch")
    if pkt.empty:
        return {}
    down = pkt[pkt["is_downlink"].eq(True)]
    up = pkt[pkt["is_uplink"].eq(True)]
    duration_us = max((pkt["time_epoch"].max() - pkt["time_epoch"].min()) * 1_000_000, 0.0)
    payload = pkt["payload_len"].astype(float)
    down_bytes = float(down["payload_len"].sum())
    up_bytes = float(up["payload_len"].sum())
    result = {
        f"{prefix}packet_count": float(len(pkt)),
        f"{prefix}down_packet_count": float(len(down)),
        f"{prefix}up_packet_count": float(len(up)),
        f"{prefix}byte_count": float(payload.sum()),
        f"{prefix}down_byte_count": down_bytes,
        f"{prefix}up_byte_count": up_bytes,
        f"{prefix}duration_us": float(duration_us),
        f"{prefix}mean_packet_size": float(payload.mean()),
        f"{prefix}max_packet_size": float(payload.max()),
        f"{prefix}std_packet_size": float(payload.std(ddof=0) if len(payload) > 1 else 0.0),
        f"{prefix}up_down_packet_ratio": _safe_ratio(len(up), len(down)),
        f"{prefix}up_down_byte_ratio": _safe_ratio(up_bytes, down_bytes),
        f"{prefix}bytes_per_us": _safe_ratio(payload.sum(), duration_us),
    }
    return result


def build_frame_training_table(packet_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if label_df.empty:
        return pd.DataFrame()
    flow_lookup = {k: g.copy() for k, g in packet_df.groupby("flow_key")}
    for row in label_df.itertuples(index=False):
        packets = parse_packet_numbers(getattr(row, "packet_numbers"))
        flow_key = getattr(row, "flow_key")
        flow = flow_lookup.get(flow_key)
        if flow is None or not packets:
            continue
        feat = frame_features_from_packets(flow, packets)
        if not feat:
            continue
        feat.update(
            {
                "capture_id": getattr(row, "capture_id"),
                "application": getattr(row, "application"),
                "flow_key": flow_key,
                "stream_id": getattr(row, "stream_id"),
                "frame_index": int(getattr(row, "frame_index")),
                "packet_numbers": getattr(row, "packet_numbers"),
                "label_iframe": int(getattr(row, "is_keyframe")),
            }
        )
        rows.append(feat)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    history_cols = [
        "byte_count",
        "packet_count",
        "down_byte_count",
        "duration_us",
        "bytes_per_us",
    ]
    df = df.sort_values(["flow_key", "frame_index"]).reset_index(drop=True)
    for col in history_cols:
        df[f"prev_{col}"] = df.groupby("flow_key")[col].shift(1).fillna(0)
        df[f"roll5_{col}_mean"] = df.groupby("flow_key")[col].transform(lambda s: s.rolling(5, min_periods=1).mean())
        df[f"{col}_to_prev_ratio"] = df.apply(lambda r: _safe_ratio(r[col], r[f"prev_{col}"]), axis=1)
    return df


def numeric_feature_columns(df: pd.DataFrame, label_cols: Iterable[str] = ()) -> list[str]:
    excluded = {
        "capture_id",
        "application",
        "flow_key",
        "stream_id",
        "packet_numbers",
        "cur_packet_number",
        "next_packet_number",
        "label_known",
        *label_cols,
    }
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols
