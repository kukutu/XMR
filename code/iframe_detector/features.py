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


def _safe_divide_array(a, b):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return np.divide(a_arr, b_arr, out=np.zeros_like(a_arr, dtype=float), where=b_arr != 0)


def pairwise_gap_features(flow_df: pd.DataFrame, label_df: pd.DataFrame | None = None) -> pd.DataFrame:
    down = add_packet_context(flow_df).reset_index(drop=True)
    if len(down) < 2:
        return pd.DataFrame()
    group_map = build_packet_group_map(label_df) if label_df is not None else {}
    all_df = flow_df.sort_values("time_epoch").reset_index(drop=True)
    all_times = all_df["time_epoch"].to_numpy()
    uplink_flags = all_df["is_uplink"].astype(int).to_numpy()
    payloads = all_df["payload_len"].astype(float).to_numpy()
    uplink_packet_cumsum = np.concatenate([[0], np.cumsum(uplink_flags)])
    uplink_byte_cumsum = np.concatenate([[0.0], np.cumsum(payloads * uplink_flags)])
    bidi_packet_cumsum = np.arange(len(all_df) + 1)
    bidi_byte_cumsum = np.concatenate([[0.0], np.cumsum(payloads)])
    cur = down.iloc[:-1].reset_index(drop=True)
    nxt = down.iloc[1:].reset_index(drop=True)
    cur_time = cur["time_epoch"].astype(float).to_numpy()
    nxt_time = nxt["time_epoch"].astype(float).to_numpy()
    cur_payload = cur["payload_len"].astype(float).to_numpy()
    nxt_payload = nxt["payload_len"].astype(float).to_numpy()
    gap_us = (nxt_time - cur_time) * 1_000_000
    left = np.searchsorted(all_times, cur_time, side="right")
    right = np.searchsorted(all_times, nxt_time, side="right")
    uplink_packets = uplink_packet_cumsum[right] - uplink_packet_cumsum[left]
    uplink_bytes = uplink_byte_cumsum[right] - uplink_byte_cumsum[left]
    bidi_packets = bidi_packet_cumsum[right] - bidi_packet_cumsum[left]
    bidi_bytes = bidi_byte_cumsum[right] - bidi_byte_cumsum[left]
    rows = pd.DataFrame(
        {
            "capture_id": cur["capture_id"].to_numpy() if "capture_id" in cur.columns else "",
            "application": cur["application"].to_numpy() if "application" in cur.columns else "",
            "flow_key": cur["flow_key"].to_numpy(),
            "cur_packet_number": cur["packet_number"].astype(int).to_numpy(),
            "next_packet_number": nxt["packet_number"].astype(int).to_numpy(),
            "cur_payload": cur_payload,
            "next_payload": nxt_payload,
            "prev_payload": cur["prev_down_payload"].astype(float).to_numpy(),
            "gap_us": gap_us,
            "next_to_cur_payload_ratio": _safe_divide_array(nxt_payload, cur_payload),
            "cur_to_prev_payload_ratio": _safe_divide_array(cur_payload, cur["prev_down_payload"].astype(float).to_numpy()),
            "payload_delta": nxt_payload - cur_payload,
            "payload_delta_abs": np.abs(nxt_payload - cur_payload),
            "uplink_packets_between": uplink_packets.astype(int),
            "uplink_bytes_between": uplink_bytes.astype(int),
            "bidi_packets_between": bidi_packets.astype(int),
            "bidi_bytes_between": bidi_bytes.astype(int),
            "uplink_to_downlink_bytes_ratio": _safe_divide_array(uplink_bytes, cur_payload + nxt_payload),
        }
    )
    for w in ROLL_WINDOWS:
        mean = cur[f"roll{w}_payload_mean"].astype(float).to_numpy()
        iat_mean = cur[f"roll{w}_iat_mean_us"].astype(float).to_numpy()
        rows[f"roll{w}_payload_mean"] = mean
        rows[f"roll{w}_payload_max"] = cur[f"roll{w}_payload_max"].astype(float).to_numpy()
        rows[f"roll{w}_payload_std"] = cur[f"roll{w}_payload_std"].astype(float).to_numpy()
        rows[f"roll{w}_iat_mean_us"] = iat_mean
        rows[f"roll{w}_iat_max_us"] = cur[f"roll{w}_iat_max_us"].astype(float).to_numpy()
        rows[f"cur_to_roll{w}_payload_mean_ratio"] = _safe_divide_array(cur_payload, mean)
        rows[f"next_to_roll{w}_payload_mean_ratio"] = _safe_divide_array(nxt_payload, mean)
        rows[f"gap_to_roll{w}_iat_mean_ratio"] = _safe_divide_array(gap_us, iat_mean)
    if group_map:
        g1 = cur["packet_number"].map(group_map).fillna(0).astype(int).to_numpy()
        g2 = nxt["packet_number"].map(group_map).fillna(0).astype(int).to_numpy()
        rows["label_boundary"] = ((g1 > 0) & (g2 > 0) & (g1 != g2)).astype(int)
        rows["label_known"] = ((g1 > 0) & (g2 > 0)).astype(int)
    return rows


def frame_features_from_packets(flow_df: pd.DataFrame, packet_numbers: list[int], prefix: str = "") -> dict[str, float]:
    if flow_df.index.name == "packet_number":
        ordered = list(dict.fromkeys(int(p) for p in packet_numbers))
        existing = [p for p in ordered if p in flow_df.index]
        pkt = flow_df.loc[existing].sort_values("time_epoch") if existing else pd.DataFrame()
    else:
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
    lookup_cols = ["capture_id", "flow_key"] if "capture_id" in packet_df.columns and "capture_id" in label_df.columns else ["flow_key"]
    flow_lookup = {
        k: g.copy().set_index("packet_number", drop=False)
        for k, g in packet_df.groupby(lookup_cols)
    }
    for row in label_df.itertuples(index=False):
        packets = parse_packet_numbers(getattr(row, "packet_numbers"))
        flow_key = getattr(row, "flow_key")
        key = (getattr(row, "capture_id"), flow_key) if len(lookup_cols) == 2 else flow_key
        flow = flow_lookup.get(key)
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
    history_group_cols = ["capture_id", "flow_key"] if "capture_id" in df.columns else ["flow_key"]
    df = df.sort_values([*history_group_cols, "frame_index"]).reset_index(drop=True)
    for col in history_cols:
        grouped = df.groupby(history_group_cols)[col]
        df[f"prev_{col}"] = grouped.shift(1).fillna(0)
        df[f"roll5_{col}_mean"] = grouped.transform(lambda s: s.rolling(5, min_periods=1).mean())
        df[f"{col}_to_prev_ratio"] = _safe_divide_array(df[col].to_numpy(), df[f"prev_{col}"].to_numpy())
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
