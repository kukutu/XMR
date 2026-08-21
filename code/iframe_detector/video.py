from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .packets import payload_bytes


@dataclass
class ByteSpan:
    packet_number: int
    start: int
    end: int


@dataclass
class VideoFrameLabel:
    stream_id: str
    codec: str
    frame_index: int
    timestamp_ms: int | None
    byte_start: int
    byte_end: int
    packet_numbers: list[int]
    is_keyframe: bool
    label_source: str


def reassemble_tcp_direction(flow_df: pd.DataFrame, downlink: bool = True) -> tuple[bytes, list[ByteSpan]]:
    rows = flow_df[flow_df["transport"].eq("tcp")].copy()
    rows = rows[rows["is_downlink"].eq(bool(downlink))]
    rows = rows[rows["tcp_len"].fillna(0).astype(int) > 0]
    if rows.empty:
        return b"", []
    rows["_payload"] = rows.apply(payload_bytes, axis=1)
    rows = rows[rows["_payload"].map(len) > 0]
    rows = rows.sort_values(["tcp_seq", "packet_number"])
    chunks: list[bytes] = []
    spans: list[ByteSpan] = []
    stream_pos = 0
    seen: set[tuple[int, int]] = set()
    for _, row in rows.iterrows():
        seq = int(row["tcp_seq"])
        data = row["_payload"]
        key = (seq, len(data))
        if key in seen:
            continue
        seen.add(key)
        start = stream_pos
        chunks.append(data)
        stream_pos += len(data)
        spans.append(ByteSpan(int(row["packet_number"]), start, stream_pos))
    return b"".join(chunks), spans


def packets_for_span(spans: list[ByteSpan], start: int, end: int) -> list[int]:
    packets = [s.packet_number for s in spans if s.start < end and s.end > start]
    return sorted(set(packets))


def parse_flv_frames(data: bytes, spans: list[ByteSpan], stream_id: str) -> list[VideoFrameLabel]:
    frames: list[VideoFrameLabel] = []
    header = data.find(b"FLV")
    if header < 0 or header + 13 > len(data):
        return frames
    data_offset = int.from_bytes(data[header + 5 : header + 9], "big", signed=False)
    pos = header + data_offset + 4
    frame_index = 0
    while pos + 15 <= len(data):
        tag_start = pos
        tag_type = data[pos]
        data_size = int.from_bytes(data[pos + 1 : pos + 4], "big", signed=False)
        ts = int.from_bytes(data[pos + 4 : pos + 7], "big", signed=False) | (data[pos + 7] << 24)
        payload_start = pos + 11
        payload_end = payload_start + data_size
        next_pos = payload_end + 4
        if payload_end > len(data) or next_pos > len(data):
            break
        if tag_type == 9 and data_size >= 1:
            first = data[payload_start]
            frame_type = first >> 4
            codec_id = first & 0x0F
            codec = {7: "avc", 12: "hevc"}.get(codec_id, f"flv_codec_{codec_id}")
            packets = packets_for_span(spans, tag_start, next_pos)
            if packets:
                frames.append(
                    VideoFrameLabel(
                        stream_id=stream_id,
                        codec=codec,
                        frame_index=frame_index,
                        timestamp_ms=ts,
                        byte_start=tag_start,
                        byte_end=next_pos,
                        packet_numbers=packets,
                        is_keyframe=(frame_type == 1),
                        label_source="flv",
                    )
                )
                frame_index += 1
        pos = next_pos
    return frames


def _annexb_start_codes(data: bytes) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    i = 0
    n = len(data)
    while i + 3 < n:
        if data[i : i + 3] == b"\x00\x00\x01":
            starts.append((i, i + 3))
            i += 3
        elif i + 4 < n and data[i : i + 4] == b"\x00\x00\x00\x01":
            starts.append((i, i + 4))
            i += 4
        else:
            i += 1
    return starts


def parse_annexb_frames(data: bytes, spans: list[ByteSpan], stream_id: str) -> list[VideoFrameLabel]:
    starts = _annexb_start_codes(data)
    if not starts:
        return []
    frames: list[VideoFrameLabel] = []
    for idx, (sc_start, payload_start) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(data)
        if payload_start >= end:
            continue
        h0 = data[payload_start]
        avc_type = h0 & 0x1F
        hevc_type = (h0 >> 1) & 0x3F
        codec = "avc" if avc_type in range(1, 32) else "unknown"
        is_key = avc_type == 5
        if hevc_type in range(0, 64):
            # HEVC VPS/SPS/PPS/IDR types are common in Annex-B streams. If an
            # HEVC IDR is present, prefer the HEVC interpretation.
            if hevc_type in {16, 17, 18, 19, 20, 21, 32, 33, 34}:
                codec = "hevc"
                is_key = hevc_type in {16, 17, 18, 19, 20, 21}
        if avc_type not in {1, 5} and not is_key:
            continue
        packets = packets_for_span(spans, sc_start, end)
        if packets:
            frames.append(
                VideoFrameLabel(
                    stream_id=stream_id,
                    codec=codec,
                    frame_index=len(frames),
                    timestamp_ms=None,
                    byte_start=sc_start,
                    byte_end=end,
                    packet_numbers=packets,
                    is_keyframe=is_key,
                    label_source="annexb",
                )
            )
    return frames


def labels_to_dataframe(labels: list[VideoFrameLabel]) -> pd.DataFrame:
    rows = []
    for label in labels:
        rows.append(
            {
                "stream_id": label.stream_id,
                "codec": label.codec,
                "frame_index": label.frame_index,
                "timestamp_ms": label.timestamp_ms if label.timestamp_ms is not None else "",
                "byte_start": label.byte_start,
                "byte_end": label.byte_end,
                "packet_numbers": "|".join(str(p) for p in label.packet_numbers),
                "packet_count": len(label.packet_numbers),
                "is_keyframe": int(label.is_keyframe),
                "label_source": label.label_source,
            }
        )
    return pd.DataFrame(rows)


def parse_packet_numbers(value: str | float | int) -> list[int]:
    if value is None:
        return []
    text = str(value)
    if not text or text == "nan":
        return []
    return [int(x) for x in text.split("|") if x]


def derive_labels_for_flow(flow_df: pd.DataFrame, stream_id: str, allow_weak_annexb: bool = False) -> list[VideoFrameLabel]:
    data, spans = reassemble_tcp_direction(flow_df, downlink=True)
    if not data:
        return []
    labels = parse_flv_frames(data, spans, stream_id)
    if labels:
        return labels
    if not allow_weak_annexb:
        return []
    return parse_annexb_frames(data, spans, stream_id)
