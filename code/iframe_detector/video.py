from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


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
    chunks: list[bytes] = []
    spans: list[ByteSpan] = []
    for packet_number, start, end, hex_value in iter_tcp_direction_payload_hex(flow_df, downlink=downlink):
        try:
            data = bytes.fromhex(hex_value)
        except ValueError:
            continue
        if not data:
            continue
        chunks.append(data)
        spans.append(ByteSpan(packet_number, start, end))
    return b"".join(chunks), spans


def iter_tcp_direction_payload_hex(flow_df: pd.DataFrame, downlink: bool = True):
    rows = flow_df[flow_df["transport"].eq("tcp")]
    rows = rows[rows["is_downlink"].eq(bool(downlink))]
    rows = rows[rows["tcp_len"].fillna(0).astype(int) > 0]
    if rows.empty:
        return
    rows = rows.sort_values(["tcp_seq", "packet_number"])
    stream_pos = 0
    seen: set[tuple[int, int]] = set()
    for row in rows.itertuples(index=False):
        hex_value = getattr(row, "tcp_payload_hex", "")
        if not isinstance(hex_value, str) or not hex_value:
            continue
        hex_value = hex_value.replace(":", "").strip()
        if not hex_value:
            continue
        data_len = len(hex_value) // 2
        if data_len <= 0:
            continue
        seq = int(getattr(row, "tcp_seq"))
        key = (seq, data_len)
        if key in seen:
            continue
        seen.add(key)
        start = stream_pos
        stream_pos += data_len
        yield int(getattr(row, "packet_number")), start, stream_pos, hex_value


def packets_for_span(spans: list[ByteSpan], start: int, end: int) -> list[int]:
    packets = [s.packet_number for s in spans if s.start < end and s.end > start]
    return sorted(set(packets))


def span_packet_lookup(spans: list[ByteSpan]):
    ends = [s.end for s in spans]

    def lookup(start: int, end: int) -> list[int]:
        packets: list[int] = []
        idx = bisect_right(ends, start)
        for span in spans[idx:]:
            if span.start >= end:
                break
            packets.append(span.packet_number)
        return sorted(set(packets))

    return lookup


def parse_flv_frames(data: bytes, spans: list[ByteSpan], stream_id: str) -> list[VideoFrameLabel]:
    frames: list[VideoFrameLabel] = []
    header = data.find(b"FLV")
    if header < 0 or header + 13 > len(data):
        return frames
    lookup_packets = span_packet_lookup(spans)
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
            packets = lookup_packets(tag_start, next_pos)
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
        pos3 = data.find(b"\x00\x00\x01", i)
        pos4 = data.find(b"\x00\x00\x00\x01", i)
        if pos3 < 0 and pos4 < 0:
            break
        if pos4 >= 0 and (pos3 < 0 or pos4 <= pos3):
            starts.append((pos4, pos4 + 4))
            i = pos4 + 4
        else:
            starts.append((pos3, pos3 + 3))
            i = pos3 + 3
    return starts


def _annexb_start_codes_hex(hex_value: str) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    i = 0
    while i + 6 <= len(hex_value):
        pos3 = hex_value.find("000001", i)
        pos4 = hex_value.find("00000001", i)
        candidates = [p for p in (pos3, pos4) if p >= 0]
        if not candidates:
            break
        pos = min(candidates)
        if pos % 2:
            i = pos + 1
            continue
        if pos4 == pos:
            starts.append((pos // 2, pos // 2 + 4))
            i = pos + 8
        else:
            starts.append((pos // 2, pos // 2 + 3))
            i = pos + 6
    return starts


def _classify_annexb_header(h0: int) -> tuple[str, bool] | None:
    avc_type = h0 & 0x1F
    hevc_type = (h0 >> 1) & 0x3F
    codec = "avc" if avc_type in range(1, 32) else "unknown"
    is_key = avc_type == 5
    if hevc_type in {16, 17, 18, 19, 20, 21, 32, 33, 34}:
        codec = "hevc"
        is_key = hevc_type in {16, 17, 18, 19, 20, 21}
    if avc_type not in {1, 5} and not is_key:
        return None
    return codec, is_key


def parse_annexb_frames(data: bytes, spans: list[ByteSpan], stream_id: str) -> list[VideoFrameLabel]:
    starts = _annexb_start_codes(data)
    if not starts:
        return []
    frames: list[VideoFrameLabel] = []
    lookup_packets = span_packet_lookup(spans)
    for idx, (sc_start, payload_start) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(data)
        if payload_start >= end:
            continue
        classified = _classify_annexb_header(data[payload_start])
        if classified is None:
            continue
        codec, is_key = classified
        packets = lookup_packets(sc_start, end)
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


def parse_annexb_frames_from_flow(flow_df: pd.DataFrame, stream_id: str) -> list[VideoFrameLabel]:
    frames: list[VideoFrameLabel] = []
    tail_hex = ""
    tail_packets: list[int] = []
    tail_base = 0
    last_start_pos = -1
    active_start: int | None = None
    active_payload_start: int | None = None
    active_header: int | None = None
    active_packets: set[int] = set()
    stream_end = 0

    def header_from_buffer(buffer_hex: str, base: int, payload_start: int) -> int | None:
        idx = (payload_start - base) * 2
        if 0 <= idx and idx + 2 <= len(buffer_hex):
            try:
                return int(buffer_hex[idx : idx + 2], 16)
            except ValueError:
                return None
        return None

    def finish_frame(end_pos: int) -> None:
        nonlocal active_start, active_payload_start, active_header
        if active_start is None or active_payload_start is None or active_payload_start >= end_pos:
            return
        if active_header is None:
            return
        classified = _classify_annexb_header(active_header)
        if classified is None or not active_packets:
            return
        codec, is_key = classified
        frames.append(
            VideoFrameLabel(
                stream_id=stream_id,
                codec=codec,
                frame_index=len(frames),
                timestamp_ms=None,
                byte_start=active_start,
                byte_end=end_pos,
                packet_numbers=sorted(active_packets),
                is_keyframe=is_key,
                label_source="annexb",
            )
        )

    for packet_number, start, end, hex_value in iter_tcp_direction_payload_hex(flow_df, downlink=True):
        stream_end = end
        tail_len = len(tail_hex) // 2
        buffer_hex = tail_hex + hex_value
        base = start - tail_len
        starts = []
        for offset, payload_offset in _annexb_start_codes_hex(buffer_hex):
            sc_start = base + offset
            payload_start = base + payload_offset
            if sc_start <= last_start_pos:
                continue
            starts.append((sc_start, payload_start, offset, payload_offset))

        if not starts:
            if active_start is not None:
                active_packets.add(packet_number)
                if active_header is None and active_payload_start is not None and start <= active_payload_start < end:
                    active_header = header_from_buffer(hex_value, start, active_payload_start)
        else:
            for sc_start, payload_start, offset, payload_offset in starts:
                if active_start is not None and sc_start > start:
                    active_packets.add(packet_number)
                finish_frame(sc_start)
                new_packets: set[int] = set()
                for byte_index in range(offset, min(payload_offset, len(buffer_hex) // 2)):
                    global_pos = base + byte_index
                    if global_pos < start:
                        tail_index = global_pos - tail_base
                        if 0 <= tail_index < len(tail_packets):
                            new_packets.add(tail_packets[tail_index])
                    else:
                        new_packets.add(packet_number)
                if payload_start < end:
                    new_packets.add(packet_number)
                active_start = sc_start
                active_payload_start = payload_start
                active_header = header_from_buffer(buffer_hex, base, payload_start)
                active_packets = new_packets
                last_start_pos = sc_start
            if active_start is not None and starts[-1][1] < end:
                active_packets.add(packet_number)
                if active_header is None and active_payload_start is not None and start <= active_payload_start < end:
                    active_header = header_from_buffer(hex_value, start, active_payload_start)

        keep_bytes = min(3, len(buffer_hex) // 2)
        if keep_bytes:
            tail_hex = buffer_hex[-keep_bytes * 2 :]
            if keep_bytes <= (end - start):
                tail_packets = [packet_number] * keep_bytes
                tail_base = end - keep_bytes
            else:
                take_old = keep_bytes - (end - start)
                tail_packets = tail_packets[-take_old:] + [packet_number] * (end - start)
                tail_base = end - keep_bytes
        else:
            tail_hex = ""
            tail_packets = []
            tail_base = end

    finish_frame(stream_end)
    return frames


def flow_contains_flv_header(flow_df: pd.DataFrame) -> bool:
    tail_hex = ""
    for _, _, _, hex_value in iter_tcp_direction_payload_hex(flow_df, downlink=True):
        buffer_hex = tail_hex + hex_value
        if buffer_hex.find("464c56") >= 0:
            return True
        tail_hex = buffer_hex[-4:]
    return False


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
    has_flv = flow_contains_flv_header(flow_df)
    if allow_weak_annexb and not has_flv:
        labels = parse_annexb_frames_from_flow(flow_df, stream_id)
        if labels:
            return labels
    if not has_flv:
        return []
    data, spans = reassemble_tcp_direction(flow_df, downlink=True)
    if not data:
        return []
    labels = parse_flv_frames(data, spans, stream_id)
    if labels:
        return labels
    if not allow_weak_annexb:
        return []
    return parse_annexb_frames(data, spans, stream_id)
