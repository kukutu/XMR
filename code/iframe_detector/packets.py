from __future__ import annotations

import csv
import gzip
import ipaddress
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd


TSHARK_FIELDS = [
    "frame.number",
    "frame.time_epoch",
    "frame.len",
    "ip.src",
    "ip.dst",
    "tcp.stream",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.seq",
    "tcp.nxtseq",
    "tcp.len",
    "tcp.payload",
    "udp.srcport",
    "udp.dstport",
    "udp.length",
    "udp.payload",
    "_ws.col.Protocol",
]


@dataclass(frozen=True)
class CaptureSpec:
    capture_id: str
    application: str
    path: Path


def infer_application(name: str) -> str:
    low = name.lower()
    if "dy" in low or "douyin" in low:
        return "douyin"
    if "pdd" in low or "pinduoduo" in low:
        return "pinduoduo"
    if "tb" in low or "taobao" in low:
        return "taobao"
    if "xhs" in low or "xiaohongshu" in low:
        return "xhs"
    if "wechat" in low or "wx" in low:
        return "wechat"
    if "jd" in low:
        return "jd"
    if "mt" in low or "meituan" in low:
        return "meituan"
    return "unknown"


def discover_captures(data_root: Path) -> list[CaptureSpec]:
    specs: list[CaptureSpec] = []
    for path in sorted(data_root.glob("*.pcapng")):
        capture_id = path.stem
        specs.append(CaptureSpec(capture_id, infer_application(path.name), path))
    return specs


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return open(path, mode, newline="", encoding="utf-8")


def run_tshark_export(
    pcap_path: Path,
    output_path: Path,
    tshark_path: str = r"F:\wireshark\tshark.exe",
    packet_limit: int | None = None,
    keep_payload: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = TSHARK_FIELDS if keep_payload else [f for f in TSHARK_FIELDS if not f.endswith("payload")]
    cmd = [
        tshark_path,
        "-r",
        str(pcap_path),
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=\t",
        "-E",
        "occurrence=f",
    ]
    if packet_limit:
        cmd.extend(["-c", str(packet_limit)])
    for field in fields:
        cmd.extend(["-e", field])

    if output_path.suffix == ".gz":
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        with gzip.open(output_path, "wb") as out:
            shutil.copyfileobj(proc.stdout, out)
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"tshark failed for {pcap_path}: {stderr.decode(errors='replace')[-2000:]}")
    else:
        with open(output_path, "wb") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"tshark failed for {pcap_path}: {proc.stderr.decode(errors='replace')[-2000:]}")


def is_private_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_private
    except ValueError:
        return False


def normalize_packet_table(path: Path, capture_id: str, application: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    rename = {
        "frame.number": "packet_number",
        "frame.time_epoch": "time_epoch",
        "frame.len": "wire_len",
        "ip.src": "src_ip",
        "ip.dst": "dst_ip",
        "tcp.stream": "tcp_stream",
        "tcp.srcport": "tcp_srcport",
        "tcp.dstport": "tcp_dstport",
        "tcp.seq": "tcp_seq",
        "tcp.nxtseq": "tcp_nxtseq",
        "tcp.len": "tcp_len",
        "tcp.payload": "tcp_payload_hex",
        "udp.srcport": "udp_srcport",
        "udp.dstport": "udp_dstport",
        "udp.length": "udp_length",
        "udp.payload": "udp_payload_hex",
        "_ws.col.Protocol": "wireshark_protocol",
    }
    df = df.rename(columns=rename)
    for col in rename.values():
        if col not in df.columns:
            df[col] = ""
    numeric_int = ["packet_number", "wire_len", "tcp_len", "udp_length", "tcp_seq", "tcp_nxtseq"]
    for col in numeric_int:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    df["time_epoch"] = pd.to_numeric(df["time_epoch"], errors="coerce").fillna(0.0)
    df["capture_id"] = capture_id
    df["application"] = application
    df["transport"] = "other"
    df.loc[df["tcp_srcport"].ne("") | df["tcp_dstport"].ne(""), "transport"] = "tcp"
    df.loc[df["udp_srcport"].ne("") | df["udp_dstport"].ne(""), "transport"] = "udp"
    df["src_port"] = df["tcp_srcport"].where(df["tcp_srcport"].ne(""), df["udp_srcport"])
    df["dst_port"] = df["tcp_dstport"].where(df["tcp_dstport"].ne(""), df["udp_dstport"])
    df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).astype("int64")
    df["dst_port"] = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype("int64")
    df["payload_len"] = df["tcp_len"].where(df["transport"].eq("tcp"), df["udp_length"].clip(lower=0) - 8)
    df["payload_len"] = df["payload_len"].clip(lower=0).astype("int64")
    df["src_private"] = df["src_ip"].map(is_private_ip)
    df["dst_private"] = df["dst_ip"].map(is_private_ip)
    df["is_uplink"] = df["src_private"] & ~df["dst_private"]
    df["is_downlink"] = ~df["src_private"] & df["dst_private"]
    df["flow_key"] = df.apply(flow_key_from_row, axis=1)
    return df


def flow_key_from_row(row: pd.Series) -> str:
    transport = row.get("transport", "other")
    a = (str(row.get("src_ip", "")), int(row.get("src_port", 0)))
    b = (str(row.get("dst_ip", "")), int(row.get("dst_port", 0)))
    left, right = sorted([a, b])
    return f"{transport}:{left[0]}:{left[1]}-{right[0]}:{right[1]}"


def payload_bytes(row: pd.Series) -> bytes:
    hex_value = ""
    if row.get("transport") == "tcp":
        hex_value = row.get("tcp_payload_hex", "")
    elif row.get("transport") == "udp":
        hex_value = row.get("udp_payload_hex", "")
    if not isinstance(hex_value, str) or not hex_value:
        return b""
    hex_value = hex_value.replace(":", "").strip()
    try:
        return bytes.fromhex(hex_value)
    except ValueError:
        return b""


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def write_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))
