from __future__ import annotations

import argparse
from pathlib import Path

from iframe_detector.packets import discover_captures, normalize_packet_table, run_tshark_export, save_dataframe, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/rebuild_packets"))
    ap.add_argument("--tshark-path", default=r"F:\wireshark\tshark.exe")
    ap.add_argument("--packet-limit", type=int, default=None)
    ap.add_argument("--keep-payload", action="store_true")
    args = ap.parse_args()

    summaries = []
    for spec in discover_captures(args.data_root):
        raw_path = args.output_root / "raw_tshark" / f"{spec.capture_id}.tsv.gz"
        packet_path = args.output_root / "packets" / f"{spec.capture_id}.packets.csv.gz"
        run_tshark_export(spec.path, raw_path, args.tshark_path, args.packet_limit, args.keep_payload)
        df = normalize_packet_table(raw_path, spec.capture_id, spec.application)
        save_dataframe(df, packet_path)
        summaries.append(
            {
                "capture_id": spec.capture_id,
                "application": spec.application,
                "pcap_path": str(spec.path),
                "packet_path": str(packet_path),
                "packet_count": int(len(df)),
                "tcp_count": int(df["transport"].eq("tcp").sum()),
                "udp_count": int(df["transport"].eq("udp").sum()),
                "downlink_bytes": int(df.loc[df["is_downlink"].eq(True), "payload_len"].sum()),
            }
        )
        print(f"exported {spec.capture_id}: {len(df)} packets")
    write_json({"captures": summaries}, args.output_root / "packet_export_summary.json")


if __name__ == "__main__":
    main()

