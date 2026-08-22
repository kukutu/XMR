import unittest

import pandas as pd

from iframe_detector.video import ByteSpan, parse_annexb_frames, parse_annexb_frames_from_flow, parse_flv_frames


class VideoParserTests(unittest.TestCase):
    def test_parse_flv_keyframe_tag(self):
        header = b"FLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00"
        payload = b"\x17\x01\x00\x00\x00abc"
        tag = b"\x09" + len(payload).to_bytes(3, "big") + b"\x00\x00\x01\x00" + b"\x00\x00\x00" + payload
        prev_size = (11 + len(payload)).to_bytes(4, "big")
        data = header + tag + prev_size
        spans = [ByteSpan(packet_number=10, start=0, end=len(data))]
        frames = parse_flv_frames(data, spans, "s")
        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0].is_keyframe)
        self.assertEqual(frames[0].packet_numbers, [10])

    def test_parse_annexb_idr(self):
        data = b"\x00\x00\x00\x01\x65abc\x00\x00\x01\x41def"
        spans = [ByteSpan(packet_number=1, start=0, end=8), ByteSpan(packet_number=2, start=8, end=len(data))]
        frames = parse_annexb_frames(data, spans, "s")
        self.assertEqual(len(frames), 2)
        self.assertTrue(frames[0].is_keyframe)
        self.assertFalse(frames[1].is_keyframe)

    def test_parse_annexb_streaming_across_packets(self):
        packets = pd.DataFrame(
            [
                {
                    "transport": "tcp",
                    "is_downlink": True,
                    "tcp_len": 5,
                    "tcp_seq": 1,
                    "packet_number": 1,
                    "tcp_payload_hex": "00:00:00:01:65",
                },
                {
                    "transport": "tcp",
                    "is_downlink": True,
                    "tcp_len": 6,
                    "tcp_seq": 6,
                    "packet_number": 2,
                    "tcp_payload_hex": "61:62:63:00:00:01",
                },
                {
                    "transport": "tcp",
                    "is_downlink": True,
                    "tcp_len": 4,
                    "tcp_seq": 12,
                    "packet_number": 3,
                    "tcp_payload_hex": "41:64:65:66",
                },
            ]
        )
        frames = parse_annexb_frames_from_flow(packets, "s")
        self.assertEqual(len(frames), 2)
        self.assertTrue(frames[0].is_keyframe)
        self.assertFalse(frames[1].is_keyframe)
        self.assertEqual(frames[0].packet_numbers, [1, 2])
        self.assertEqual(frames[1].packet_numbers, [2, 3])


if __name__ == "__main__":
    unittest.main()
