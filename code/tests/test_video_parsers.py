import unittest

from iframe_detector.video import ByteSpan, parse_annexb_frames, parse_flv_frames


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


if __name__ == "__main__":
    unittest.main()

