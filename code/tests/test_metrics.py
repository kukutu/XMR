import unittest

from iframe_detector.metrics import match_packet_sets, packet_iou


class MetricsTests(unittest.TestCase):
    def test_packet_iou(self):
        self.assertAlmostEqual(packet_iou([1, 2, 3], [2, 3, 4]), 0.5)

    def test_match_packet_sets(self):
        pred = [("p1", [1, 2, 3]), ("p2", [10])]
        truth = [("t1", [1, 2, 3]), ("t2", [20])]
        m = match_packet_sets(pred, truth, minimum_iou=0.9)
        self.assertEqual(m["true_positive_count"], 1)
        self.assertEqual(m["false_positive_count"], 1)
        self.assertEqual(m["false_negative_count"], 1)


if __name__ == "__main__":
    unittest.main()

