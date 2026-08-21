import unittest

import pandas as pd

from iframe_detector.splits import canonicalize_capture_columns, load_split_config, split_masks


class SplitRecoveryTests(unittest.TestCase):
    def test_aliases_map_to_recovered_roles(self):
        config = load_split_config()
        df = pd.DataFrame(
            {
                "capture_id": ["livestreaming_dy", "livstreaming_xhs_test", "not_in_split"],
                "application": ["douyin", "xhs", "unknown"],
                "value": [1, 2, 3],
            }
        )
        mapped = canonicalize_capture_columns(df, config, drop_unassigned=False)
        self.assertEqual(mapped.loc[0, "capture_id"], "douyin_primary")
        self.assertEqual(mapped.loc[1, "capture_id"], "xiaohongshu_secondary")
        self.assertEqual(mapped.loc[1, "application"], "xiaohongshu")
        self.assertEqual(mapped.loc[2, "split_role"], "unassigned")

    def test_split_masks_use_development_and_final_ood(self):
        config = load_split_config()
        df = pd.DataFrame(
            {
                "capture_id": ["livstreaming_wechat", "livstreaming_xhs"],
                "application": ["wechat", "xhs"],
                "value": [1, 2],
            }
        )
        mapped = canonicalize_capture_columns(df, config, drop_unassigned=True)
        train, test = split_masks(mapped)
        self.assertEqual(train.tolist(), [True, False])
        self.assertEqual(test.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
