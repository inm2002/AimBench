import unittest

import numpy as np

from aimbench.game_observer import TimerFilter
from aimbench.vision.heatmap_decode import decode_heatmap_cpu


class VisionTests(unittest.TestCase):
    def test_gaussian_centers_keep_fractional_precision(self):
        yy, xx = np.mgrid[:48, :80]
        centers = [(12.25, 10.3), (40.2, 24.6), (63.75, 35.1)]
        probability = np.maximum.reduce(
            [np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5**2)) for x, y in centers]
        )
        probability = np.clip(probability, 1e-12, 1 - 1e-12)
        logits = (np.log(probability) - np.log1p(-probability)).astype(np.float32)
        points = sorted((x * 79, y * 47) for x, y, _ in decode_heatmap_cpu(logits))
        np.testing.assert_allclose(points, centers, atol=1e-5, rtol=0)

    def test_flat_peak_is_one_target(self):
        logits = np.full((48, 80), -20, np.float32)
        logits[20:24, 30:34] = 20
        self.assertEqual(len(decode_heatmap_cpu(logits)), 1)

    def test_three_saturated_peaks_remain_distinct(self):
        logits = np.full((48, 80), -20, np.float32)
        logits[10, 10] = logits[24, 40] = logits[35, 65] = 20
        self.assertEqual(len(decode_heatmap_cpu(logits)), 3)

    def test_empty_heatmap_has_no_padding_targets(self):
        self.assertEqual(decode_heatmap_cpu(np.full((48, 80), -20, np.float32)), [])

    def test_invalid_heatmap_rejected(self):
        logits = np.zeros((48, 80), np.float32)
        logits[1, 1] = np.nan
        with self.assertRaises(ValueError):
            decode_heatmap_cpu(logits)

    def test_timer_bridges_only_brief_recognition_damage(self):
        timer = TimerFilter(75)
        valid = {"best_value": 53, "value": 53, "glyph_count": 4, "glyphs": []}
        self.assertEqual(timer.update(valid, 1_000_000_000, True), (53, "direct"))
        damaged = {**valid, "best_value": None, "value": None}
        self.assertEqual(timer.update(damaged, 1_010_000_000, True), (53, "held"))
        self.assertIsNone(timer.update(damaged, 1_076_000_000, True)[0])
        self.assertIsNone(timer.update({**damaged, "glyph_count": 0}, 1_010_000_000, True)[0])
        self.assertIsNone(timer.update(damaged, 1_010_000_000, False)[0])
