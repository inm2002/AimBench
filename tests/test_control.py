import math
import random
import unittest
from itertools import combinations, permutations
from unittest.mock import patch

from aimbench.components import NullInput
from aimbench.control.aim_controller import AimController
from aimbench.control.camera_model import CameraModel
from aimbench.control.freshness import MotionFeedbackTimeout, best_match
from aimbench.vision.base import Detection


class ControlTests(unittest.TestCase):
    def controller(self):
        device = NullInput()
        controller = AimController(CameraModel(1280, 720), device.move, device.click)
        return controller, device

    def test_matching_agrees_with_exhaustive_assignment(self):
        rng = random.Random(713)
        for n, m in [(1, 12), (2, 2), (2, 3), (2, 12), (12, 2), (3, 3), (4, 3)]:
            for _ in range(20):
                ref = [(rng.randrange(-20, 21), rng.randrange(-20, 21)) for _ in range(n)]
                cur = [(rng.randrange(-20, 21), rng.randrange(-20, 21)) for _ in range(m)]
                count = min(n, m)
                expected = min(
                    math.sqrt(sum(math.dist(a, b) ** 2 for a, b in zip(r, c)) / count)
                    for r in combinations(ref, count)
                    for subset in combinations(cur, count)
                    for c in permutations(subset)
                )
                self.assertAlmostEqual(best_match(ref, cur), expected, places=12)
        self.assertIsNone(best_match([], [(0, 0)]))
        self.assertAlmostEqual(best_match([(0, 0)] * 2, [(1, 0), (5, 0)]), math.sqrt(13))

    def test_center_hold_clicks_without_moving(self):
        controller, device = self.controller()
        for seq in (1, 2):
            controller.process([Detection(642, 360)], seq)
        self.assertEqual((device.moves, device.clicks), (0, 2))

    def test_single_target_waits_before_next_action(self):
        controller, device = self.controller()
        for seq in range(1, 5):
            controller.process([Detection(700, 360)], seq)
        self.assertEqual(device.moves, 1)
        controller.process([Detection(700, 360)], 5)
        self.assertEqual(device.moves, 2)

    def test_unchanged_large_move_waits_then_times_out(self):
        controller, device = self.controller()
        points = [Detection(700, 360), Detection(400, 260), Detection(900, 460)]
        with patch("time.perf_counter_ns", return_value=1_000_000_000) as clock:
            controller.process(points, 1)
            clock.return_value += 10_000_000
            controller.process(points, 5)
            self.assertEqual((device.moves, device.clicks), (1, 1))
            self.assertEqual(controller.stats.gate_pending_motion_frames, 1)
            clock.return_value += 20_000_000
            with self.assertRaises(MotionFeedbackTimeout):
                controller.process(points, 6)
        self.assertEqual((device.moves, device.clicks), (1, 1))

    def test_fresh_geometry_releases_on_next_frame(self):
        controller, device = self.controller()
        controller.process([Detection(700, 360), Detection(400, 260), Detection(900, 460)], 1)
        points = [Detection(x, y) for x, y in controller.predicted_landmarks]
        controller.process(points, 2)
        self.assertEqual(device.clicks, 2)
        self.assertEqual(controller.stats.gate_early_predictive_releases, 1)

    def test_duplicate_shot_does_not_become_landmark(self):
        controller, _ = self.controller()
        shot = Detection(700, 360)
        controller._build_predictive_landmarks(
            [shot, Detection(701, 361), Detection(400, 260)], shot, 10, 0
        )
        self.assertEqual(controller.old_landmarks, [(400, 260)])

    def test_expired_plan_never_sends_input(self):
        controller, device = self.controller()
        controller.frame_deadline_ns = 1
        self.assertEqual(controller.process([Detection(700, 360)], 1), "REJECT_STALE_PLAN")
        self.assertEqual((device.moves, device.clicks), (0, 0))

    def test_out_of_frame_target_never_sends_input(self):
        controller, device = self.controller()
        self.assertEqual(controller.process([Detection(-1, 360)], 1), "REJECT_OUT_OF_FRAME")
        self.assertEqual(device.clicks, 0)
