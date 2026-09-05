"""Predictive freshness with bounded protection against repeated large movements."""

import math
import time
from dataclasses import dataclass
from itertools import combinations, permutations


def best_match(reference, current):
    """Minimum RMS distance under a distinct-point assignment."""
    if not reference or not current:
        return None
    count = min(len(reference), len(current))
    if count == 1:
        return math.sqrt(
            min((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for a in reference for b in current)
        )
    if count == 2:
        if len(reference) > len(current):
            reference, current = current, reference
        first = second = other_first = other_second = math.inf
        first_index = other_index = -1
        a, b = reference
        for index, point in enumerate(current):
            da = (a[0] - point[0]) ** 2 + (a[1] - point[1]) ** 2
            db = (b[0] - point[0]) ** 2 + (b[1] - point[1]) ** 2
            if da < first:
                second, first, first_index = first, da, index
            elif da < second:
                second = da
            if db < other_first:
                other_second, other_first, other_index = other_first, db, index
            elif db < other_second:
                other_second = db
        # 两点分配：最优两条边撞到同一个点时，退而取代价最小的次优组合
        total = (
            first + other_first
            if first_index != other_index
            else min(first + other_second, second + other_first)
        )
        return math.sqrt(total / 2)
    return min(
        math.sqrt(sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for a, b in zip(ref, cur)) / count)
        for ref in combinations(reference, count)
        for subset in combinations(current, count)
        for cur in permutations(subset)
    )


def unique_points(points, radius):
    accepted = []
    for point in points:
        if not any(
            math.hypot(point[0] - other[0], point[1] - other[1]) < radius for other in accepted
        ):
            accepted.append(point)
    return accepted


class MotionFeedbackTimeout(RuntimeError):
    """The previous large movement has not appeared before the feedback deadline."""


@dataclass(frozen=True)
class Match:
    old_error: float | None
    predicted_error: float | None
    passed: bool


class PredictiveFreshnessGate:
    def __init__(
        self,
        landmark_separation_px=16.0,
        shot_exclusion_radius_px=16.0,
        max_landmarks=2,
        max_match_candidates=12,
        pending_timeout_ms=25.0,
    ):
        for name, value in {
            "landmark_separation_px": landmark_separation_px,
            "shot_exclusion_radius_px": shot_exclusion_radius_px,
        }.items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(max_landmarks) is not int or not 1 <= max_landmarks <= 3:
            raise ValueError("max_landmarks must be between 1 and 3")
        if type(max_match_candidates) is not int or not max_landmarks <= max_match_candidates <= 16:
            raise ValueError("max_match_candidates must be between max_landmarks and 16")
        if (
            isinstance(pending_timeout_ms, bool)
            or not math.isfinite(pending_timeout_ms)
            or not 0 < pending_timeout_ms <= 100
        ):
            raise ValueError("pending_timeout_ms must be in (0, 100]")
        self.landmark_separation_px = float(landmark_separation_px)
        self.shot_exclusion_radius_px = float(shot_exclusion_radius_px)
        self.max_landmarks = max_landmarks
        self.max_match_candidates = max_match_candidates
        self.pending_timeout_ms = float(pending_timeout_ms)

    def metadata(self):
        return {
            "policy": "predictive",
            "projection": "independent",
            "pending_motion_guard": True,
            "pending_timeout_ms": self.pending_timeout_ms,
            "pending_max_frames": 12,
            "landmark_separation_px": self.landmark_separation_px,
        }

    def build_landmarks(self, ctx, detections, fired_target, mouse_dx, mouse_dy):
        old, predicted = [], []
        for detection in detections:
            if detection is fired_target or len(old) >= self.max_landmarks:
                continue
            point = (float(detection.x), float(detection.y))
            if (
                math.hypot(point[0] - fired_target.x, point[1] - fired_target.y)
                < self.shot_exclusion_radius_px
            ):
                continue
            if any(
                math.hypot(point[0] - other[0], point[1] - other[1]) < self.landmark_separation_px
                for other in old
            ):
                continue
            new = ctx._predict_point_after_move(*point, mouse_dx, mouse_dy)
            if ctx._point_inside_frame(new):
                old.append(point)
                predicted.append(new)
        ctx.old_landmarks, ctx.predicted_landmarks = old, predicted
        ctx.expected_landmark_shift_px = (
            sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(old, predicted)) / len(old)
            if old
            else 0.0
        )

    def _evaluate(self, ctx, detections):
        points = unique_points(
            [(float(d.x), float(d.y)) for d in detections], self.landmark_separation_px
        )
        if len(points) > self.max_match_candidates:
            anchors = ctx.old_landmarks + ctx.predicted_landmarks
            points = sorted(
                points, key=lambda p: min((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2 for a in anchors)
            )[: self.max_match_candidates]
        if not len(points) >= len(ctx.old_landmarks) > 0:
            return Match(None, None, False)
        old = best_match(ctx.old_landmarks, points)
        predicted = best_match(ctx.predicted_landmarks, points)
        cfg = ctx.config
        passed = (
            predicted <= cfg.max_predicted_match_px
            and old - predicted >= cfg.min_predicted_advantage_px
            and predicted <= old * cfg.predicted_vs_old_ratio
        )
        return Match(old, predicted, passed)

    @staticmethod
    def _release(ctx):
        ctx.gate_mode = "NONE"
        ctx.gate_wait_count = 0
        ctx.old_landmarks = []
        ctx.predicted_landmarks = []
        ctx.expected_landmark_shift_px = 0.0
        return True

    def allows_action(self, ctx, detections, frame_seq):
        if ctx.gate_mode == "NONE":
            return True
        anchor = ctx.gate_action_frame_seq
        ctx.gate_wait_count = (
            max(ctx.gate_wait_count + 1, frame_seq - anchor)
            if anchor is not None
            else ctx.gate_wait_count + 1
        )
        ctx.stats.gated_frames += 1
        cfg = ctx.config
        if ctx.gate_mode == "FIXED_SINGLE":
            if ctx.gate_wait_count > cfg.fallback_skip_frames:
                ctx.stats.gate_single_target_releases += 1
                return self._release(ctx)
            ctx.stats.gate_min_wait_frames += 1
            return False

        eligible = (
            bool(ctx.old_landmarks and ctx.predicted_landmarks)
            and ctx.expected_landmark_shift_px > cfg.small_expected_shift_px
        )
        match = self._evaluate(ctx, detections) if ctx.gate_wait_count == 1 and eligible else None
        if ctx.gate_wait_count == 1 and cfg.early_predict_n1 and match and match.passed:
            ctx.stats.gate_predictive_releases += 1
            ctx.stats.gate_early_predictive_releases += 1
            return self._release(ctx)
        if ctx.gate_wait_count <= cfg.min_skip_frames:
            ctx.stats.gate_min_wait_frames += 1
            return False

        if ctx.old_landmarks and ctx.predicted_landmarks:
            if ctx.expected_landmark_shift_px <= cfg.small_expected_shift_px:
                ctx.stats.gate_small_shift_releases += 1
                return self._release(ctx)
            match = match or self._evaluate(ctx, detections)
            # 画面仍贴合移动前的旧几何，预测位置又对不上 → 大位移尚未生效，限时等待
            pending = (
                ctx.gate_wait_count > cfg.fallback_skip_frames
                and len(ctx.old_landmarks) >= 2
                and ctx.expected_landmark_shift_px >= 32
                and match.old_error is not None
                and match.old_error <= 2
                and match.predicted_error > cfg.max_predicted_match_px
                and not match.passed
            )
            if pending:
                if not ctx.pending_motion_wait_frames:
                    ctx.stats.gate_pending_motion_actions += 1
                ctx.pending_motion_wait_frames += 1
                ctx.stats.gate_pending_motion_frames += 1
                ctx.stats.gate_inconclusive_frames += 1
                elapsed = (time.perf_counter_ns() - ctx.gate_action_click_ns) / 1e6
                if elapsed >= self.pending_timeout_ms or ctx.gate_wait_count >= 12:
                    ctx.stats.gate_pending_motion_timeouts += 1
                    raise MotionFeedbackTimeout("Large movement still has old geometry")
                return False
            if match.passed:
                ctx.stats.gate_predictive_releases += 1
                return self._release(ctx)
        ctx.stats.gate_inconclusive_frames += 1
        if ctx.gate_wait_count > cfg.fallback_skip_frames:
            ctx.stats.gate_fallback_releases += 1
            return self._release(ctx)
        return False
