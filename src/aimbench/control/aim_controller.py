import math
import time
from dataclasses import dataclass


@dataclass
class AimControllerConfig:
    """Target selection and post-move freshness thresholds."""

    min_skip_frames: int = 1
    fallback_skip_frames: int = 3
    small_expected_shift_px: float = 10.0
    max_predicted_match_px: float = 18.0
    min_predicted_advantage_px: float = 10.0
    predicted_vs_old_ratio: float = 0.6
    zero_move_fast: bool = True
    center_deadzone_px: float = 6.0
    early_predict_n1: bool = True
    hard_limit_margin_counts: int = 20


@dataclass
class AimControllerStats:
    actions: int = 0
    mouse_moves: int = 0
    fire_attempts: int = 0
    zero_move_shots: int = 0
    single_target_actions: int = 0
    multi_target_actions: int = 0
    gated_frames: int = 0
    gate_min_wait_frames: int = 0
    gate_predictive_releases: int = 0
    gate_early_predictive_releases: int = 0
    gate_small_shift_releases: int = 0
    gate_fallback_releases: int = 0
    gate_single_target_releases: int = 0
    gate_inconclusive_frames: int = 0
    gate_pending_motion_actions: int = 0
    gate_pending_motion_frames: int = 0
    gate_pending_motion_timeouts: int = 0
    gate_zero_fast_bypasses: int = 0
    rejected_out_of_frame: int = 0
    rejected_hard_limit: int = 0


class AimController:
    """Select the nearest target, submit input, then wait for fresh geometry."""

    VERSION = "1.0"
    GATE_NONE = "NONE"
    GATE_PREDICTIVE = "PREDICTIVE"
    GATE_FIXED_SINGLE = "FIXED_SINGLE"

    def __init__(self, camera_model, move_fn, click_fn, config=None):
        from .freshness import PredictiveFreshnessGate

        self.gate_policy = PredictiveFreshnessGate()
        self.camera_model = camera_model
        self.gate_action_frame_seq = None
        self.gate_action_click_ns = None
        self.pending_motion_wait_frames = 0
        self.frame_deadline_ns = None
        self.move_fn = move_fn
        self.click_fn = click_fn
        self.config = config if config is not None else AimControllerConfig()
        self.stats = AimControllerStats()
        self.gate_mode = self.GATE_NONE
        self.gate_wait_count = 0
        self.old_landmarks = []
        self.predicted_landmarks = []
        self.expected_landmark_shift_px = 0.0
        # 由屏幕四条边推算单次位移硬上限，再加余量，异常大位移直接拒绝
        left_dx, _ = self.camera_model.screen_to_mouse(0.0, self.camera_model.cy)
        right_dx, _ = self.camera_model.screen_to_mouse(self.width - 1.0, self.camera_model.cy)
        _, top_dy = self.camera_model.screen_to_mouse(self.camera_model.cx, 0.0)
        _, bottom_dy = self.camera_model.screen_to_mouse(self.camera_model.cx, self.height - 1.0)
        self.hard_max_dx = (
            max(abs(int(left_dx)), abs(int(right_dx))) + self.config.hard_limit_margin_counts
        )
        self.hard_max_dy = (
            max(abs(int(top_dy)), abs(int(bottom_dy))) + self.config.hard_limit_margin_counts
        )

    @property
    def width(self):
        return self.camera_model.cx * 2.0

    @property
    def height(self):
        return self.camera_model.cy * 2.0

    def _select_nearest_target(self, detections):
        if not detections:
            return None
        cx = self.camera_model.cx
        cy = self.camera_model.cy
        return min(
            detections, key=lambda detection: (detection.x - cx) ** 2 + (detection.y - cy) ** 2
        )

    def _target_inside_frame(self, target):
        return 0.0 <= target.x < self.width and 0.0 <= target.y < self.height

    def _point_inside_frame(self, point):
        x, y = point
        return 0.0 <= x < self.width and 0.0 <= y < self.height

    def _predict_point_after_move(self, x, y, mouse_dx, mouse_dy):
        return self.camera_model.project_independent(x, y, mouse_dx, mouse_dy)

    def _record_fire(self):
        self.stats.fire_attempts += 1

    def _gate_allows_action(self, detections, frame_seq):
        return self.gate_policy.allows_action(self, detections, frame_seq)

    def _build_predictive_landmarks(self, detections, fired_target, mouse_dx, mouse_dy):
        return self.gate_policy.build_landmarks(self, detections, fired_target, mouse_dx, mouse_dy)

    def process(self, detections, frame_seq):
        """
        Called exactly once per NEW WGC frame.
        """
        if not self._gate_allows_action(detections, frame_seq):
            if self.gate_mode == self.GATE_FIXED_SINGLE:
                return "SINGLE_FIXED_GATE_WAIT"
            return "PREDICTIVE_GATE_WAIT"
        if not detections:
            return "NO_TARGET"
        detection_count = len(detections)
        target = self._select_nearest_target(detections)
        if target is None:
            return "NO_TARGET"
        if not self._target_inside_frame(target):
            self.stats.rejected_out_of_frame += 1
            return "REJECT_OUT_OF_FRAME"
        target_distance = math.hypot(
            target.x - self.camera_model.cx, target.y - self.camera_model.cy
        )
        raw_dx, raw_dy = self.camera_model.screen_to_mouse(target.x, target.y)
        raw_dx = int(raw_dx)
        raw_dy = int(raw_dy)
        center_hold = target_distance <= self.config.center_deadzone_px
        if center_hold:
            raw_dx = raw_dy = 0
        if abs(raw_dx) > self.hard_max_dx or abs(raw_dy) > self.hard_max_dy:
            self.stats.rejected_hard_limit += 1
            return "REJECT_HARD_LIMIT"
        is_zero_move = raw_dx == 0 and raw_dy == 0
        # 没有移动就没有画面变化要等，跳过门控直接开火
        if self.config.zero_move_fast and is_zero_move:
            if detection_count >= 2:
                self.stats.multi_target_actions += 1
            else:
                self.stats.single_target_actions += 1
            self.old_landmarks = []
            self.predicted_landmarks = []
            self.expected_landmark_shift_px = 0.0
            next_gate_mode = self.GATE_NONE
        elif detection_count >= 2:
            self.stats.multi_target_actions += 1
            self._build_predictive_landmarks(
                detections=detections, fired_target=target, mouse_dx=raw_dx, mouse_dy=raw_dy
            )
            if self.old_landmarks and self.predicted_landmarks:
                next_gate_mode = self.GATE_PREDICTIVE
            else:
                next_gate_mode = self.GATE_FIXED_SINGLE
        else:
            self.stats.single_target_actions += 1
            self.old_landmarks = []
            self.predicted_landmarks = []
            self.expected_landmark_shift_px = 0.0
            next_gate_mode = self.GATE_FIXED_SINGLE
        # 输入尚未发出，deadline 已过则整个计划作废
        if self.frame_deadline_ns is not None and time.perf_counter_ns() >= self.frame_deadline_ns:
            return "REJECT_STALE_PLAN"
        if not is_zero_move:
            self.move_fn(raw_dx, raw_dy)
        self.click_fn()
        click_end_ns = time.perf_counter_ns()
        if not is_zero_move:
            self.stats.mouse_moves += 1
        else:
            self.stats.zero_move_shots += 1
        self._record_fire()
        self.stats.actions += 1
        self.gate_action_frame_seq = frame_seq
        self.gate_action_click_ns = click_end_ns
        self.pending_motion_wait_frames = 0
        self.gate_mode = next_gate_mode
        self.gate_wait_count = 0
        if detection_count == 1:
            if is_zero_move:
                action_name = (
                    "ONE_SHOT_SINGLE_ZERO_FAST"
                    if self.config.zero_move_fast
                    else "ONE_SHOT_SINGLE_ZERO_MOVE"
                )
            else:
                action_name = "ONE_SHOT_SINGLE"
        elif is_zero_move:
            action_name = (
                "ONE_SHOT_ZERO_FAST" if self.config.zero_move_fast else "ONE_SHOT_ZERO_MOVE"
            )
        else:
            action_name = "ONE_SHOT"
        if self.config.zero_move_fast and is_zero_move:
            self.stats.gate_zero_fast_bypasses += 1
        return action_name
