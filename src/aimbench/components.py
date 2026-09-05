"""Lazy builtin adapters. Windows and vision libraries load only when selected."""

import math
from dataclasses import asdict, fields


class SendInputDevice:
    def __init__(self):
        from .control.input_backend import MouseInputBackend

        self.backend = MouseInputBackend()
        self.move = self.backend.move
        self.click = self.backend.click
        self.close = self.backend.close
        self.bind_guard = self.backend.bind_guard

    def metadata(self):
        return self.backend.metadata()


class NullInput:
    def __init__(self):
        self.moves = 0
        self.clicks = 0

    def move(self, dx, dy):
        self.moves += 1

    def click(self):
        self.clicks += 1

    def metadata(self):
        return {"physical_input": False, "api": "noop"}


class ControllerAdapter:
    """Validate settings and expose the shared controller to the runner."""

    def __init__(self, camera_model, move_fn, click_fn, config):
        from aimbench.control.aim_controller import AimController, AimControllerConfig

        config = AimControllerConfig(**config)
        for item in fields(config):
            value = getattr(config, item.name)
            if item.type in (bool, int) and type(value) is not item.type:
                raise ValueError(f"controller.{item.name} 类型应为 {item.type.__name__}")
            if item.type is float and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (not math.isfinite(value))
                or (value < 0)
            ):
                raise ValueError(f"controller.{item.name} 必须是有限非负数")
        if config.min_skip_frames < 0 or config.fallback_skip_frames < config.min_skip_frames:
            raise ValueError("跳帧数要求 0 <= min_skip_frames <= fallback_skip_frames")
        if config.hard_limit_margin_counts < 0 or not 0 <= config.predicted_vs_old_ratio <= 1:
            raise ValueError("hard_limit_margin_counts >= 0 且 predicted_vs_old_ratio 在 0～1 内")
        self.policy = AimController(camera_model, move_fn, click_fn, config)

    def process(self, detections, frame_seq):
        return self.policy.process(detections, frame_seq)

    def bind_gate(self, gate):
        self.policy.gate_policy = gate

    def set_frame_deadline(self, deadline_ns):
        self.policy.frame_deadline_ns = deadline_ns

    @property
    def stats(self):
        return self.policy.stats

    def final_stats(self):
        return asdict(self.stats)

    def metadata(self):
        return {
            "policy": "aimbench.control.aim_controller.AimController",
            "policy_version": self.policy.VERSION,
            "config": asdict(self.policy.config),
            "planner": "pixel_nearest",
            "hard_max_dx": self.policy.hard_max_dx,
            "hard_max_dy": self.policy.hard_max_dy,
        }
