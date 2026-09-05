"""Application configuration and paths, without loading GPU or Windows libraries."""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"


def project_path(value):
    return Path(value).expanduser().resolve()


@dataclass
class RunOptions:
    window_keyword: str = "aimlab"
    output_dir: str = "runs"
    start_delay: float = 5.0
    countdown_wait: float = 3.0
    startup_timeout_seconds: float = 10.0
    auto_start_click: bool = True
    max_run_seconds: float = 60.3
    normal_end_min_seconds: float = 59.0
    capture_timeout_seconds: float = 2.0
    max_result_age_ms: float = 20.0
    max_publish_age_ms: float = 20.0
    zero_target_grace_ms: float = 150.0
    scene_unknown_grace_ms: float = 1000.0
    timer_hold_ms: float = 75.0
    gc_policy: str = "defer"
    max_memory_growth_mb: float = 512.0
    health_interval_seconds: float = 1.0
    save_result_image: bool = True
    prompt_result: bool = True
    result_screen_delay_seconds: float = 2.0
    result_screen_timeout_seconds: float = 8.0
    result_screen_stable_seconds: float = 0.8

    def validate(self):
        for name, value in asdict(self).items():
            if name in {"auto_start_click", "save_result_image", "prompt_result"}:
                if type(value) is not bool:
                    raise ValueError(f"run.{name} must be a boolean")
            elif name not in {"window_keyword", "output_dir", "gc_policy"}:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"run.{name} must be a finite nonnegative number")
        for name in (
            "max_run_seconds",
            "capture_timeout_seconds",
            "startup_timeout_seconds",
            "health_interval_seconds",
            "max_memory_growth_mb",
            "result_screen_timeout_seconds",
            "result_screen_stable_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"run.{name} must be positive")
        if self.timer_hold_ms > 250:
            raise ValueError("run.timer_hold_ms must not exceed 250")
        if self.gc_policy not in {"defer", "observe"}:
            raise ValueError("run.gc_policy must be defer or observe")
        for name in ("window_keyword", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"run.{name} must be a nonempty string")


@dataclass
class AppConfig:
    algorithm: str = "cnn"
    detector_params: dict = field(default_factory=dict)
    controller_params: dict = field(default_factory=dict)
    gate_params: dict = field(default_factory=dict)
    capture_params: dict = field(default_factory=dict)
    calibration: dict = field(
        default_factory=lambda: {
            "fx": 512.056,
            "fy": 456.726,
            "yaw_per_count": 0.00219291,
            "pitch_per_count": 0.00244552,
        }
    )
    conditions: dict = field(
        default_factory=lambda: {
            "scenario": "Gridshot",
            "sensitivity": 1.8,
            "fov": 103,
            "game_fps_limit": 300,
            "weapon_hidden": True,
            "calibration_reference_resolution": [1280, 720],
        }
    )
    label: str = ""
    run: RunOptions = field(default_factory=RunOptions)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["run"] = RunOptions(**data.get("run", {}))
        result = cls(**data)
        result.validate()
        return result

    def validate(self):
        self.run.validate()
        if not isinstance(self.algorithm, str) or not self.algorithm:
            raise ValueError("algorithm must be a detector name or module:factory")
        if not isinstance(self.label, str):
            raise ValueError("label must be a string")
        for name in (
            "detector_params",
            "controller_params",
            "gate_params",
            "capture_params",
            "calibration",
            "conditions",
        ):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be an object")
        for name, value in self.calibration.items():
            if name not in {"fx", "fy", "yaw_per_count", "pitch_per_count"}:
                raise ValueError(f"Unknown calibration field: {name}")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"calibration.{name} must be finite and positive")
        size = self.conditions.get("calibration_reference_resolution")
        if size is not None and (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)
        ):
            raise ValueError("calibration_reference_resolution must be [width, height]")

    def to_dict(self):
        return asdict(self)


def load_config(path=None):
    if path is None:
        return AppConfig()
    # utf-8-sig 顺带容忍手工编辑留下的 BOM
    with Path(path).open(encoding="utf-8-sig") as file:
        return AppConfig.from_dict(json.load(file))


def runtime_paths():
    path = project_path(".local/runtime.json")
    local = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {
        "tensorrt_root": os.environ.get("TENSORRT_ROOT", local.get("tensorrt_root", "")),
        "cuda_root": os.environ.get("CUDA_PATH", local.get("cuda_root", "")),
    }
