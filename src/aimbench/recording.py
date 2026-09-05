"""Bounded numeric metrics and a small lifecycle log; disk writes happen after input stops."""

import csv
import json
import math
import os
from array import array
from collections import Counter, deque
from dataclasses import asdict, is_dataclass
from pathlib import Path


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class Metric:
    """Exact totals with at most 100,000 retained numeric percentile samples."""

    limit = 100_000

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.samples = array("f")

    def add(self, value):
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        if len(self.samples) < self.limit:
            self.samples.append(value)

    def summary(self):
        values = sorted(self.samples)

        def percentile(fraction):
            if not values:
                return None
            index = (len(values) - 1) * fraction
            lo, hi = math.floor(index), math.ceil(index)
            return values[lo] + (values[hi] - values[lo]) * (index - lo)

        return {
            "count": self.count,
            "mean_ms": self.total / self.count if self.count else None,
            "p50_ms": percentile(0.5),
            "p95_ms": percentile(0.95),
            "max_ms": self.maximum if self.count else None,
            "retained_samples": len(values),
            "percentiles_truncated": len(values) < self.count,
        }


class RunRecorder:
    """Aggregate one row per second without keeping coordinates, images or input traces."""

    stages = ("capture", "vision", "control", "observer")

    def __init__(self):
        self.metrics = {name: Metric() for name in self.stages}
        self.actions = Counter()
        self.target_counts = Counter()
        self.events = deque(maxlen=64)
        self.event_count = 0
        self.rows = []
        self.window = None
        self.frame_count = 0
        self.health_peak = 0
        self.health_growth = 0

    def submit(self, event):
        if event.get("event_type") == "startup_observation":
            return
        self.event_count += 1
        keep = {
            key: value
            for key, value in event.items()
            if key not in {"input_events", "controller_stats", "capture_stats"}
        }
        self.events.append(keep)

    def record_frame(self, elapsed, durations, targets, action, moves, clicks):
        second = int(elapsed)
        if self.window is None or self.window["second"] != second:
            self._finish_window()
            self.window = {"second": second, "frames": 0, **{name: 0.0 for name in self.stages}}
        self.frame_count += 1
        self.window["frames"] += 1
        self.window["moves_end"] = moves
        self.window["clicks_end"] = clicks
        self.actions[action] += 1
        self.target_counts[targets] += 1
        for name, value in durations.items():
            self.metrics[name].add(value)
            self.window[name] += value

    def _finish_window(self):
        if self.window is None:
            return
        window = self.window
        self.rows.append(
            {
                "second": window["second"],
                "frames": window["frames"],
                "moves_total": window["moves_end"],
                "clicks_total": window["clicks_end"],
                **{name + "_mean_ms": window[name] / window["frames"] for name in self.stages},
            }
        )
        self.window = None

    def observe_health(self, sample):
        self.health_peak = max(
            self.health_peak, sample.get("memory", {}).get("private_bytes", 0) or 0
        )
        self.health_growth = max(self.health_growth, sample.get("private_growth_bytes", 0) or 0)

    def finish(self, directory):
        self._finish_window()
        directory = Path(directory)
        if self.rows:
            with (directory / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        return {
            "frames": self.frame_count,
            "actions": dict(self.actions),
            "target_counts": dict(self.target_counts),
            "latency": {name: metric.summary() for name, metric in self.metrics.items()},
            "events": list(self.events),
            "events_truncated": max(0, self.event_count - len(self.events)),
            "memory": {
                "peak_private_bytes": self.health_peak,
                "max_growth_bytes": self.health_growth,
            },
            "recording": {
                "metrics_rows": len(self.rows),
                "interval_seconds": 1,
                "numeric_sample_bytes": sum(
                    len(metric.samples) * metric.samples.itemsize
                    for metric in self.metrics.values()
                ),
                "scope": "Host stage durations and counts; no per-frame coordinates or causal hit attribution.",
            },
        }


class Screenshots:
    """Retain at most one failure image and one result image; encode after the run."""

    def __init__(self, directory, enabled=True):
        self.directory = Path(directory)
        self.enabled = enabled
        self.images = {}

    def capture(self, frame, seq, now, reasons=(), detector=None, force=False, critical=False):
        if self.enabled and hasattr(frame, "copy"):
            # seq 为 0 约定是结算图，其余一律按失败现场存
            name = "result" if seq == 0 else "failure"
            self.images[name] = frame.copy()

    def close(self):
        import cv2

        saved = []
        for name, frame in self.images.items():
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError(f"Could not encode {name} image")
            path = self.directory / f"{name}.png"
            encoded.tofile(path)
            saved.append(path.name)
        self.images.clear()
        return saved
