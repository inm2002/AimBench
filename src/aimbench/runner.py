"""Game lifecycle, shared control and compact per-run reporting."""

import math
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from .components import ControllerAdapter, NullInput, SendInputDevice
from .config import AppConfig, project_path
from .console import Console
from .control.camera_model import CameraModel
from .control.freshness import MotionFeedbackTimeout, PredictiveFreshnessGate
from .input_guard import InputBlocked, InputGuard
from .lifecycle import classify_end, synchronize_start
from .metadata import component_metadata, environment, source_fingerprint
from .recording import RunRecorder, Screenshots, write_json
from .registry import create_detector
from .runtime_health import RuntimeHealth


class RunFailure(RuntimeError):
    def __init__(self, message, directory):
        super().__init__(message)
        self.directory = directory


@contextmanager
def protect_finalization():
    # 收尾阶段屏蔽 Ctrl+C，避免 summary 写到一半被打断
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


class RunSession:
    def __init__(
        self, config, *, dry_run=False, detector=None, capture=None, input_device=None, console=None
    ):
        config.validate()
        self.config, self.options = config, config.run
        self.dry_run = dry_run
        self.detector, self.capture, self.device = detector, capture, input_device
        self.console = console or Console()
        self.controller = self.observer = self.guard = None
        self.physical = False
        self.started = None
        self.last_frame = None
        self.zero_since = self.unknown_since = None
        self.counts = {
            "capture_calls": 0,
            "duplicate_polls": 0,
            "empty_polls": 0,
            "stale_results": 0,
        }
        label = "".join(c if c.isalnum() or c in "-_" else "_" for c in config.algorithm)[:40]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = (
            project_path(self.options.output_dir) / f"{stamp}_{label}_{uuid4().hex[:8]}"
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.recorder = RunRecorder()
        self.screenshots = Screenshots(self.directory, self.options.save_result_image)
        self.health = RuntimeHealth(
            self.options.gc_policy,
            self.options.max_memory_growth_mb,
            self.options.health_interval_seconds,
        )
        self.summary = {
            "schema_version": 1,
            "application_version": "1.0.0",
            "run_id": self.directory.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "initializing",
            "algorithm": config.algorithm,
            "label": config.label,
            "config": config.to_dict(),
            "environment": environment(),
            "source_sha256": source_fingerprint(),
            "components": {},
            "game_result": None,
            "synthetic": False,
            "physical_input": False,
        }

    def _prepare(self):
        if self.detector is None:
            self.detector, spec = create_detector(
                self.config.algorithm, self.config.detector_params
            )
        else:
            spec = {"factory": type(self.detector).__name__, "parameters": {}}
        self.summary["components"]["detector"] = component_metadata(self.detector, spec)
        if self.capture is None:
            from .capture import WGCCapture

            self.capture = WGCCapture(self.options.window_keyword, **self.config.capture_params)
        self.summary["synthetic"] = bool(
            getattr(self.capture, "metadata", lambda: {})().get("synthetic")
        )
        reference = self.config.conditions.get("calibration_reference_resolution")
        if reference and list(reference) != [self.capture.width, self.capture.height]:
            raise ValueError("Client dimensions differ from calibration_reference_resolution")
        if self.device is None:
            self.device = NullInput() if self.dry_run else SendInputDevice()
        self.physical = bool(self.device.metadata().get("physical_input"))
        if self.summary["synthetic"] and self.physical:
            raise ValueError("Synthetic capture requires null input")
        if self.dry_run and self.physical:
            raise ValueError("Dry runs cannot use physical input")
        if hasattr(self.capture, "hwnd"):
            from .game_observer import GameObserver

            self.observer = GameObserver(timer_hold_ms=self.options.timer_hold_ms)
        self.guard = InputGuard(
            self.capture, self.physical, scene_required=self.observer is not None
        )
        bind = getattr(self.device, "bind_guard", None)
        if self.physical and not callable(bind):
            raise ValueError("Physical input must implement bind_guard()")
        if callable(bind):
            bind(self.guard)
        camera = CameraModel(self.capture.width, self.capture.height, **self.config.calibration)
        self.controller = ControllerAdapter(
            camera, self.device.move, self.device.click, self.config.controller_params
        )
        self.controller.bind_gate(PredictiveFreshnessGate(**self.config.gate_params))
        self.summary["physical_input"] = self.physical
        self.summary["components"]["input"] = self.device.metadata()
        self.summary["components"]["controller"] = self.controller.metadata()
        self.summary["components"]["gate"] = self.controller.policy.gate_policy.metadata()
        self.capture.start()
        metadata = getattr(self.capture, "metadata", lambda: {})()
        self.summary["components"]["capture"] = {
            key: metadata[key]
            for key in ("backend", "output", "window_screen_rect", "timestamp_domain")
            if key in metadata
        }
        self.console.start(self.config, self.directory)
        time.sleep(self.options.start_delay)
        if hasattr(self.capture, "probe"):
            self.capture.get_frame()
        self.health.start()
        if self.observer and self.physical:
            pending, startup = synchronize_start(
                self.capture,
                self.observer,
                self.device,
                self.guard,
                self.options,
                self.recorder,
                self.screenshots,
                self.console,
            )
            self.summary["startup"] = startup
            return pending
        self.summary["startup"] = {"verified_full_start": False}
        return None

    def _frame_action(self, result, seen, published, deadline):
        cfg = self.options
        now = time.perf_counter_ns()
        if result.detections:
            self.zero_since = None
        guard_state = self.guard.last_check
        if now >= deadline:
            return "NOT_CALLED_DEADLINE", "hard_stop"
        if not guard_state["ok"]:
            state = self.guard.scene
            reason = guard_state["reason"]
            self.unknown_since = self.unknown_since if self.unknown_since is not None else now
            classified = classify_end(reason, self.observer, self.started, now, cfg)
            # 暂停/计时停滞/非场景类原因立即停；场景原因给一段宽限期
            stop = state in {"paused", "timer_stalled"} or not reason.startswith("scene_")
            stop |= (
                classified == "normal_game_end"
                or now - self.unknown_since >= cfg.scene_unknown_grace_ms * 1e6
            )
            return "INPUT_BLOCKED", classified if stop else None
        if not result.detections:
            self.zero_since = self.zero_since if self.zero_since is not None else now
            if now - self.zero_since >= cfg.zero_target_grace_ms * 1e6:
                return "NO_TARGET", classify_end(
                    "zero_targets", self.observer, self.started, now, cfg
                )
            return "WAIT_ZERO_TARGETS", None
        self.zero_since = None
        # 两个时效口径：主机收到帧的 seen，与生产端发布时间 published
        stale = bool(cfg.max_result_age_ms and now - seen > cfg.max_result_age_ms * 1e6)
        stale = stale or bool(
            published is not None
            and cfg.max_publish_age_ms
            and now - published > cfg.max_publish_age_ms * 1e6
        )
        if stale:
            self.counts["stale_results"] += 1
            return "NOT_CALLED_STALE_RESULT", None
        self.unknown_since = None
        limits = [deadline]
        if cfg.max_result_age_ms:
            limits.append(seen + int(cfg.max_result_age_ms * 1e6))
        if published is not None and cfg.max_publish_age_ms:
            limits.append(published + int(cfg.max_publish_age_ms * 1e6))
        self.guard.deadline_ns = min(limits)
        self.controller.set_frame_deadline(self.guard.deadline_ns)
        action = self.controller.process(result.detections, self.recorder.frame_count + 1)
        if action == "REJECT_STALE_PLAN":
            self.counts["stale_results"] += 1
        return action, None

    def _process_frame(self, frame, timestamp, seen, published, capture_ms, deadline):
        stages = {"capture": capture_ms, "vision": 0.0, "control": 0.0, "observer": 0.0}
        action, reason, targets = "ERROR", None, 0
        try:
            if self.observer:
                observation = self.observer.observe(frame, self.recorder.frame_count + 1, seen)
                self.guard.scene = observation["state"]
                stages["observer"] = observation["observer_ms"]
                if observation["state_changed"]:
                    self.recorder.submit(
                        {"event_type": "game_state", "state": observation["state"]}
                    )
            else:
                self.guard.scene = "active"
            self.guard.deadline_ns = deadline
            self.guard.check()
            begin_frame = getattr(self.device, "begin_frame", None)
            if callable(begin_frame):
                begin_frame()
            begin = time.perf_counter_ns()
            result = self.detector.process(frame, timestamp)
            stages["vision"] = (time.perf_counter_ns() - begin) / 1e6
            if not math.isfinite(float(result.frame_timestamp)) or float(
                result.frame_timestamp
            ) != float(timestamp):
                raise ValueError("Detector results must belong to the current capture")
            for detection in result.detections:
                if not all(
                    math.isfinite(float(getattr(detection, key)))
                    for key in ("x", "y", "area", "confidence")
                ):
                    raise ValueError("Detector returned nonfinite coordinates or confidence")
            targets = len(result.detections)
            begin = time.perf_counter_ns()
            try:
                action, reason = self._frame_action(result, seen, published, deadline)
            finally:
                stages["control"] = (time.perf_counter_ns() - begin) / 1e6
        except MotionFeedbackTimeout:
            action, reason = "MOTION_FEEDBACK_TIMEOUT", "pending_motion_timeout"
        except InputBlocked as exc:
            action = "INPUT_BLOCKED"
            reason = classify_end(
                exc.reason, self.observer, self.started, time.perf_counter_ns(), self.options
            )
        finally:
            stats = self.controller.stats
            self.recorder.record_frame(
                (seen - self.started) / 1e9,
                stages,
                targets,
                action,
                stats.mouse_moves,
                stats.fire_attempts,
            )
        return reason

    def _loop(self, pending):
        cfg = self.options
        self.started = pending[3] if pending else time.perf_counter_ns()
        deadline = self.started + int(cfg.max_run_seconds * 1e9)
        self.guard.startup = False
        self.guard.deadline_ns = deadline
        self.summary["status"] = "running"
        self.recorder.submit({"event_type": "run_start"})
        self.console.line("运行中；F8 / Esc 或切出游戏可停止。")
        previous = None
        last_seen = self.started
        capture_ns = 0
        while True:
            poll = pending[2] if pending else time.perf_counter_ns()
            if poll >= deadline:
                return "hard_stop"
            if callable(getattr(self.capture, "window_state", None)):
                state = self.capture.window_state()
                if not state["ok"]:
                    raise InputBlocked(state["reason"], state)
            # 开局同步留下的那一帧直接作为循环首帧
            if pending:
                frame, timestamp, _, seen, metadata = pending
                pending = None
            else:
                frame, timestamp = self.capture.get_frame()
                seen = time.perf_counter_ns()
                metadata = getattr(self.capture, "last_frame_metadata", {}) or {}
            self.counts["capture_calls"] += 1
            capture_ns += seen - poll
            if seen >= deadline:
                return "hard_stop"
            if frame is None or timestamp == previous:
                self.counts["empty_polls" if frame is None else "duplicate_polls"] += 1
                if seen - last_seen >= cfg.capture_timeout_seconds * 1e9:
                    return "capture_stalled"
                continue
            if timestamp is None or not math.isfinite(float(timestamp)):
                raise ValueError("Capture must provide a finite timestamp")
            if previous is not None and timestamp < previous:
                raise ValueError("Capture timestamp moved backwards")
            previous, last_seen = timestamp, seen
            self.last_frame = frame
            reason = self._process_frame(
                frame,
                timestamp,
                seen,
                metadata.get("producer_publish_ns"),
                capture_ns / 1e6,
                deadline,
            )
            capture_ns = 0
            sample = self.health.sample()
            if sample:
                self.recorder.observe_health(sample)
                if sample["memory_limit_exceeded"]:
                    return "memory_limit"
            if reason:
                return reason

    def _finish(self, reason, failure):
        ended = time.perf_counter_ns()
        cleanup_errors = []

        def close(name, component):
            method = getattr(component, "close", None)
            if callable(method):
                try:
                    return method()
                except Exception as exc:
                    cleanup_errors.append(f"{name}: {exc}")
            return None

        # 先关输入（顺带补发按键释放），再等待结算画面或停止采集
        close("input", self.device)
        if (
            self.started is not None
            and self.physical
            and reason in {"normal_game_end", "hard_stop"}
            and self.options.save_result_image
        ):
            try:
                from .result_capture import capture_result

                self.summary["result_capture"] = capture_result(
                    self.capture, self.screenshots, self.options, self.console
                )
            except Exception as exc:
                cleanup_errors.append(f"result image: {exc}")
        elif self.last_frame is not None and reason not in {"normal_game_end", "hard_stop"}:
            self.screenshots.capture(self.last_frame, -1, ended, [reason], force=True)
        try:
            self.summary["capture_counts"] = getattr(self.capture, "final_stats", lambda: {})()
        except Exception as exc:
            cleanup_errors.append(f"capture statistics: {exc}")
        close("capture", self.capture)
        close("detector", self.detector)
        try:
            if self.health.active:
                sample = self.health.sample(force=True)
                self.recorder.observe_health(sample)
        except Exception as exc:
            cleanup_errors.append(f"memory statistics: {exc}")
        finally:
            close("health", self.health)
        observed = close("observer", self.observer)
        images = close("screenshots", self.screenshots)
        self.recorder.submit({"event_type": "run_end", "reason": reason})
        self.summary.update(self.recorder.finish(self.directory))
        self.summary.update(
            status="finished" if reason == "normal_game_end" else "incomplete",
            end_reason=reason,
            elapsed_s=(ended - self.started) / 1e9 if self.started is not None else 0.0,
            counts=self.counts,
            controller=self.controller.final_stats() if self.controller else {},
            observer=observed,
            images=images or [],
            error=str(failure) if failure else None,
            cleanup_errors=cleanup_errors,
        )
        self.summary["eligible_for_comparison"] = (
            reason == "normal_game_end"
            and self.physical
            and not self.summary["synthetic"]
            and self.summary.get("startup", {}).get("verified_full_start", False)
            and not cleanup_errors
        )
        write_json(self.directory / "summary.json", self.summary)
        self.console.summary(self.summary, self.directory)

    def execute(self):
        reason, failure = "initialization_error", None
        write_json(self.directory / "summary.json", self.summary)
        try:
            pending = self._prepare()
            reason = self._loop(pending)
        except InputBlocked as exc:
            reason = classify_end(
                exc.reason, self.observer, self.started, time.perf_counter_ns(), self.options
            )
            self.recorder.submit({"event_type": "input_blocked", "reason": exc.reason})
        except KeyboardInterrupt:
            reason = "interrupted"
        except Exception as exc:
            failure = exc
            reason = "runtime_error" if self.started is not None else "initialization_error"
            (self.directory / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            with protect_finalization():
                self._finish(reason, failure)
        if failure is not None:
            raise RunFailure(str(failure), self.directory) from failure
        return self.summary, self.directory


def run(config=None, **kwargs):
    return RunSession(config or AppConfig(), **kwargs).execute()
