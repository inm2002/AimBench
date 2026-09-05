import gc
import json
import time
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from aimbench.capture_probe import DXcamProbe
from aimbench.components import NullInput
from aimbench.config import ASSETS, AppConfig, RunOptions
from aimbench.console import Console
from aimbench.control.input_backend import MouseInputBackend
from aimbench.input_guard import InputBlocked, InputGuard
from aimbench.lifecycle import classify_end, synchronize_start
from aimbench.recording import Metric, RunRecorder, Screenshots, write_json
from aimbench.result_capture import capture_result
from aimbench.results import save_result
from aimbench.runner import RunFailure, run
from aimbench.vision.base import Detection

from .support import ReplayCapture, ReplayDetector


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.console = Console(StringIO())

    def run_config(self, **options):
        return AppConfig.from_dict(
            {
                "run": {
                    "output_dir": str(self.path),
                    "start_delay": 0,
                    "countdown_wait": 0,
                    "save_result_image": False,
                    "prompt_result": False,
                    **options,
                }
            }
        )

    def test_replay_writes_only_compact_results_and_restores_gc(self):
        capture = ReplayCapture([([Detection(642, 360)], 1), ([Detection(642, 360)], 2)])
        detector, device = ReplayDetector(), NullInput()
        before = gc.isenabled()
        summary, path = run(
            self.run_config(),
            capture=capture,
            detector=detector,
            input_device=device,
            console=self.console,
        )
        self.assertEqual((summary["frames"], device.clicks), (2, 2))
        self.assertFalse(summary["eligible_for_comparison"])
        self.assertEqual(set(p.name for p in path.iterdir()), {"summary.json", "metrics.csv"})
        self.assertTrue(capture.closed and detector.closed)
        self.assertEqual(gc.isenabled(), before)

    def test_stale_detection_does_not_click(self):
        device = NullInput()
        summary, _ = run(
            self.run_config(max_result_age_ms=1),
            capture=ReplayCapture([([Detection(642, 360)], 1)]),
            detector=ReplayDetector(delay=0.01),
            input_device=device,
            console=self.console,
        )
        self.assertEqual((device.clicks, summary["counts"]["stale_results"]), (0, 1))

    def test_cleanup_error_still_restores_gc_and_saves_summary(self):
        capture = ReplayCapture([([Detection(642, 360)], 1)])
        capture.final_stats = lambda: (_ for _ in ()).throw(RuntimeError("statistics failure"))
        before = gc.isenabled()
        summary, path = run(
            self.run_config(),
            capture=capture,
            detector=ReplayDetector(),
            input_device=NullInput(),
            console=self.console,
        )
        self.assertTrue(capture.closed)
        self.assertEqual(gc.isenabled(), before)
        self.assertTrue(summary["cleanup_errors"])
        self.assertTrue((path / "summary.json").is_file())

    def test_disabled_age_limits_are_supported(self):
        device = NullInput()
        run(
            self.run_config(max_result_age_ms=0, max_publish_age_ms=0),
            capture=ReplayCapture([([Detection(642, 360)], 1)]),
            detector=ReplayDetector(),
            input_device=device,
            console=self.console,
        )
        self.assertEqual(device.clicks, 1)

    def test_invalid_detector_result_stops_and_closes(self):
        device, capture, detector = (
            NullInput(),
            ReplayCapture([([Detection(float("nan"), 1)], 1)]),
            ReplayDetector(),
        )
        before = gc.isenabled()
        with self.assertRaises(RunFailure) as caught:
            run(
                self.run_config(),
                capture=capture,
                detector=detector,
                input_device=device,
                console=self.console,
            )
        self.assertEqual(device.clicks, 0)
        self.assertTrue(capture.closed and detector.closed)
        self.assertEqual(gc.isenabled(), before)
        self.assertTrue((caught.exception.directory / "error.log").is_file())
        summary = json.loads(
            (caught.exception.directory / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["end_reason"], "runtime_error")

    def test_capture_owns_pixels_and_skips_duplicates(self):
        class Runtime:
            max_buffer_len = 2
            head = 0
            has_frame = False
            frame_buffer = np.zeros((2, 2, 2, 4), np.uint8)

            def commit_write(self, index, ticks):
                self.head, self.has_frame = (index + 1) % 2, True
                return True

        runtime = Runtime()
        camera = SimpleNamespace(
            is_capturing=True, _duplicator=SimpleNamespace(performance_frequency=10**9)
        )
        for key, value in {
            "_DXCamera__lock": Lock(),
            "_DXCamera__frame_available": Event(),
            "_DXCamera__capture_runtime": runtime,
        }.items():
            setattr(camera, key, value)
        original = runtime.commit_write
        probe = DXcamProbe(camera)
        runtime.frame_buffer[0].fill(7)
        runtime.commit_write(0, 1234)
        pixels, timestamp = probe.get_frame(0.001)
        runtime.frame_buffer[0].fill(99)
        self.assertTrue(np.all(pixels == 7))
        self.assertEqual(timestamp, 1234 / 1e9)
        self.assertEqual(probe.get_frame(0.001), (None, None))
        self.assertEqual(probe.stats()["consumer_wait_timeouts"], 1)
        probe.close()
        self.assertEqual(runtime.commit_write, original)

    def test_partial_click_releases_button_even_when_guard_blocks(self):
        with patch("ctypes.WinDLL") as dll:
            dll.return_value.SendInput.side_effect = [1, 1]
            backend = MouseInputBackend()
            with self.assertRaises(OSError):
                backend.click()
            backend.bind_guard(
                SimpleNamespace(require=lambda: (_ for _ in ()).throw(InputBlocked("focus_lost")))
            )
            backend.close()
            self.assertFalse(backend._button_down_possible)
            self.assertEqual(dll.return_value.SendInput.call_count, 2)

    def test_end_requires_recent_zero_timer(self):
        now = 60_000_000_000
        observer = SimpleNamespace(last_time=0, last_time_change_ns=now - 100_000_000)
        self.assertEqual(
            classify_end("zero_targets", observer, 0, now, RunOptions()), "normal_game_end"
        )
        self.assertEqual(classify_end("focus_lost", observer, 0, now, RunOptions()), "focus_lost")
        observer.last_time = 53
        self.assertEqual(
            classify_end("zero_targets", observer, 0, now, RunOptions()), "zero_targets"
        )

    def test_late_start_is_rejected_before_click(self):
        frame = np.zeros((32, 32, 4), np.uint8)
        capture = SimpleNamespace(
            get_frame=lambda: (frame, time.perf_counter()), window_state=lambda: {"ok": True}
        )
        observer = SimpleNamespace(startup_read=lambda _: {"score": 0, "time": 56})
        device = NullInput()
        with self.assertRaises(InputBlocked) as caught:
            synchronize_start(
                capture,
                observer,
                device,
                InputGuard(capture, False),
                RunOptions(),
                RunRecorder(),
                Screenshots(self.path),
                self.console,
            )
        self.assertEqual(caught.exception.reason, "late_start")
        self.assertEqual(device.clicks, 0)

    def test_metric_storage_is_bounded(self):
        metric = Metric()
        with patch.object(Metric, "limit", 10):
            for value in range(100):
                metric.add(value)
        summary = metric.summary()
        self.assertEqual((summary["count"], summary["retained_samples"]), (100, 10))
        self.assertTrue(summary["percentiles_truncated"])
        self.assertEqual(summary["mean_ms"], 49.5)

    def test_recording_aggregates_one_row_per_second(self):
        recorder = RunRecorder()
        for frame in range(600):
            recorder.record_frame(frame / 100, dict.fromkeys(recorder.stages, 0.1), 3, "WAIT", 0, 0)
        summary = recorder.finish(self.path)
        self.assertEqual(summary["recording"]["metrics_rows"], 6)
        self.assertEqual(summary["recording"]["numeric_sample_bytes"], 600 * 4 * 4)

    def test_result_preserves_shots_and_targets(self):
        write_json(self.path / "summary.json", {"run_id": "test"})
        result = save_result(self.path, 226752, 100, 588, 591)
        self.assertEqual((result["shots"], result["total_targets"]), (588, 591))
        with self.assertRaises(ValueError):
            save_result(self.path, 1, 101, 1)

    def test_result_image_waits_for_stability_without_sample_history(self):
        image = np.zeros((720, 1280, 4), np.uint8)
        labels = cv2.imdecode(
            np.fromfile(ASSETS / "gridshot_result_labels.png", np.uint8), cv2.IMREAD_GRAYSCALE
        )
        image[174:188, 990:1178, :3] = labels[:, :, None]
        clock = SimpleNamespace(now=0)

        def sleep(seconds):
            clock.now += int(seconds * 1e9)

        capture = SimpleNamespace(get_frame=lambda: (image, clock.now / 1e9))
        options = RunOptions(
            result_screen_delay_seconds=0,
            result_screen_timeout_seconds=1,
            result_screen_stable_seconds=0.4,
        )
        with (
            patch("time.perf_counter_ns", side_effect=lambda: clock.now),
            patch("time.sleep", side_effect=sleep),
        ):
            result = capture_result(capture, Screenshots(self.path), options, self.console)
        self.assertTrue(result["stable"])
        self.assertGreaterEqual(result["wall_ms"], 400)
        self.assertNotIn("samples", result)
