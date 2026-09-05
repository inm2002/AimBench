"""Wait for stable result evidence only after the input backend is closed."""

import time

import cv2

from .game_observer import ASSETS


def capture_result(capture, evidence, cfg, console):
    import numpy as np

    started = time.perf_counter_ns()
    sample_count = 0
    last_key = None
    stable_since = None
    marker = cv2.imdecode(
        np.fromfile(ASSETS / "gridshot_result_labels.png", np.uint8), cv2.IMREAD_GRAYSCALE
    )
    result = {
        "status": "timeout",
        "stable": False,
        "input_closed": True,
        "scope": "result display evidence outside the timed run; pixel stability is not numeric OCR",
    }
    console.line("输入已停止，等待结算数字稳定并保存截图…")
    last_frame = last_timestamp = None
    while (time.perf_counter_ns() - started) / 1e9 < cfg.result_screen_timeout_seconds:
        state = (
            capture.window_state()
            if callable(getattr(capture, "window_state", None))
            else {"ok": True}
        )
        if not state.get("ok"):
            result["status"] = "window_unavailable"
            result["window_state"] = state
            break
        frame, timestamp = capture.get_frame()
        now = time.perf_counter_ns()
        if frame is None:
            time.sleep(0.05)
            continue
        h, w = frame.shape[:2]

        def roi(box):
            x, y, r, b = box
            image = frame[
                round(y * h / 720) : round(b * h / 720), round(x * w / 1280) : round(r * w / 1280)
            ]
            gray = cv2.cvtColor(
                image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            )
            return cv2.resize(gray, (r - x, b - y)) if gray.shape != (b - y, r - x) else gray

        labels = roi((990, 174, 1178, 188))
        match = float(cv2.matchTemplate(labels, marker, cv2.TM_CCOEFF_NORMED)[0, 0])
        # 回放视频、历史图表、悬停按钮都在持续动画，不适合当稳定判据
        key = roi((300, 58, 565, 98)).tobytes() + roi((637, 160, 1180, 174)).tobytes()
        if match < 0.85 or key != last_key:
            stable_since = now
        last_key = key
        stable_ms = (now - stable_since) / 1e6
        sample_count += 1
        if match >= 0.85:
            last_frame = frame.copy()
            last_timestamp = timestamp
            if (
                (now - started) / 1e9 >= cfg.result_screen_delay_seconds
                and stable_ms >= cfg.result_screen_stable_seconds * 1000
            ):
                result.update(status="stable", stable=True)
                break
        time.sleep(0.2)
    if last_frame is not None:
        evidence.capture(
            last_frame,
            0,
            time.perf_counter_ns(),
            ["stable_result_screen" if result["stable"] else "unsettled_result_screen"],
            force=True,
        )
        result["capture_timestamp"] = last_timestamp
    result["wall_ms"] = (time.perf_counter_ns() - started) / 1e6
    result["sample_count"] = sample_count
    return result
