"""Game phase synchronization and end classification; no detector-specific rules."""

import time

from .input_guard import InputBlocked


def startup_state(hud):
    seconds = hud.get("time")
    score = hud.get("score")
    if seconds is None or score is None:
        return "waiting"
    if seconds < 60 or score > 0:
        return "late_start"
    return "ready" if seconds == 60 and score == 0 else "waiting"


def synchronize_start(capture, observer, device, guard, cfg, journal, evidence, console):
    """Read before clicking, then require a fresh 01:00/zero-score frame.
    countdown_wait is only the lower bound following our own start click.
    A visibly late round is rejected before any click. Manual starts use
    auto_start_click=False, so no extra countdown wait is inserted.
    All preparation, including pre-run GC, happens before this function.
    """
    begun = time.perf_counter_ns()
    deadline = begun + int(cfg.startup_timeout_seconds * 1e9)
    clicked = False
    checked_initial = False
    not_before = begun
    previous = None
    sequence = 0
    console.line("检查开局状态，等待 01:00 / 0 分的新画面…")
    while time.perf_counter_ns() < deadline:
        guard.require()
        poll = time.perf_counter_ns()
        frame, timestamp = capture.get_frame()
        seen = time.perf_counter_ns()
        if frame is None or timestamp == previous:
            continue
        previous = timestamp
        sequence += 1
        metadata = dict(getattr(capture, "last_frame_metadata", {}) or {})
        published = metadata.get("producer_publish_ns")
        if (
            published is not None
            and cfg.max_publish_age_ms
            and seen - published > cfg.max_publish_age_ms * 1e6
        ):
            continue
        hud = observer.startup_read(frame)
        state = startup_state(hud)
        record = {
            "event_type": "startup_observation",
            "now_ns": seen,
            "capture_timestamp": timestamp,
            "hud": hud,
            "state": state,
            "start_click_sent": clicked,
            "countdown_guard_active": seen < not_before,
        }
        journal.submit(record)
        if state == "late_start":
            evidence.capture(frame, -1, seen, ["late_start"], force=True, critical=True)
            console.line(
                f"本局已开始：剩余 {hud['time']} 秒 / {hud['score']} 分；已停止，回到开始界面后重试。"
            )
            raise InputBlocked("late_start", record)
        if not checked_initial:
            checked_initial = True
            if cfg.auto_start_click:
                begin = getattr(device, "begin_frame", None)
                if callable(begin):
                    begin()
                click_start = time.perf_counter_ns()
                device.click()
                click_end = time.perf_counter_ns()
                journal.submit(
                    {
                        "event_type": "start_click",
                        "start_ns": click_start,
                        "end_ns": click_end,
                        "input_events": list(getattr(device, "events", []) or []),
                    }
                )
                clicked = True
                not_before = click_end + int(cfg.countdown_wait * 1e9)
                deadline = not_before + int(cfg.startup_timeout_seconds * 1e9)
                continue
        if state == "ready" and seen >= not_before:
            journal.submit(
                {
                    "event_type": "startup_ready",
                    "now_ns": seen,
                    "capture_timestamp": timestamp,
                    "hud": hud,
                    "wait_ms": (seen - begun) / 1e6,
                    "start_click_sent": clicked,
                }
            )
            return (frame, timestamp, poll, seen, metadata), {
                "verified_full_start": True,
                "hud": hud,
                "wait_ms": (seen - begun) / 1e6,
                "start_click_sent": clicked,
                "policy": "fresh_hud_after_own_click_countdown_guard",
            }
    raise InputBlocked(
        "startup_timeout",
        {"wait_ms": (time.perf_counter_ns() - begun) / 1e6, "start_click_sent": clicked},
    )


def classify_end(reason, observer, started, now_ns, cfg):
    if reason not in ("game_cursor_visible", "scene_hud_unrecognized", "zero_targets"):
        return reason
    if observer is not None:
        # 只有刚归零的计时器能把弱结束信号判成正常结束；F8/Esc/失焦/暂停不走这里
        changed = observer.last_time_change_ns
        if (
            observer.last_time == 0
            and changed is not None
            and 0 <= now_ns - changed <= 2_500_000_000
        ):
            return "normal_game_end"
        return reason
    if started is not None and (now_ns - started) / 1e9 >= cfg.normal_end_min_seconds:
        return "normal_game_end"
    return reason
