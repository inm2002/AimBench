"""Per-frame and immediately-before-native-input checks shared by all detectors."""

import time


class InputBlocked(RuntimeError):
    def __init__(self, reason, details=None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


class InputGuard:
    def __init__(self, capture, physical, scene_required=True, cancel_event=None):
        self.capture = capture
        self.physical = physical
        self.scene_required = scene_required
        self.scene = "startup"
        self.startup = True
        self.deadline_ns = None
        self.last_check = {}
        self.blocked = []
        self.cursor_hidden_seen = False
        self.cancel_event = cancel_event
        if physical and not callable(getattr(capture, "window_state", None)):
            raise ValueError(
                "真实输入需要 capture.window_state() 前台/窗口检查接口；离线插件使用 noop"
            )

    def check(self):
        start = time.perf_counter_ns()
        state = (
            self.capture.window_state()
            if callable(getattr(self.capture, "window_state", None))
            else {"ok": True}
        )
        reason = state.get("reason") if not state.get("ok") else None
        if reason is None and self.cancel_event is not None and self.cancel_event.is_set():
            reason = "session_cancelled"
        if not self.startup and self.scene == "active" and state.get("cursor_visible") is False:
            self.cursor_hidden_seen = True
        # 游戏接管后会隐藏系统光标；一旦见过隐藏，光标再出现即视为切出游戏
        if (
            reason is None
            and not self.startup
            and self.scene_required
            and self.cursor_hidden_seen
            and state.get("cursor_visible")
        ):
            reason = "game_cursor_visible"
        if reason is None and not self.startup and self.scene_required and self.scene != "active":
            reason = "scene_" + self.scene
        if (
            reason is None
            and self.deadline_ns is not None
            and time.perf_counter_ns() >= self.deadline_ns
        ):
            reason = "input_deadline"
        self.last_check = {
            **state,
            "scene": self.scene,
            "reason": reason,
            "ok": reason is None,
            "cursor_hidden_seen": self.cursor_hidden_seen,
            "checked_ns": start,
            "check_ns": time.perf_counter_ns() - start,
        }
        return self.last_check

    def require(self):
        state = self.check()
        if not state["ok"]:
            self.blocked.append(state)
            raise InputBlocked(state["reason"], state)
