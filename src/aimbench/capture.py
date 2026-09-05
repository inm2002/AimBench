"""Window ROI on a DXcam WGC output; bounded reads and consumer-owned frames."""

import ctypes
import math

# -4 即 PER_MONITOR_AWARE_V2；没有 DPI 感知，ClientToScreen 会拿到被缩放的坐标
try:
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    pass
import dxcam
import win32gui


class WGCCapture:
    timestamp_domain = "wgc_qpc_seconds"

    def __init__(
        self,
        window_keyword,
        device_idx=None,
        output_idx=None,
        max_buffer_len=8,
        poll_timeout_s=0.05,
    ):
        if type(max_buffer_len) is not int or max_buffer_len < 2:
            raise ValueError("max_buffer_len must be >=2")
        if not math.isfinite(poll_timeout_s) or not 0 < poll_timeout_s <= 0.1:
            raise ValueError("poll_timeout_s must be in (0,0.1]")
        self.window_keyword = window_keyword
        self.requested_output = {"device_idx": device_idx, "output_idx": output_idx}
        self.poll_timeout_s = poll_timeout_s
        self.hwnd = self._find_window(window_keyword)
        if self.hwnd is None:
            raise RuntimeError(f"找不到窗口: {window_keyword}")
        self.title = win32gui.GetWindowText(self.hwnd)
        self.available_outputs = self._dxgi_outputs()
        if device_idx is None and output_idx is None:
            _, _, right, bottom = win32gui.GetClientRect(self.hwnd)
            left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))
            candidates = [
                o
                for o in self.available_outputs
                if o["rect"][0] <= left
                and o["rect"][1] <= top
                and (left + right <= o["rect"][2])
                and (top + bottom <= o["rect"][3])
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"窗口没有完整位于唯一可捕获屏幕内，请把游戏放在单个屏幕：{self.available_outputs}"
                )
            device_idx, output_idx = (candidates[0]["device_idx"], candidates[0]["output_idx"])
        elif device_idx is None or output_idx is None:
            raise ValueError(
                "手动选择 output 时必须同时指定 device_idx 和 output_idx；自动选择均留空"
            )
        self.device_idx, self.output_idx = (device_idx, output_idx)
        self.camera = dxcam.create(
            device_idx=device_idx,
            output_idx=output_idx,
            backend="winrt",
            output_color="BGRA",
            max_buffer_len=max_buffer_len,
        )
        self.running = False
        self.probe = None
        try:
            self.output = self._output_info()
            self.region = self._get_region()
            self._validate_region()
            from aimbench.capture_probe import DXcamProbe

            self.probe = DXcamProbe(self.camera)
        except BaseException:
            self.camera.release()
            raise

    @staticmethod
    def _find_window(keyword):
        matches = []

        def visit(hwnd, _):
            if (
                win32gui.IsWindowVisible(hwnd)
                and keyword.lower() in win32gui.GetWindowText(hwnd).lower()
            ):
                matches.append(hwnd)

        win32gui.EnumWindows(visit, None)
        return matches[0] if matches else None

    @staticmethod
    def _dxgi_outputs():
        factory = getattr(dxcam, "__factory", None)
        if factory is None or not hasattr(factory, "outputs"):
            raise RuntimeError("DXcam 输出枚举接口不兼容，需更新捕获适配器")
        rows = []
        for di, outputs in enumerate(factory.outputs):
            for oi, output in enumerate(outputs):
                output.update_desc()
                r = output.desc.DesktopCoordinates
                rows.append(
                    {
                        "device_idx": di,
                        "output_idx": oi,
                        "name": output.devicename,
                        "rect": [r.left, r.top, r.right, r.bottom],
                    }
                )
        return rows

    def window_state(self):
        foreground = win32gui.GetForegroundWindow()
        data = {"hwnd": self.hwnd, "foreground_hwnd": foreground, "ok": False}
        try:
            flags, cursor, position = win32gui.GetCursorInfo()
            data.update(cursor_visible=bool(flags & 1 and cursor), cursor_screen_xy=list(position))
        except Exception as exc:
            data["cursor_query_error"] = repr(exc)
        # 119=F8、27=Esc，检查按键状态的最高位
        if (
            ctypes.windll.user32.GetAsyncKeyState(119) & 32768
            or ctypes.windll.user32.GetAsyncKeyState(27) & 32768
        ):
            return {**data, "reason": "user_stop_key"}
        if not win32gui.IsWindow(self.hwnd):
            return {**data, "reason": "window_closed"}
        if win32gui.IsIconic(self.hwnd):
            return {**data, "reason": "window_minimized"}
        if foreground != self.hwnd:
            return {**data, "reason": "focus_lost"}
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        a = win32gui.ClientToScreen(self.hwnd, (left, top))
        z = win32gui.ClientToScreen(self.hwnd, (right, bottom))
        rect = [*a, *z]
        if rect != self.window_screen_rect:
            return {**data, "reason": "window_geometry_changed", "rect": rect}
        return {**data, "ok": True, "reason": None}

    def _output_info(self):
        import win32api

        output = self.camera._output
        rect = output.desc.DesktopCoordinates
        data = {
            "device_name": output.devicename,
            "desktop_rect": [rect.left, rect.top, rect.right, rect.bottom],
            "resolution": list(output.resolution),
            "rotation_degrees": output.rotation_angle,
            "hmonitor": int(getattr(output.hmonitor, "value", output.hmonitor) or 0),
        }
        try:
            mode = win32api.EnumDisplaySettings(output.devicename, -1)
            data["refresh_rate_hz"] = mode.DisplayFrequency
            data["mode_size"] = [mode.PelsWidth, mode.PelsHeight]
        except Exception as exc:
            data["mode_error"] = repr(exc)
        return data

    def _get_region(self):
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        a = win32gui.ClientToScreen(self.hwnd, (left, top))
        b = win32gui.ClientToScreen(self.hwnd, (right, bottom))
        self.window_screen_rect = [*a, *b]
        ox, oy = self.output["desktop_rect"][:2]
        # 屏幕坐标换算成相对采集 output 原点的 ROI
        return (a[0] - ox, a[1] - oy, b[0] - ox, b[1] - oy)

    def _validate_region(self):
        left, top, right, bottom = self.region
        if not (0 <= left < right <= self.camera.width and 0 <= top < bottom <= self.camera.height):
            raise RuntimeError(
                f"窗口不在所选 DXcam output 内: window={self.window_screen_rect}, output={self.output}; 请在 capture_params 中设置 device_idx / output_idx"
            )

    def start(self):
        if self.running:
            return
        old_size = (self.width, self.height)
        self.output = self._output_info()
        self.region = self._get_region()
        self._validate_region()
        if old_size != (self.width, self.height):
            raise RuntimeError("窗口尺寸改变，请重新运行以使用正确标定")
        self.camera.start(region=self.region, target_fps=0, video_mode=False)
        self.running = True

    def get_frame(self):
        if not self.running:
            raise RuntimeError("Capture 尚未启动")
        return self.probe.get_frame(self.poll_timeout_s)

    @property
    def last_frame_metadata(self):
        return self.probe.last_frame_metadata if self.probe else {}

    def stop(self):
        if self.running:
            self.camera.stop()
            self.running = False

    def close(self):
        self.stop()
        if self.probe:
            self.probe.close()
        self.camera.release()

    def final_stats(self):
        return self.probe.stats() if self.probe else {}

    def metadata(self):
        return {
            "backend": "dxcam winrt/WGC",
            "color": "BGRA",
            "frame_ownership": "consumer_owned_reusable_copy",
            "video_mode": False,
            "target_fps": 0,
            "latest_frame": True,
            "timestamp_domain": self.timestamp_domain,
            "device_idx": self.device_idx,
            "output_idx": self.output_idx,
            "region": self.region,
            "window_title": self.title,
            "window_screen_rect": self.window_screen_rect,
            "output": self.output,
            "selection": "window_monitor"
            if self.requested_output["device_idx"] is None
            else "explicit",
            "available_outputs": self.available_outputs,
            "capture_scope": "monitor ROI; foreground/geometry and scene guards required",
            "new_frame_wait_bounded": True,
            "poll_timeout_s": self.poll_timeout_s,
            "timeout_scope": "event waiting; OS scheduling, lock and memory copy are not real-time guarantees",
            "buffer_length": self.probe.runtime.max_buffer_len if self.probe else None,
            "adapter": "aimbench.capture_probe.DXcamProbe",
        }

    @property
    def width(self):
        return self.region[2] - self.region[0]

    @property
    def height(self):
        return self.region[3] - self.region[1]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
