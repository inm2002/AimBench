"""Reusable SendInput buffers with a guard before every native submission."""

import ctypes
from ctypes import wintypes


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class InputUnion(ctypes.Union):
    _fields_ = [("mi", MouseInput)]


class Input(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", InputUnion)]


class MouseInputBackend:
    def __init__(self):
        self._dll = ctypes.WinDLL("user32", use_last_error=True)
        self._send = self._dll.SendInput
        self._send.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        self._send.restype = wintypes.UINT
        self._move = (Input * 1)()
        self._click = (Input * 2)()
        self._release = (Input * 1)()
        # 1=MOVE，2=LEFTDOWN，4=LEFTUP
        for buffer, flags in ((self._move, (1,)), (self._click, (2, 4)), (self._release, (4,))):
            for event, flag in zip(buffer, flags):
                event.type = 0
                event.mi.dwFlags = flag
        self._size = ctypes.sizeof(Input)
        self._button_down_possible = False
        self.guard = None

    def bind_guard(self, guard):
        self.guard = guard

    def _submit(self, buffer, *, click=False, recovery=False):
        # 补发抬起时跳过 guard；就算焦点已丢，按下的左键也要放掉
        if self.guard is not None and not recovery:
            self.guard.require()
        ctypes.set_last_error(0)
        sent = int(self._send(len(buffer), buffer, self._size))
        if click:
            self._button_down_possible = sent == 1
        if sent != len(buffer):
            raise ctypes.WinError(ctypes.get_last_error() or 1)

    def move(self, dx, dy):
        self._move[0].mi.dx = int(dx)
        self._move[0].mi.dy = int(dy)
        self._submit(self._move)

    def click(self):
        self._submit(self._click, click=True)

    def metadata(self):
        return {
            "api": "Win32 SendInput",
            "physical_input": True,
            "move_click_submission": "separate",
            "click_submission": "down/up in one call",
            "reused_input_buffers": True,
            "input_struct_bytes": self._size,
        }

    def close(self):
        if self._button_down_possible:
            self._submit(self._release, recovery=True)
            self._button_down_possible = False
