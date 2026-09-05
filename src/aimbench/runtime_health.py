"""Bounded host diagnostics and scoped cyclic-GC policy for timed runs."""

import ctypes
import gc
import os
import time


class ProcessMemory:
    def __init__(self):
        self.available = os.name == "nt"
        if self.available:
            from ctypes import wintypes as w

            class Counters(ctypes.Structure):
                _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [
                    (n, ctypes.c_size_t)
                    for n in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                        "PrivateUsage",
                    )
                ]

            self.counters = Counters()
            self.fn = ctypes.windll.psapi.GetProcessMemoryInfo
            self.fn.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]
            self.fn.restype = w.BOOL
            self.handle = ctypes.c_void_p(-1)

    def read(self):
        if not self.available:
            return {"available": False}
        c = self.counters
        c.cb = ctypes.sizeof(c)
        if not self.fn(self.handle, ctypes.byref(c), c.cb):
            return {"available": False, "winerror": ctypes.GetLastError()}
        return {
            "available": True,
            "working_set_bytes": c.WorkingSetSize,
            "private_bytes": c.PrivateUsage,
            "peak_working_set_bytes": c.PeakWorkingSetSize,
            "page_fault_count": c.PageFaultCount,
        }


class RuntimeHealth:
    """Check memory growth and restore the caller's cyclic-GC setting."""

    def __init__(self, policy="defer", max_growth_mb=512, interval_s=1.0):
        self.policy = policy
        self.max_growth_bytes = int(max_growth_mb * 1024**2)
        self.interval_ns = int(interval_s * 1e9)
        self.memory = ProcessMemory()
        self.enabled_before = gc.isenabled()
        self.active = False
        self.baseline = {}
        self.last_sample_ns = 0

    def start(self):
        self.enabled_before = gc.isenabled()
        self.active = True
        if self.policy == "defer":
            gc.collect()
            gc.disable()
        self.baseline = self.memory.read()
        self.last_sample_ns = time.perf_counter_ns()

    def sample(self, force=False):
        now = time.perf_counter_ns()
        if not force and now - self.last_sample_ns < self.interval_ns:
            return None
        self.last_sample_ns = now
        memory = self.memory.read()
        private = memory.get("private_bytes", 0)
        growth = max(0, private - self.baseline.get("private_bytes", private))
        return {
            "memory": memory,
            "private_growth_bytes": growth,
            "memory_limit_exceeded": growth > self.max_growth_bytes,
        }

    def close(self):
        if not self.active:
            return
        try:
            if self.policy == "defer":
                gc.collect()
        finally:
            if self.enabled_before:
                gc.enable()
            else:
                gc.disable()
            self.active = False
