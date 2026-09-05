"""Read the latest committed DXcam frame into reusable, consumer-owned memory."""

import time

import numpy as np


class DXcamProbe:
    def __init__(self, camera):
        self.camera = camera
        self.lock = getattr(camera, "_DXCamera__lock", None)
        self.available = getattr(camera, "_DXCamera__frame_available", None)
        self.runtime = getattr(camera, "_DXCamera__capture_runtime", None)
        if self.lock is None or self.available is None or self.runtime is None:
            raise RuntimeError("DXcam 0.3.0 capture interface is required")
        if not callable(getattr(self.runtime, "commit_write", None)):
            raise RuntimeError("DXcam commit interface is unavailable")
        if self.runtime.max_buffer_len < 2:
            raise ValueError("DXcam buffer length must be >= 2")
        self.frequency = int(getattr(camera._duplicator, "performance_frequency", 0))
        if self.frequency <= 0:
            raise RuntimeError("DXcam source clock frequency is invalid")
        self.slots = [None] * self.runtime.max_buffer_len
        self.consumer = None
        self.last_ticks = self.produced_ticks = self.last_consumed_unique = None
        self.unique_produced = self.publications = self.duplicates = self.skipped = (
            self.timeouts
        ) = 0
        self.last_frame_metadata = {}
        self.original_commit = self.runtime.commit_write
        self.runtime.commit_write = self._commit

    def _commit(self, write_idx, frame_ticks):
        # DXcam 在生产者锁内回调这里，该锁与 get_frame() 共用
        if not self.original_commit(write_idx, frame_ticks):
            return False
        self.publications += 1
        if frame_ticks != self.produced_ticks:
            self.unique_produced += 1
            self.produced_ticks = frame_ticks
        else:
            self.duplicates += 1
        self.slots[write_idx] = {
            "producer_publish_ns": time.perf_counter_ns(),
            "source_unique_seq": self.unique_produced,
            "source_ticks": int(frame_ticks),
        }
        return True

    def get_frame(self, timeout_s=0.05):
        deadline = time.perf_counter() + timeout_s
        while True:
            with self.lock:
                runtime = self.runtime
                if runtime.has_frame:
                    slot = (runtime.head - 1) % runtime.max_buffer_len
                    meta = self.slots[slot]
                    if meta and meta["source_ticks"] != self.last_ticks:
                        frame = runtime.frame_buffer[slot]
                        if self.consumer is None:
                            self.consumer = np.empty_like(frame)
                        elif self.consumer.shape != frame.shape:
                            raise RuntimeError(
                                "Capture size changed; restart with matching calibration"
                            )
                        # 锁内复制进自有缓冲，返回后生产者怎么覆写都无妨
                        np.copyto(self.consumer, frame)
                        unique = meta["source_unique_seq"]
                        if self.last_consumed_unique is not None:
                            self.skipped += max(0, unique - self.last_consumed_unique - 1)
                        self.last_ticks = meta["source_ticks"]
                        self.last_consumed_unique = unique
                        self.last_frame_metadata = meta
                        self.available.clear()
                        return self.consumer, meta["source_ticks"] / self.frequency
                self.available.clear()
                if not self.camera.is_capturing:
                    raise RuntimeError("DXcam capture thread stopped")
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self.timeouts += 1
                self.last_frame_metadata = {"timeout": True}
                return None, None
            self.available.wait(min(remaining, 0.05))

    def stats(self):
        return {
            "publications": self.publications,
            "unique_produced": self.unique_produced,
            "duplicate_publications": self.duplicates,
            "consumer_skipped_unique": self.skipped,
            "consumer_wait_timeouts": self.timeouts,
        }

    def close(self):
        self.runtime.commit_write = self.original_commit
