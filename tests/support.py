import time

from aimbench.vision.base import VisionResult


class ReplayCapture:
    width, height = 1280, 720

    def __init__(self, sequence):
        self.sequence = iter(sequence)
        self.closed = False

    def start(self):
        pass

    def get_frame(self):
        try:
            return next(self.sequence)
        except StopIteration:
            raise KeyboardInterrupt

    def close(self):
        self.closed = True

    def metadata(self):
        return {"synthetic": True}


class ReplayDetector:
    def __init__(self, delay=0):
        self.delay = delay
        self.closed = False

    def process(self, frame, frame_timestamp):
        if isinstance(frame, BaseException):
            raise frame
        if self.delay:
            time.sleep(self.delay)
        return VisionResult(frame, 0, frame_timestamp)

    def close(self):
        self.closed = True
