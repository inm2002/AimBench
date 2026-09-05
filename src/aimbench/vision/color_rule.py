import time

import cv2
import numpy as np

from .base import Detection, VisionBackend, VisionResult


class ColorRuleDetector(VisionBackend):
    def __init__(
        self,
        lower_bgra=(170, 160, 0, 0),
        upper_bgra=(255, 255, 80, 255),
        min_area=100,
        max_area=10000,
        max_targets=3,
        min_circularity=0.45,
        min_aspect=0.55,
        max_aspect=1.8,
    ):
        # 绿/蓝下限抬高，挡掉天空与 HUD 的青色碎片
        self.lower = np.array(lower_bgra, dtype=np.uint8)
        self.upper = np.array(upper_bgra, dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        self.max_targets = max_targets
        self.min_circularity = min_circularity
        self.min_aspect, self.max_aspect = (min_aspect, max_aspect)

    @property
    def name(self):
        return "Color Rule (BGRA + Contour Moments)"

    def metadata(self):
        return {
            "version": "balanced_cyan_v2",
            "lower_bgra": self.lower.tolist(),
            "upper_bgra": self.upper.tolist(),
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_circularity": self.min_circularity,
            "min_aspect": self.min_aspect,
            "max_aspect": self.max_aspect,
            "max_targets": self.max_targets,
            "reason": "higher green/blue floors reject cyan sky/HUD fragments without extra per-pixel passes",
        }

    def process(self, frame, frame_timestamp):
        start_ns = time.perf_counter_ns()
        mask = cv2.inRange(frame, self.lower, self.upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            moments = cv2.moments(contour)
            m00 = moments["m00"]
            if m00 != 0:
                x = moments["m10"] / m00
                y = moments["m01"] / m00
            else:
                # 退化轮廓拿不到质心，先给占位值，后面 m00==0 一并过滤
                x = -1.0
                y = -1.0
            if area < self.min_area:
                continue
            if area > self.max_area:
                continue
            if m00 == 0:
                continue
            _, _, width, height = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(1.0, perimeter * perimeter)
            if (
                circularity < self.min_circularity
                or not self.min_aspect <= width / height <= self.max_aspect
            ):
                continue
            detections.append(Detection(x=float(x), y=float(y), area=float(area), confidence=1.0))
        detections.sort(key=lambda d: d.area, reverse=True)
        detections = detections[: self.max_targets]
        end_ns = time.perf_counter_ns()
        return VisionResult(
            detections=detections,
            process_ms=(end_ns - start_ns) / 1000000.0,
            frame_timestamp=frame_timestamp,
        )
