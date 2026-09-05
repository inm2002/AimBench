"""Grayscale template matching with local geometry validation, no color detector."""

import math
import time
from pathlib import Path

import cv2
import numpy as np

from .base import Detection, VisionBackend, VisionResult


class TemplateDetector(VisionBackend):
    def __init__(
        self,
        template_path,
        scales=(0.85, 1.0, 1.2, 1.45),
        max_targets=3,
        coarse_factor=4,
        coarse_threshold=0.65,
        refine_threshold=0.68,
        min_contrast=18.0,
        min_circularity=0.70,
        max_candidates=12,
        scale_policy="common_then_fallback",
    ):
        self.template_path = str(Path(template_path))
        original = cv2.imdecode(np.fromfile(template_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        if original is None or min(original.shape) < 16 or original.std() < 2:
            raise ValueError("模板无法读取、过小或没有足够纹理")
        if (
            coarse_factor < 1
            or max_targets < 1
            or max_candidates < 1
            or not scales
            or any(s <= 0 for s in scales)
        ):
            raise ValueError("Invalid template scales/capacity")
        if (
            not 0 < coarse_threshold <= 1
            or not 0 < refine_threshold <= 1
            or not 0 <= min_circularity <= 1
            or min_contrast < 0
        ):
            raise ValueError("Invalid template thresholds")
        self.factor = coarse_factor
        self.max_targets = max_targets
        self.coarse_threshold = coarse_threshold
        self.refine_threshold = refine_threshold
        self.min_contrast = min_contrast
        self.min_circularity = min_circularity
        self.max_candidates = max_candidates
        self.bank = []
        self.last_profile = {}
        if scale_policy not in ("all", "common_then_fallback"):
            raise ValueError("scale_policy must be all or common_then_fallback")
        self.scale_policy = scale_policy
        self.masks = {}
        for scale in scales:
            w, h = [max(12, int(round(v * scale))) for v in original.shape[::-1]]
            full = cv2.resize(original, (w, h), interpolation=cv2.INTER_LINEAR)
            small = cv2.resize(
                full,
                (max(4, w // coarse_factor), max(4, h // coarse_factor)),
                interpolation=cv2.INTER_AREA,
            )
            self.bank.append((full, small, float(scale)))
            yy, xx = np.ogrid[:h, :w]
            radius = ((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / (min(w, h) * 0.39) ** 2
            self.masks[(h, w)] = (radius < 0.40, (radius > 1.12) & (radius < 1.55))
        self.primary_indices = tuple(
            i for i, b in enumerate(self.bank) if scale_policy == "all" or b[2] >= 1.0
        )
        if not self.primary_indices:
            self.primary_indices = tuple(range(len(self.bank)))
        self.remaining_indices = tuple(
            i for i in range(len(self.bank)) if i not in self.primary_indices
        )
        self.primary_scales = [self.bank[i][2] for i in self.primary_indices]
        self.fallback_scales = [
            self.bank[i][2] for i in self.primary_indices + self.remaining_indices
        ]

    @property
    def name(self):
        return "Grayscale multiscale template + circular contrast validation"

    def metadata(self):
        return {
            "version": "template_v4",
            "color_segmentation": False,
            "scale_policy": self.scale_policy,
            "fallback": "search remaining scales if primary detections are fewer than max_targets",
            "validation": "grayscale center component / circularity / local contrast",
            "coarse_threshold": self.coarse_threshold,
            "refine_threshold": self.refine_threshold,
            "min_contrast": self.min_contrast,
            "min_circularity": self.min_circularity,
            "scales": [b[2] for b in self.bank],
            "forced_target_count": False,
        }

    def _validate(self, gray, x, y, w, h):
        patch = gray[y : y + h, x : x + w]
        cy, cx = (h - 1) / 2, (w - 1) / 2
        r = min(w, h) * 0.39
        inside_mask, outside_mask = self.masks[(h, w)]
        inside = patch[inside_mask]
        outside = patch[outside_mask]
        if not inside.size or not outside.size:
            return None, {"rejected": "small_patch"}
        core = float(np.median(inside))
        ring = float(np.median(outside))
        contrast = core - ring
        # 准星/命中动画会遮住中心少量像素；用中位数偏移度量表面，小遮挡不误杀
        deviation = float(np.median(np.abs(inside.astype(np.float32) - core)))
        details = {"contrast": contrast, "core_std": float(inside.std()), "core_mad": deviation}
        if contrast < self.min_contrast or deviation > 12:
            return None, {**details, "rejected": "contrast_or_texture"}
        binary = (patch > (core + ring) / 2).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.pointPolygonTest(c, (cx, cy), False) < 0:
                continue
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            bx, by, bw, bh = cv2.boundingRect(c)
            circularity = 4 * math.pi * area / max(perimeter**2, 1)
            details.update(circularity=circularity, area=area, bbox=[x + bx, y + by, bw, bh])
            if (
                circularity < self.min_circularity
                or not 0.65 <= bw / bh <= 1.55
                or not 0.65 <= area / (math.pi * r * r) <= 1.3
                or bx == 0
                or by == 0
                or bx + bw >= w
                or by + bh >= h
            ):
                return None, {**details, "rejected": "geometry"}
            m = cv2.moments(c)
            return (x + m["m10"] / m["m00"], y + m["m01"] / m["m00"], area), details
        return None, {**details, "rejected": "no_center_component"}

    def _coarse(self, small, indices):
        candidates = []
        for index in indices:
            full, template, scale = self.bank[index]
            if min(small.shape[i] - template.shape[i] for i in (0, 1)) < 0:
                continue
            match = cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED)
            # 每个尺度只留若干峰值，绝不拿低分凑数
            for _ in range(self.max_targets + 1):
                _, score, _, (x, y) = cv2.minMaxLoc(match)
                if score < self.coarse_threshold:
                    break
                candidates.append((score, x * self.factor, y * self.factor, index))
                rh, rw = template.shape
                match[
                    max(0, y - rh // 2) : y + rh // 2 + 1, max(0, x - rw // 2) : x + rw // 2 + 1
                ] = -1
        return candidates

    def _refine_candidates(self, gray, candidates, cache):
        accepted = []
        details = []
        reused = 0
        for coarse_score, x, y, index in sorted(candidates, reverse=True)[: self.max_candidates]:
            full, _, scale = self.bank[index]
            h, w = full.shape
            pad = self.factor * 2
            if any(
                math.hypot(x + w / 2 - a.x, y + h / 2 - a.y) < max(16, math.sqrt(a.area / math.pi))
                for a in accepted
            ):
                continue
            key = (index, x, y)
            cached = cache.get(key)
            if cached is not None:
                point, row = cached
                reused += 1
            else:
                left, top = max(0, x - pad), max(0, y - pad)
                right, bottom = (min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad))
                if right - left < w or bottom - top < h:
                    continue
                match = cv2.matchTemplate(gray[top:bottom, left:right], full, cv2.TM_CCOEFF_NORMED)
                _, score, _, (ox, oy) = cv2.minMaxLoc(match)
                x, y = left + ox, top + oy
                row = {
                    "coarse_score": coarse_score,
                    "refine_score": score,
                    "xy": [x, y],
                    "scale": scale,
                }
                if score < self.refine_threshold:
                    point = None
                    row["rejected"] = "refine_score"
                else:
                    point, validation = self._validate(gray, x, y, w, h)
                    row.update(validation)
                cache[key] = (point, row)
            details.append(row)
            if point:
                px, py, area = point
                accepted.append(
                    Detection(float(px), float(py), float(area), float(row["refine_score"]))
                )
                if len(accepted) == self.max_targets:
                    break
        return accepted, details, reused

    def process(self, frame, frame_timestamp):
        start = time.perf_counter_ns()
        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        )
        small = cv2.resize(
            gray,
            (gray.shape[1] // self.factor, gray.shape[0] // self.factor),
            interpolation=cv2.INTER_AREA,
        )
        pre = time.perf_counter_ns()
        candidates = self._coarse(small, self.primary_indices)
        coarse = time.perf_counter_ns()
        coarse_ns = coarse - pre
        # 只缓存同一帧内的精配结果，绝不跨帧携带目标状态
        refine_cache = {}
        accepted, details, reused = self._refine_candidates(gray, candidates, refine_cache)
        fallback = bool(self.remaining_indices and len(accepted) < self.max_targets)
        if fallback:
            extra_start = time.perf_counter_ns()
            candidates += self._coarse(small, self.remaining_indices)
            coarse_ns += time.perf_counter_ns() - extra_start
            # fallback 后对全尺度候选重新排序，而不是把弱匹配直接缀在后面
            accepted, details, reused = self._refine_candidates(gray, candidates, refine_cache)
        refined = time.perf_counter_ns()
        detections = []
        for d in sorted(accepted, key=lambda d: d.confidence, reverse=True):
            if any(
                math.hypot(d.x - a.x, d.y - a.y) < max(16, math.sqrt(min(d.area, a.area) / math.pi))
                for a in detections
            ):
                continue
            detections.append(d)
            if len(detections) == self.max_targets:
                break
        end = time.perf_counter_ns()
        self.last_profile = {
            "preprocess_ms": (pre - start) / 1e6,
            "coarse_ms": coarse_ns / 1e6,
            "refine_validate_ms": (refined - pre - coarse_ns) / 1e6,
            "nms_ms": (end - refined) / 1e6,
            "total_ms": (end - start) / 1e6,
            "scale_policy": self.scale_policy,
            "scale_fallback": fallback,
            "refine_cache_hits": reused,
            "scales_searched": self.fallback_scales if fallback else self.primary_scales,
            "candidate_count": len(candidates),
            "candidates": details,
            "detection_count": len(detections),
        }
        return VisionResult(detections, (end - start) / 1e6, frame_timestamp)
