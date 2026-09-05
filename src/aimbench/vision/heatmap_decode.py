"""CPU heatmap decoding. No Torch/TRT imports; peaks denote distinct objects.

NMS ranks raw logits, collapses connected flat maxima, suppresses accepted peaks,
and checks separation again AFTER subpixel refinement. No forced top-k padding.
"""

import math
from functools import lru_cache

import cv2
import numpy as np

DEFAULT_REFINEMENT = "log_probability_quadratic"
DECODER_VERSION = "logit_nms_v6_local_log_probability"


@lru_cache(maxsize=16)
def _kernel(size):
    kernel = np.ones((size, size), np.uint8)
    kernel.flags.writeable = False
    return kernel


def _refine(logits, x, y, *, log_probability=False):
    h, w = logits.shape
    ix, iy = (int(round(x)), int(round(y)))
    if ix < 1 or iy < 1 or ix >= w - 1 or (iy >= h - 1):
        return (x, y, "border")
    p = logits[iy - 1 : iy + 2, ix - 1 : ix + 2].astype(np.float64, copy=False)
    if log_probability:
        p = -np.logaddexp(0.0, -p)
    gx, gy = ((p[1, 2] - p[1, 0]) * 0.5, (p[2, 1] - p[0, 1]) * 0.5)
    xx, yy = (p[1, 2] - 2 * p[1, 1] + p[1, 0], p[2, 1] - 2 * p[1, 1] + p[0, 1])
    xy = (p[2, 2] - p[2, 0] - p[0, 2] + p[0, 0]) * 0.25
    det = xx * yy - xy * xy
    scale = abs(xx) + abs(yy)
    # Hessian 负定才算真峰；平顶或鞍点退回原坐标
    if xx >= 0 or yy >= 0 or scale < 1e-14 or (det <= scale * scale * 1e-06):
        return (x, y, "flat_or_nonmaximum")
    dx, dy = (-(yy * gx - xy * gy) / det, -(-xy * gx + xx * gy) / det)
    if not math.isfinite(dx + dy):
        return (x, y, "nonfinite")
    return (ix + float(max(-1, min(1, dx))), iy + float(max(-1, min(1, dy))), "refined")


def decode_heatmap_cpu(
    logits_2d,
    k=3,
    nms_kernel=5,
    border=1,
    *,
    min_confidence=0.5,
    pixel_scale=(1.0, 1.0),
    min_separation_px=0.0,
    refinement=DEFAULT_REFINEMENT,
):
    if refinement not in ("raw_logit_quadratic", "log_probability_quadratic"):
        raise ValueError("Unknown subpixel refinement: " + str(refinement))
    if (
        type(k) is not int
        or k < 1
        or type(nms_kernel) is not int
        or (nms_kernel < 1)
        or (nms_kernel % 2 == 0)
    ):
        raise ValueError("k must be positive; nms_kernel must be positive and odd")
    if (
        isinstance(min_confidence, bool)
        or not 0 < min_confidence < 1
        or type(border) is not int
        or (border < 0)
        or isinstance(min_separation_px, bool)
        or (not math.isfinite(min_separation_px))
        or (min_separation_px < 0)
    ):
        raise ValueError("Require 0 < min_confidence < 1 and nonnegative border/separation")
    logits = np.asarray(logits_2d, dtype=np.float32)
    if logits.ndim != 2 or min(logits.shape) <= 2 * border:
        raise ValueError("Invalid heatmap shape/border")
    if not np.isfinite(logits).all():
        raise ValueError("Heatmap contains NaN/Inf; rejecting invalid model output")
    h, w = logits.shape
    sx, sy = pixel_scale
    if not math.isfinite(sx + sy) or sx <= 0 or sy <= 0:
        raise ValueError("pixel_scale must be finite and positive")
    cutoff = math.log(min_confidence) - math.log1p(-min_confidence)
    maxima = cv2.dilate(logits, _kernel(nms_kernel))
    mask = ((logits >= maxima) & (logits >= cutoff)).astype(np.uint8)
    if border:
        mask[:border] = 0
        mask[-border:] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
    # 孤立峰直接取点；相邻峰（平顶/饱和）按连通域归并，一个目标只出一个候选
    flat_indices = np.flatnonzero(mask)
    raw_candidates = int(flat_indices.size)
    candidates = []
    if raw_candidates <= 32:
        ys, xs = np.divmod(flat_indices, w)
        touching = any(
            (
                abs(int(xs[i]) - int(xs[j])) <= 1 and abs(int(ys[i]) - int(ys[j])) <= 1
                for i in range(raw_candidates)
                for j in range(i + 1, raw_candidates)
            )
        )
    else:
        touching = True
    if not touching:
        candidates = [(float(logits[y, x]), float(x), float(y), 1) for y, x in zip(ys, xs)]
    else:
        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, components):
            x0, y0, bw, bh, area = stats[label]
            if area == 1:
                candidates.append((float(logits[y0, x0]), float(x0), float(y0), 1))
                continue
            region_y, region_x = np.nonzero(labels[y0 : y0 + bh, x0 : x0 + bw] == label)
            region_y, region_x = (region_y + y0, region_x + x0)
            scores = logits[region_y, region_x]
            score = float(scores.max())
            top = scores == score
            x, y = (float(region_x[top].mean()), float(region_y[top].mean()))
            candidates.append((score, x, y, int(area)))
    candidates.sort(key=lambda p: (-p[0], p[2], p[1]))
    log_probability = refinement == "log_probability_quadratic"
    radius = nms_kernel // 2
    selected, accepted_grid = ([], [])
    for score, x, y, area in candidates:
        if any((abs(x - a) <= radius and abs(y - b) <= radius for a, b in accepted_grid)):
            continue
        rx, ry, refine_reason = (
            _refine(logits, x, y, log_probability=log_probability)
            if area == 1
            else (x, y, "plateau_center")
        )
        rx, ry = (float(max(0, min(w - 1, rx))), float(max(0, min(h - 1, ry))))
        if any(
            (
                math.hypot((rx - a) * sx, (ry - b) * sy) < min_separation_px
                or (abs(rx - a) <= radius and abs(ry - b) <= radius)
                for a, b, _ in selected
            )
        ):
            continue
        confidence = (
            1.0 / (1.0 + math.exp(-score))
            if score >= 0
            else math.exp(score) / (1.0 + math.exp(score))
        )
        selected.append((rx, ry, confidence))
        accepted_grid.append((x, y))
        if len(selected) == k:
            break
    return [(x / max(1, w - 1), y / max(1, h - 1), c) for x, y, c in selected]
