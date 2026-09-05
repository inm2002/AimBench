"""Passive, bounded HUD observations shared by every detector. Never sends input."""

import math
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
POPCOUNT = np.array([i.bit_count() for i in range(256)], np.uint8)  # 按字节预查 popcount，整批算汉明距离


class HudReader:
    fields = {
        "score": (400, 43, 553, 62),
        "time": (573, 43, 659, 62),
        "accuracy": (720, 43, 820, 62),
    }

    def __init__(self, bank_path=None):
        with np.load(bank_path or ASSETS / "hud_digits.npz", allow_pickle=False) as bank:
            glyphs = bank["glyphs"].astype(np.uint8)
            digits = bank["digits"].copy()
        packed = np.packbits(glyphs.reshape(len(glyphs), -1), axis=1)
        _, indices = np.unique(
            np.column_stack((packed, digits.astype(np.uint8))), axis=0, return_index=True
        )
        indices = np.sort(indices)
        self.glyphs = glyphs[indices].astype(np.float32)
        self.packed = packed[indices]
        self.digits = digits[indices]
        self.cache = {}
        self.field_cache = {}

    @staticmethod
    def crop(frame):
        h, w = frame.shape[:2]
        crop = frame[
            round(20 * h / 720) : round(72 * h / 720), round(396 * w / 1280) : round(884 * w / 1280)
        ]
        if crop.shape[:2] != (52, 488):
            crop = cv2.resize(crop, (488, 52), interpolation=cv2.INTER_AREA)
        return crop

    @staticmethod
    def mask(frame):
        crop = HudReader.crop(frame)
        lower = (210, 210, 210, 0) if crop.shape[2] == 4 else (210, 210, 210)
        upper = (255, 255, 255, 255) if crop.shape[2] == 4 else (255, 255, 255)
        return cv2.inRange(crop, lower, upper) // 255

    def recognize(self, image, min_height=9, min_area=12):
        _, _, stats, _ = cv2.connectedComponentsWithStats(image, 8)
        boxes = sorted(
            [tuple(map(int, s[:4])) for s in stats[1:] if s[4] >= min_area and s[3] >= min_height]
        )
        glyphs = []
        for x, y, w, h in boxes:
            glyph = cv2.resize(
                image[y : y + h, x : x + w], (16, 20), interpolation=cv2.INTER_NEAREST
            )
            packed = np.packbits(glyph).tobytes()
            match = self.cache.get(packed)
            if match is None:
                distances = (
                    POPCOUNT[np.bitwise_xor(self.packed, np.frombuffer(packed, np.uint8))].sum(
                        axis=1
                    )
                    / 320.0
                )
                idx = int(distances.argmin())
                digit = int(self.digits[idx])
                error = float(distances[idx])
                other = distances[self.digits != digit]
                margin = float(other.min() - error) if len(other) else 1.0
                match = {"digit": digit, "error": error, "margin": margin}
                if len(self.cache) < 4096:
                    self.cache[packed] = match
            glyphs.append({**match, "box": [x, y, w, h]})
        text = "".join((str(g["digit"]) for g in glyphs))
        return {
            "text": text,
            "glyphs": glyphs,
            "glyph_count": len(glyphs),
            "error": max((g["error"] for g in glyphs), default=None),
            "min_margin": min((g["margin"] for g in glyphs), default=None),
        }

    def read_details(self, mask, field):
        x, y, r, b = self.fields[field]
        image = mask[y - 20 : b - 20, x - 396 : r - 396]
        key = image.tobytes()
        cached = self.field_cache.get(field)
        if cached and cached[0] == key:
            return cached[1]
        result = self.recognize(image)
        text = result["text"]
        candidate = int(text) if text else None
        if field == "time":
            candidate = (
                int(text[:2]) * 60 + int(text[2:])
                if len(text) == 4 and int(text[2:]) < 60
                else None
            )
            if candidate is not None and candidate > 65:
                candidate = None
        elif field == "accuracy":
            candidate = None
        result.update(
            best_value=candidate,
            value=candidate if result["error"] is not None and result["error"] <= 0.25 else None,
        )
        self.field_cache[field] = (key, result)
        return result

    def read(self, mask, field):
        result = self.read_details(mask, field)
        return (result["value"], result["error"])


class TimerFilter:
    """A confident clock can bridge small glyph damage, never arbitrary menus."""

    def __init__(self, hold_ms=75):
        self.hold_ns = int(hold_ms * 1000000.0)
        self.value = None
        self.changed_ns = None
        self.trusted_ns = None

    def update(self, detail, seen_ns, score_readable):
        candidate = detail.get("best_value")
        raw = detail.get("value")
        elapsed = (
            (seen_ns - self.changed_ns) / 1000000000.0 if self.changed_ns is not None else None
        )
        plausible = candidate is not None and (
            self.value is None or 0 <= self.value - candidate <= math.ceil(max(0, elapsed)) + 1
        )
        uncertain = [g for g in detail.get("glyphs", []) if g["error"] > 0.25]
        rescued = (
            self.value is not None
            and plausible
            and score_readable
            and (detail.get("glyph_count") == 4)
            and bool(uncertain)
            and all((g["error"] <= 0.3 and g["margin"] >= 0.05 for g in uncertain))
        )
        if plausible and (raw is not None or rescued):
            if candidate != self.value:
                self.value = candidate
                self.changed_ns = seen_ns
            self.trusted_ns = seen_ns
            return (self.value, "temporal_candidate" if rescued else "direct")
        held = (
            self.value is not None
            and detail.get("glyph_count") == 4
            and score_readable
            and (self.trusted_ns is not None)
            and (0 <= seen_ns - self.trusted_ns <= self.hold_ns)
        )
        return (self.value, "held") if held else (None, "unrecognized")


class GameObserver:
    def __init__(self, timer_hold_ms=75):
        self.reader = HudReader()
        self.pause = cv2.imdecode(
            np.fromfile(ASSETS / "gridshot_pause.png", np.uint8), cv2.IMREAD_GRAYSCALE
        )
        self.last_hud = None
        self.last_scene = None
        self.last_pixels_ns = None
        self.last_pause_key = None
        self.pause_score = 0
        self.last_values = {}
        self.last_details = {}
        self.timer = TimerFilter(timer_hold_ms)
        self.last_time = None
        self.last_time_change_ns = None
        self.last_score = None
        self.score_events = 0
        self.first_active_ns = None
        self.state_counts = {}
        self.pause_seen = False
        self.last = {}
        self.timer_source_counts = {}
        self.confirmed_score = None
        self.score_candidate = None
        self.score_candidate_count = 0

    def startup_read(self, frame):
        mask = self.reader.mask(frame)
        return {
            field: self.reader.read_details(mask, field)["value"] for field in ("score", "time")
        }

    def observe(self, frame, seq, seen_ns):
        start = time.perf_counter_ns()
        h, w = frame.shape[:2]
        mask = self.reader.mask(frame)
        packed = np.packbits(mask).tobytes()
        changed = packed != self.last_hud
        if changed:
            self.last_details = {
                field: self.reader.read_details(mask, field) for field in ("score", "time")
            }
            self.last_values = {}
            for field, detail in self.last_details.items():
                self.last_values[field] = detail["value"]
                self.last_values[field + "_glyph_error"] = detail["error"]
            self.last_hud = packed
        # 抽稀采样后取 CRC，开销固定，只用于判断画面有没有动
        scene = np.ascontiguousarray(frame[round(100 * h / 720) : h : 12, ::12, :3])
        signature = zlib.crc32(scene)
        pixel_changed = signature != self.last_scene
        if pixel_changed:
            self.last_pixels_ns = seen_ns
        self.last_scene = signature
        crop = frame[
            round(432 * h / 720) : round(459 * h / 720),
            round(606 * w / 1280) : round(675 * w / 1280),
        ]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY if crop.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        if crop.shape != self.pause.shape:
            crop = cv2.resize(crop, self.pause.shape[::-1])
        pause_key = crop.tobytes()
        # 暂停区域没变就沿用上次的匹配分
        if pause_key != self.last_pause_key:
            self.pause_score = float(
                cv2.matchTemplate(crop, self.pause, cv2.TM_CCOEFF_NORMED)[0, 0]
            )
            self.last_pause_key = pause_key
        paused = self.pause_score >= 0.85
        self.pause_seen |= paused
        seconds, timer_source = self.timer.update(
            self.last_details["time"], seen_ns, self.last_values.get("score") is not None
        )
        self.last_time = self.timer.value
        self.last_time_change_ns = self.timer.changed_ns
        timer_stalled = (
            self.last_time_change_ns is not None and seen_ns - self.last_time_change_ns > 2000000000
        )
        state = (
            "paused"
            if paused
            else "timer_stalled"
            if timer_stalled
            else "hud_unrecognized"
            if seconds is None
            else "active"
        )
        if state == "active" and self.first_active_ns is None:
            self.first_active_ns = seen_ns
        self.state_counts[state] = self.state_counts.get(state, 0) + 1
        self.timer_source_counts[timer_source] = self.timer_source_counts.get(timer_source, 0) + 1
        score = self.last_values.get("score")
        delta = None
        if state == "active" and score is not None:
            self.score_candidate_count = (
                self.score_candidate_count + 1 if score == self.score_candidate else 1
            )
            self.score_candidate = score
            if self.score_candidate_count >= 2 and (
                self.confirmed_score is None or 0 <= score - self.confirmed_score <= 5000
            ):
                self.confirmed_score = score
        else:
            self.score_candidate = None
            self.score_candidate_count = 0
        if score is not None and score != self.last_score:
            if self.last_score is not None:
                delta = score - self.last_score
                self.score_events += 1
            self.last_score = score
        previous = self.last.get("state")
        self.last = {
            "enabled": True,
            "state": state,
            "previous_state": previous,
            "state_changed": previous is not None and previous != state,
            "pause_match": self.pause_score,
            "phase": "ending" if seconds == 0 else "playing" if state == "active" else state,
            **self.last_values,
            "time_raw": self.last_values.get("time"),
            "time": seconds,
            "timer_source": timer_source,
            "score_confirmed": self.confirmed_score,
            "timer_candidate": self.last_details["time"].get("best_value"),
            "timer_glyph_count": self.last_details["time"]["glyph_count"],
            "timer_min_margin": self.last_details["time"].get("min_margin"),
            "timer_glyphs": self.last_details["time"]["glyphs"] if changed else None,
            "hud_changed": changed,
            "score_delta_observed": delta,
            "scene_signature": signature,
            "scene_pixels_changed": pixel_changed,
            "scene_unchanged_ms": (seen_ns - self.last_pixels_ns) / 1000000.0,
            "observer_ms": (time.perf_counter_ns() - start) / 1000000.0,
        }
        return self.last

    def close(self):
        return {
            "state_counts": self.state_counts,
            "pause_seen": self.pause_seen,
            "confirmed_score": self.confirmed_score,
        }
