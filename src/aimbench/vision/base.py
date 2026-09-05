from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Detection:
    x: float
    y: float
    area: float = 0.0
    confidence: float = 1.0


@dataclass
class VisionResult:
    detections: List[Detection]
    process_ms: float
    frame_timestamp: float


class VisionBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def process(self, frame: np.ndarray, frame_timestamp: float) -> VisionResult:
        pass
