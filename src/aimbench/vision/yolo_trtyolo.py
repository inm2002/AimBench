"""YOLO26n inference through the TensorRT-YOLO native binding."""

import importlib.util
import os
import time
from pathlib import Path

import cv2
import numpy as np

from ..metadata import sha256
from .base import Detection, VisionBackend, VisionResult

# add_dll_directory 的句柄要保住引用，目录被回收后 DLL 就找不到了
_DLL_HANDLES = []
_RUNTIME_MODULE = None


def _load_runtime(runtime_root, runtime_bin, tensorrt_root, cuda_root):
    global _RUNTIME_MODULE
    if _RUNTIME_MODULE is not None:
        return _RUNTIME_MODULE
    for directory in (runtime_bin, tensorrt_root / "lib", cuda_root / "bin"):
        if not directory.is_dir():
            raise FileNotFoundError(f"TensorRT-YOLO DLL directory missing: {directory}")
        _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    candidates = list((runtime_root / "libs").glob("py_trtyolo*.pyd"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one Python binding in {runtime_root / 'libs'}")
    spec = importlib.util.spec_from_file_location("py_trtyolo", candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module.result.DetectRes, "xyxy_float"):
        raise RuntimeError(
            "TensorRT-YOLO binding requires the xyxy_float extension; see docs/development.md"
        )
    _RUNTIME_MODULE = module
    return module


class TensorRTYOLODetector(VisionBackend):
    """One-class detection with persistent BGRA to BGR storage."""

    def __init__(
        self,
        engine_path,
        runtime_root,
        runtime_bin,
        tensorrt_root,
        cuda_root,
        precision="FP16",
        variant="640x384",
        max_targets=3,
        min_confidence=0.05,
        source_width=1280,
        source_height=720,
        warmup_iterations=30,
    ):
        self.engine_path = Path(engine_path).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.runtime_bin = Path(runtime_bin).resolve()
        self.tensorrt_root = Path(tensorrt_root).resolve()
        self.cuda_root = Path(cuda_root).resolve()
        self.precision, self.variant = precision, variant
        self.max_targets = int(max_targets)
        self.min_confidence = float(min_confidence)
        self.source_width, self.source_height = int(source_width), int(source_height)
        if not self.engine_path.is_file():
            raise FileNotFoundError(self.engine_path)
        runtime = _load_runtime(
            self.runtime_root, self.runtime_bin, self.tensorrt_root, self.cuda_root
        )
        option = runtime.option.InferOption()
        option.set_device_id(0)
        # 引擎按 RGB 训练，predict 收到的是 BGR，交给 swap_rb 翻转
        option.enable_swap_rb()
        # 6.4 底层 binding 收的是源图 H,W，不是网络输入尺寸
        option.set_input_dimensions(self.source_height, self.source_width)
        self.model = runtime.model.DetectModel(str(self.engine_path), option)
        self._bgr = np.empty((self.source_height, self.source_width, 3), np.uint8)
        for _ in range(int(warmup_iterations)):
            self.model.predict(np.zeros_like(self._bgr))

    @property
    def name(self):
        return f"YOLO26n {self.variant} {self.precision}"

    def process(self, frame, frame_timestamp):
        start = time.perf_counter_ns()
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError("YOLO expects a uint8 BGRA image")
        if frame.shape[:2] != (self.source_height, self.source_width):
            raise ValueError(f"YOLO expects source size {self.source_width}x{self.source_height}")
        cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR, dst=self._bgr)
        result = self.model.predict(self._bgr)
        detections = []
        for box, confidence, class_id in zip(result.xyxy_float, result.confidence, result.class_id):
            if int(class_id) != 0 or float(confidence) < self.min_confidence:
                continue
            x1, y1, x2, y2 = map(float, box)
            detections.append(
                Detection(
                    x=(x1 + x2) * 0.5,
                    y=(y1 + y2) * 0.5,
                    area=max(0.0, x2 - x1) * max(0.0, y2 - y1),
                    confidence=float(confidence),
                )
            )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return VisionResult(
            detections[: self.max_targets], (time.perf_counter_ns() - start) / 1e6, frame_timestamp
        )

    def metadata(self):
        binding = next((self.runtime_root / "libs").glob("py_trtyolo*.pyd"))
        native = [
            binding,
            self.runtime_bin / "trtyolo.dll",
            self.runtime_bin / "custom_plugins.dll",
        ]
        return {
            "architecture": "YOLO26n",
            "runtime": "TensorRT-YOLO 6.4.0",
            "precision": self.precision,
            "network_size": self.variant,
            "source_size": [self.source_width, self.source_height],
            "native_files": {path.name: sha256(path) for path in native},
            "box_coordinates": "xyxy_float",
            "native_profiling": False,
        }
