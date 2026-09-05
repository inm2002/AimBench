import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

from .base import Detection, VisionBackend, VisionResult
from .heatmap_decode import DECODER_VERSION, DEFAULT_REFINEMENT, decode_heatmap_cpu


def interpolation_from_name(name):
    name = str(name).lower()
    if name == "linear":
        return cv2.INTER_LINEAR
    if name == "area":
        return cv2.INTER_AREA
    raise ValueError(f"Unsupported resize mode: {name}")


class HeatmapCNNTRTU8Detector(VisionBackend):
    """TensorRT heatmap inference with reused pinned buffers and local peak refinement."""

    def __init__(
        self,
        engine_path,
        metadata_path,
        max_targets=3,
        min_confidence=0.5,
        border=1,
        nms_kernel=5,
        min_separation_px=16.0,
        warmup_iterations=100,
        refinement=DEFAULT_REFINEMENT,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")
        self.device = torch.device("cuda")
        self.engine_path = Path(engine_path)
        self.metadata_path = Path(metadata_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(self.engine_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)
        self.max_targets = int(max_targets)
        self.min_confidence = float(min_confidence)
        self.border = int(border)
        self.nms_kernel = int(nms_kernel)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.input_width = int(metadata["input_width"])
        self.input_height = int(metadata["input_height"])
        self.heatmap_width = int(metadata["heatmap_width"])
        self.heatmap_height = int(metadata["heatmap_height"])
        self.resize_mode = metadata.get("resize", "linear")
        self.interpolation = interpolation_from_name(self.resize_mode)
        self.logger = trt.Logger(trt.Logger.WARNING)
        if hasattr(trt, "init_libnvinfer_plugins"):
            trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        engine_bytes = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Could not deserialize TensorRT engine.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TRT execution context.")
        input_names = []
        output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Expected exactly one input and one output.\nInputs: {input_names}\nOutputs: {output_names}"
            )
        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.input_shape = tuple(
            (int(value) for value in self.engine.get_tensor_shape(self.input_name))
        )
        self.output_shape = tuple(
            (int(value) for value in self.engine.get_tensor_shape(self.output_name))
        )
        expected_input_shape = (1, self.input_height, self.input_width, 3)
        expected_output_shape = (1, 1, self.heatmap_height, self.heatmap_width)
        if self.input_shape != expected_input_shape:
            raise RuntimeError(
                f"\nTRT UINT8 input shape mismatch.\nEngine:   {self.input_shape}\nExpected: {expected_input_shape}"
            )
        if self.output_shape != expected_output_shape:
            raise RuntimeError(
                f"\nTRT output shape mismatch.\nEngine:   {self.output_shape}\nExpected: {expected_output_shape}"
            )
        input_np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        output_np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        if input_np_dtype != np.dtype(np.uint8):
            raise RuntimeError(f"Expected TRT UINT8 input, got {input_np_dtype}")
        if output_np_dtype != np.dtype(np.float16):
            raise RuntimeError(f"Expected TRT FP16 output, got {output_np_dtype}")
        self.pinned_bgra = torch.empty(
            (self.input_height, self.input_width, 4), dtype=torch.uint8, pin_memory=True
        )
        self.pinned_rgb = torch.empty(
            (self.input_height, self.input_width, 3), dtype=torch.uint8, pin_memory=True
        )
        self.pinned_bgra_np = self.pinned_bgra.numpy()
        self.pinned_rgb_np = self.pinned_rgb.numpy()
        self.input_tensor = torch.empty(self.input_shape, dtype=torch.uint8, device=self.device)
        self.output_tensor = torch.empty(self.output_shape, dtype=torch.float16, device=self.device)
        self.pinned_output = torch.empty(self.output_shape, dtype=torch.float16, pin_memory=True)
        self.pinned_output_np = self.pinned_output.numpy()
        if not self.context.set_tensor_address(self.input_name, int(self.input_tensor.data_ptr())):
            raise RuntimeError("Failed to bind TRT input.")
        if not self.context.set_tensor_address(
            self.output_name, int(self.output_tensor.data_ptr())
        ):
            raise RuntimeError("Failed to bind TRT output.")
        self.stream = torch.cuda.Stream(device=self.device)
        self.min_separation_px = float(min_separation_px)
        self.refinement = refinement
        self._warmup(warmup_iterations)

    @property
    def name(self):
        return f"Tiny Heatmap CNN TRT-U8 {self.input_width}x{self.input_height} (UINT8 NHWC input, FP16 CNN, LINEAR, CPU DARK)"

    def _warmup(self, iterations):
        with torch.cuda.stream(self.stream):
            self.input_tensor.zero_()
            for _ in range(int(iterations)):
                if not self.context.execute_async_v3(int(self.stream.cuda_stream)):
                    raise RuntimeError("TensorRT warmup failed")
        self.stream.synchronize()

    def _prepare_pinned_rgb(self, frame):
        """
        Resize directly into persistent pinned BGRA memory,
        then color-convert directly into persistent pinned
        RGB memory.
        No per-frame small RGB allocation.
        """
        cv2.resize(
            frame,
            (self.input_width, self.input_height),
            dst=self.pinned_bgra_np,
            interpolation=self.interpolation,
        )
        cv2.cvtColor(self.pinned_bgra_np, cv2.COLOR_BGRA2RGB, dst=self.pinned_rgb_np)

    def process(self, frame, frame_timestamp):
        start = time.perf_counter_ns()
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError("CNN expects a uint8 BGRA image")
        self._prepare_pinned_rgb(frame)
        with torch.cuda.stream(self.stream):
            self.input_tensor[0].copy_(self.pinned_rgb, non_blocking=True)
            if not self.context.execute_async_v3(int(self.stream.cuda_stream)):
                raise RuntimeError("TensorRT inference failed")
            self.pinned_output.copy_(self.output_tensor, non_blocking=True)
        self.stream.synchronize()
        logits = self.pinned_output_np[0, 0].astype(np.float32)
        height, width = frame.shape[:2]
        points = decode_heatmap_cpu(
            logits,
            k=self.max_targets,
            nms_kernel=self.nms_kernel,
            border=self.border,
            min_confidence=self.min_confidence,
            min_separation_px=self.min_separation_px,
            pixel_scale=(width / max(1, logits.shape[1] - 1), height / max(1, logits.shape[0] - 1)),
            refinement=self.refinement,
        )
        detections = [
            Detection(x * width, y * height, confidence=confidence) for x, y, confidence in points
        ]
        return VisionResult(
            detections, (time.perf_counter_ns() - start) / 1000000.0, frame_timestamp
        )

    def metadata(self):
        return {
            "decoder_version": DECODER_VERSION,
            "decoder_refinement": self.refinement,
            "confidence_semantics": "sigmoid peak score; not calibrated probability",
        }
