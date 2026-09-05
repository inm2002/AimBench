"""The four supported detectors share the same capture, control and input path."""

from importlib import import_module

from .config import ASSETS, project_path, runtime_paths

DETECTORS = {
    "color": ("aimbench.vision.color_rule:ColorRuleDetector", {}),
    "template": (
        "aimbench.vision.template:TemplateDetector",
        {"template_path": str(ASSETS / "gridshot_target.png")},
    ),
    "cnn": (
        "aimbench.vision.heatmap_cnn_trt_u8:HeatmapCNNTRTU8Detector",
        {
            "engine_path": ".local/models/cnn.engine",
            "metadata_path": str(ASSETS / "cnn.json"),
            "max_targets": 3,
            "min_confidence": 0.5,
            "border": 1,
            "nms_kernel": 5,
            "min_separation_px": 16.0,
            "warmup_iterations": 100,
        },
    ),
    "yolo": (
        "aimbench.vision.yolo_trtyolo:TensorRTYOLODetector",
        {
            "engine_path": ".local/models/yolo.engine",
            "runtime_root": ".local/runtime/yolo",
            "runtime_bin": ".local/runtime/yolo/bin",
            "max_targets": 3,
            "min_confidence": 0.05,
            "source_width": 1280,
            "source_height": 720,
            "warmup_iterations": 30,
            "precision": "FP16",
            "variant": "640x384",
        },
    ),
}


def detector_spec(name, overrides=None):
    if name in DETECTORS:
        factory, defaults = DETECTORS[name]
    elif ":" in name:
        factory, defaults = name, {}
    else:
        raise ValueError(f"Unknown detector {name!r}; choose {', '.join(DETECTORS)}")
    parameters = {**defaults, **(runtime_paths() if name == "yolo" else {}), **(overrides or {})}
    # 相对路径统一解析到项目根，运行目录不影响资产定位
    for key, value in list(parameters.items()):
        if (
            key.endswith("_path")
            or key in {"runtime_root", "runtime_bin", "cuda_root", "tensorrt_root"}
        ) and value:
            parameters[key] = str(project_path(value))
    return factory, parameters


def create_detector(name, overrides=None):
    factory_name, parameters = detector_spec(name, overrides)
    module_name, attribute = factory_name.split(":", 1)
    factory = getattr(import_module(module_name), attribute)
    return factory(**parameters), {"factory": factory_name, "parameters": parameters}
