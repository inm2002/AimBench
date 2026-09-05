"""Read-only dependency checks. This module never captures frames or sends input."""

import os
import sys
from importlib.util import find_spec
from pathlib import Path

from .registry import detector_spec


def check_environment(algorithm):
    rows = [
        ("Python", sys.version.split()[0], sys.version_info >= (3, 12)),
        ("Windows", sys.platform, os.name == "nt"),
    ]
    packages = ["numpy", "cv2", "dxcam", "win32gui"]
    if algorithm == "cnn":
        packages.extend(["torch", "tensorrt"])
    for package in packages:
        found = find_spec(package) is not None
        rows.append((package, "available" if found else "missing", found))
    _, parameters = detector_spec(algorithm)
    for key in (
        "engine_path",
        "metadata_path",
        "template_path",
        "runtime_root",
        "runtime_bin",
        "tensorrt_root",
        "cuda_root",
    ):
        if key in parameters:
            value = parameters[key]
            rows.append((key, value or "not configured", bool(value) and Path(value).exists()))
    if algorithm == "yolo":
        binding = list(Path(parameters["runtime_root"]).joinpath("libs").glob("py_trtyolo*.pyd"))
        rows.append(("YOLO Python ABI", "requires Python 3.12", sys.version_info[:2] == (3, 12)))
        rows.append(
            (
                "YOLO binding",
                str(binding[0]) if len(binding) == 1 else "missing or ambiguous",
                len(binding) == 1,
            )
        )
    return rows
