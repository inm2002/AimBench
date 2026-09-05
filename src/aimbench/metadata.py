"""Small provenance records collected outside the timed loop."""

import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path

from .recording import plain


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def environment():
    packages = {}
    for name in ("numpy", "opencv-python", "dxcam", "pywin32", "torch", "tensorrt"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    if packages["tensorrt"] is None:
        try:
            packages["tensorrt"] = metadata.version("tensorrt-cu12")
        except metadata.PackageNotFoundError:
            pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


def source_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def component_metadata(component, spec):
    custom = getattr(component, "metadata", None)
    result = {**spec, "runtime": plain(custom()) if callable(custom) else {}}
    result["assets"] = {
        key: {"file": Path(value).name, "sha256": sha256(value)}
        for key, value in spec.get("parameters", {}).items()
        if key.endswith("_path") and value and Path(value).is_file()
    }
    return result
