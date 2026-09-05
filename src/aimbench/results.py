"""Read, validate and compare the compact run summaries."""

import json
import math
from pathlib import Path

from .recording import write_json


def resolve_run(value):
    if value == "latest":
        candidates = sorted(Path("runs").glob("*/summary.json"))
        if not candidates:
            raise ValueError("No recorded runs in runs/")
        return candidates[-1]
    path = Path(value)
    return path / "summary.json" if path.is_dir() else path


def save_result(path, score, accuracy, shots, targets=None):
    path = resolve_run(str(path))
    if (
        not math.isfinite(score)
        or score < 0
        or not math.isfinite(accuracy)
        or not 0 <= accuracy <= 100
    ):
        raise ValueError("Score must be nonnegative and accuracy must be between 0 and 100")
    if (
        type(shots) is not int
        or shots < 0
        or targets is not None
        and (type(targets) is not int or targets < 0)
    ):
        raise ValueError("Shots and targets must be nonnegative integers")
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["game_result"] = {
        "score": score,
        "accuracy": accuracy,
        "shots": shots,
        "total_targets": targets,
        "source": "manual_result_screen",
    }
    write_json(path, summary)
    return summary["game_result"]


def prompt_result(path):
    score = input("结算分数（回车跳过）：").strip()
    if not score:
        return
    accuracy = input("准确率（%）：").strip()
    shots = input("总射击次数：").strip()
    targets = input("总目标数（可留空）：").strip()
    save_result(path, float(score), float(accuracy), int(shots), int(targets) if targets else None)


def comparison_rows(directory):
    rows = []
    for path in sorted(Path(directory).glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("game_result"):
            continue
        common = dict(summary["config"])
        for key in ("algorithm", "detector_params", "label"):
            common.pop(key, None)
        common["run"] = {
            k: v
            for k, v in common["run"].items()
            if k not in {"output_dir", "prompt_result", "save_result_image"}
        }
        common["capture"] = summary.get("components", {}).get("capture")
        common["environment"] = summary.get("environment")
        common["source_sha256"] = summary.get("source_sha256")
        rows.append(
            {
                "run": summary["run_id"],
                "algorithm": summary["algorithm"],
                **summary["game_result"],
                "eligible": summary["eligible_for_comparison"],
                "conditions": json.dumps(common, sort_keys=True),
            }
        )
    return rows
