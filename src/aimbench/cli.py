"""One entry point; running without arguments starts the configured algorithm."""

import argparse
import os
import sys
from pathlib import Path

from .config import load_config
from .registry import DETECTORS


def parser():
    root = argparse.ArgumentParser(
        prog="aimbench", description="Aimlabs vision comparison with a shared control pipeline"
    )
    root.add_argument("--version", action="version", version="AimBench 1.0.0")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行 configs/default.json 中的方案（默认）")
    run.add_argument("--config", default="configs/default.json")
    run.add_argument("--algorithm", help="覆盖检测器名称，也支持 module:factory")
    run.add_argument("--label")
    run.add_argument("--dry-run", action="store_true", help="检测但不发送鼠标输入")
    run.add_argument("--no-prompt", action="store_true", help="结束后不询问游戏成绩")
    check = commands.add_parser("check", help="检查当前解释器、依赖与模型路径，不操作游戏")
    check.add_argument("--algorithm", choices=DETECTORS, default="cnn")
    result = commands.add_parser("result", help="补填已有运行的结算成绩")
    result.add_argument("path", nargs="?", default="latest")
    result.add_argument("--score", type=float, required=True)
    result.add_argument("--accuracy", type=float, required=True)
    result.add_argument("--shots", type=int, required=True)
    result.add_argument("--targets", type=int)
    compare = commands.add_parser("compare", help="查看已有结算成绩")
    compare.add_argument("directory", nargs="?", default="runs")
    return root


def main(argv=None, project_root=None):
    if project_root is not None:
        os.chdir(project_root)
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv or ["run"])
    try:
        if args.command == "run":
            from .results import prompt_result
            from .runner import run

            config_path = Path(args.config)
            config = load_config(config_path)
            if args.algorithm:
                config.algorithm = args.algorithm
            if args.label is not None:
                config.label = args.label
            summary, directory = run(config, dry_run=args.dry_run)
            if (
                config.run.prompt_result
                and not args.no_prompt
                and summary["physical_input"]
                and summary["end_reason"] in {"normal_game_end", "hard_stop"}
            ):
                try:
                    prompt_result(directory)
                except (ValueError, EOFError, KeyboardInterrupt) as exc:
                    print(f"成绩未保存，可稍后用 result 补填：{exc}")
            return (
                0
                if summary["end_reason"]
                in {"normal_game_end", "hard_stop", "interrupted", "user_stop_key"}
                else 1
            )
        if args.command == "check":
            from .checks import check_environment

            rows = check_environment(args.algorithm)
            print(f"解释器：{sys.executable}")
            for name, detail, ok in rows:
                print(f"{'OK' if ok else 'MISSING':7} {name}: {detail}")
            return 0 if all(row[2] for row in rows) else 1
        if args.command == "result":
            from .results import save_result

            print(save_result(args.path, args.score, args.accuracy, args.shots, args.targets))
            return 0
        from .results import comparison_rows

        rows = comparison_rows(args.directory)
        print("算法\t分数\t准确率\t射击数\t完整有效\t运行")
        for row in rows:
            print(
                f"{row['algorithm']}\t{row['score']:g}\t{row['accuracy']:g}%\t{row['shots']}\t{row['eligible']}\t{row['run']}"
            )
        if len({row["conditions"] for row in rows if row["eligible"]}) > 1:
            print("包含不同公共配置或采集条件，按局展示，不合并排名。")
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"AimBench: {exc}", file=sys.stderr)
        return 1
