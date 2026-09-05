"""Concise status output shared by the CLI and runtime."""

import sys


class Console:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout

    def line(self, text):
        print(text, file=self.stream, flush=True)

    def start(self, config, directory):
        self.line(f"AimBench · {config.algorithm} · {config.label or 'single run'}")
        self.line(f"输出：{directory}")
        self.line(f"请在 {config.run.start_delay:g} 秒内切回游戏准备界面。")

    def summary(self, summary, directory):
        self.line(
            f"结束：{summary['end_reason']} · {summary['frames']} 帧 · {summary['elapsed_s']:.2f} 秒"
        )
        vision = summary["latency"]["vision"]
        if vision["count"]:
            self.line(f"视觉：平均 {vision['mean_ms']:.3f} ms · P95 {vision['p95_ms']:.3f} ms")
        self.line(f"结果：{directory / 'summary.json'}")
