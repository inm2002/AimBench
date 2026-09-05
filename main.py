"""IDE entry point. Select your existing Python interpreter and run this file."""

import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "src"))
    from aimbench.cli import main

    raise SystemExit(main(project_root=project_root))
