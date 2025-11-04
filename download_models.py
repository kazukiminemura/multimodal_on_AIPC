from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    sys.path.append(str(project_root / "src"))

    from app.download_models import main as _main

    _main()


if __name__ == "__main__":
    main()
