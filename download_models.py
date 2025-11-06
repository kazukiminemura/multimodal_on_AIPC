from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.config import Settings
from app.services import StableDiffusionService

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("download-models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Stable Diffusion model snapshot into the local cache directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a re-download even if the snapshot already exists.",
    )
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Skip download when the snapshot already exists (default).",
    )
    parser.set_defaults(force=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    settings.ensure_directories()

    cache_dir: Path = settings.models_cache_dir / "stable-diffusion"
    service = StableDiffusionService(settings=settings)

    if not args.force and service.models_cached():
        logger.info("Snapshot already present under %s", cache_dir)
        return

    if args.force and cache_dir.exists():
        logger.info("Removing previous snapshot at %s", cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)

    logger.info(
        "Downloading snapshot %s into %s",
        settings.stable_diffusion_repo_id,
        cache_dir,
    )
    service.ensure_model_snapshot()
    logger.info("Done.")


if __name__ == "__main__":
    main()
