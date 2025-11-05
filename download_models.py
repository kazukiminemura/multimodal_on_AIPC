"""CLI helper to download model snapshots."""

from __future__ import annotations

import argparse
import logging

from app.download_models import run as run_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required model snapshots.")
    parser.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Skip downloading when models already exist.",
    )
    parser.set_defaults(force=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()
    statuses = run_download(force=args.force)
    for name, status in statuses.items():
        logging.info("%s -> %s", name, status)


if __name__ == "__main__":
    main()

