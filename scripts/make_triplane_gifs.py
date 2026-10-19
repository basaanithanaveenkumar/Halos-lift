"""Render the step-by-step triplane lifting GIFs used in the README.

python scripts/make_triplane_gifs.py                    # all techniques -> assets/triplane/
python scripts/make_triplane_gifs.py --only tiim bevformer --out /tmp/gifs
"""

from __future__ import annotations

import argparse
import os

from loguru import logger


def main() -> None:
    from lifting.triplane.animation import STORYBOARDS, render_all

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="assets/triplane", help="output directory")
    parser.add_argument(
        "--only", nargs="*", choices=STORYBOARDS.names(), help="subset of techniques"
    )
    args = parser.parse_args()
    for name, path in render_all(args.out, args.only).items():
        logger.info("{:12s} -> {}  ({:.0f} KB)", name, path, os.path.getsize(path) / 1024)


if __name__ == "__main__":
    main()
