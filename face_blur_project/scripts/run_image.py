"""
run_image.py — Convenience Script for Image Processing
=======================================================

Process all images in ``datasets/images/`` (or a custom path) with
a single command.

Usage:
    python scripts/run_image.py
    python scripts/run_image.py --input path/to/image.jpg
    python scripts/run_image.py --mode pixelate --debug
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import cfg
from app.utils import setup_logging, ensure_dirs
from app.image_processor import ImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Run face blur on images.")
    parser.add_argument("--input", type=str, default=None,
                        help="Single image path or directory (default: datasets/images/)")
    parser.add_argument("--mode", choices=["gaussian", "pixelate"], default="gaussian")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg.blur_mode = args.mode
    cfg.debug = args.debug
    logger = setup_logging()
    ensure_dirs()

    processor = ImageProcessor(cfg)

    input_path = Path(args.input) if args.input else cfg.datasets_images_dir

    if input_path.is_file():
        processor.process_image(str(input_path))
    elif input_path.is_dir():
        processor.process_directory(str(input_path))
    else:
        logger.error("Input not found: %s", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
