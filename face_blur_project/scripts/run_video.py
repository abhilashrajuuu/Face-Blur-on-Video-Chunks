"""
run_video.py — Convenience Script for Video Processing
=======================================================

Process all videos in ``datasets/videos/`` (or a custom path) with
a single command.

Usage:
    python scripts/run_video.py
    python scripts/run_video.py --input path/to/video.mp4
    python scripts/run_video.py --mode pixelate --debug
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import cfg
from app.utils import setup_logging, ensure_dirs
from app.video_processor import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Run face blur on videos.")
    parser.add_argument("--input", type=str, default=None,
                        help="Single video path or directory (default: datasets/videos/)")
    parser.add_argument("--mode", choices=["gaussian", "pixelate"], default="gaussian")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg.blur_mode = args.mode
    cfg.debug = args.debug
    logger = setup_logging()
    ensure_dirs()

    processor = VideoProcessor(cfg)

    input_path = Path(args.input) if args.input else cfg.datasets_videos_dir

    if input_path.is_file():
        processor.process_video(str(input_path))
    elif input_path.is_dir():
        processor.process_directory(str(input_path))
    else:
        logger.error("Input not found: %s", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
