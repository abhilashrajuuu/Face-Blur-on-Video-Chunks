"""
main.py — CLI Entry Point for Face Blur Pipeline
==================================================

Supports the following commands:

    # Process a single video
    python main.py --video sample_inputs/sample_video.mp4

    # Process a single image
    python main.py --image sample_inputs/sample_image.jpg

    # Batch-process a directory of images
    python main.py --input-dir datasets/images/

    # Batch-process a directory of videos
    python main.py --input-dir datasets/videos/ --type video

    # Use pixelation instead of Gaussian blur
    python main.py --video sample_inputs/sample_video.mp4 --mode pixelate

    # Enable debug output (detection boxes, tracker IDs)
    python main.py --video sample_inputs/sample_video.mp4 --debug

    # Set custom confidence threshold
    python main.py --image sample_inputs/sample_image.jpg --confidence 0.5

All outputs are saved to  outputs/blurred_videos/  or  outputs/blurred_images/.
Debug visualisations go to  outputs/debug/.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'app' is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: app.* imports are deferred into main() so that `--help`
# works even before dependencies are installed via pip.


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="face_blur",
        description=(
            "Face Blur on Surveillance Videos & Images\n"
            "==========================================\n"
            "Detects and anonymises every visible human face in videos\n"
            "and images using deep-learning-based detection (RetinaFace)\n"
            "and multi-object tracking (ByteTrack)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input sources (mutually exclusive: --video, --image, --input-dir)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video",
        type=str,
        help="Path to a single video file (MP4, AVI, MKV, MOV).",
    )
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single image file (JPG, PNG, BMP, etc.).",
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Path to a directory of images or videos for batch processing.",
    )

    # ── Processing options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gaussian", "pixelate"],
        default="gaussian",
        help="Blur mode: 'gaussian' (default) or 'pixelate'.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override minimum confidence threshold for HIGH-tier detections (default: 0.8).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (overrides default output paths).",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["image", "video"],
        default="image",
        help="Content type when using --input-dir (default: image).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (detection boxes, tracker IDs, confidence scores).",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # ── Deferred imports (so --help works without deps) ───────────────
    from app.config import cfg
    from app.utils import setup_logging, ensure_dirs

    # ── Setup logging ─────────────────────────────────────────────────
    logger = setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("=" * 60)
    logger.info("  Face Blur Pipeline — Starting")
    logger.info("=" * 60)

    # ── Apply CLI overrides to config ─────────────────────────────────
    # We mutate the global singleton config so all modules pick up changes.
    cfg.blur_mode = args.mode
    cfg.debug = args.debug

    if args.confidence is not None:
        cfg.confidence_high = args.confidence
        logger.info("Confidence threshold overridden to %.2f", args.confidence)

    logger.info("Blur mode : %s", cfg.blur_mode)
    logger.info("Device    : %s", cfg.device)
    logger.info("Debug     : %s", cfg.debug)

    ensure_dirs()

    # ── Route to the appropriate processor ────────────────────────────

    if args.video:
        # Single video processing
        from app.video_processor import VideoProcessor
        processor = VideoProcessor(cfg)
        result = processor.process_video(args.video, args.output_dir)
        if result:
            logger.info("✓ Output saved to: %s", result)
        else:
            logger.error("✗ Video processing failed.")
            sys.exit(1)

    elif args.image:
        # Single image processing
        from app.image_processor import ImageProcessor
        processor = ImageProcessor(cfg)
        result = processor.process_image(args.image, args.output_dir)
        if result:
            logger.info("✓ Output saved to: %s", result)
        else:
            logger.error("✗ Image processing failed.")
            sys.exit(1)

    elif args.input_dir:
        # Batch processing
        if args.type == "video":
            from app.video_processor import VideoProcessor
            processor = VideoProcessor(cfg)
            results = processor.process_directory(args.input_dir, args.output_dir)
        else:
            from app.image_processor import ImageProcessor
            processor = ImageProcessor(cfg)
            results = processor.process_directory(args.input_dir, args.output_dir)

        logger.info("✓ Batch complete: %d outputs generated.", len(results))

    logger.info("=" * 60)
    logger.info("  Face Blur Pipeline — Done")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
