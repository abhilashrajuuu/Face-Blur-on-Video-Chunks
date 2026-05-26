"""
image_processor.py — Single-Image & Batch Image Processing
============================================================

Orchestrates the full pipeline for still images:
    enhance → detect → blur → save

Supports:
  • processing a single image file
  • batch-processing every image in a directory
  • saving debug visualisations alongside the blurred output
"""

import logging
from pathlib import Path
from typing import List, Optional

# pyrefly: ignore [missing-import]
import cv2

from app.config import cfg
from app.detector import FaceDetector
from app.enhance import FrameEnhancer
from app.blur import FaceBlurrer
from app.utils import draw_debug_frame, ensure_dirs

logger = logging.getLogger("face_blur.image_processor")

# Supported image extensions (case-insensitive)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class ImageProcessor:
    """
    End-to-end image face-blurring processor.

    Usage
    -----
    >>> proc = ImageProcessor()
    >>> proc.process_image("sample_inputs/sample_image.jpg")
    >>> proc.process_directory("datasets/images/")
    """

    def __init__(self, config=None):
        """
        Initialise the pipeline components.

        Parameters
        ----------
        config : Config, optional
            Override the global config.
        """
        self.cfg = config or cfg

        # Create pipeline components
        self.enhancer = FrameEnhancer(self.cfg)
        self.detector = FaceDetector(self.cfg)
        self.blurrer = FaceBlurrer(self.cfg)

        ensure_dirs()
        logger.info("ImageProcessor initialised (blur_mode=%s)", self.cfg.blur_mode)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Process a single image: enhance, detect faces, blur, save.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        output_dir : str, optional
            Custom output directory.  Defaults to ``outputs/blurred_images/``.

        Returns
        -------
        str or None
            Path to the saved blurred image, or None on failure.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            return None

        out_dir = Path(output_dir) if output_dir else self.cfg.blurred_images_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing image: %s", image_path.name)

        # ── Load image ────────────────────────────────────────────────
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.error("Failed to read image: %s", image_path)
            return None

        # ── Enhance ───────────────────────────────────────────────────
        enhanced = self.enhancer.enhance(frame)

        # ── Detect faces ──────────────────────────────────────────────
        detections = self.detector.detect(enhanced)
        logger.info(
            "  Found %d faces (high=%d, medium=%d, low=%d)",
            len(detections),
            sum(1 for d in detections if d.tier == "high"),
            sum(1 for d in detections if d.tier == "medium"),
            sum(1 for d in detections if d.tier == "low"),
        )

        # For images, we blur HIGH and MEDIUM detections (no tracker context)
        faces_to_blur = [d for d in detections if d.tier in ("high", "medium")]

        # ── Blur faces ────────────────────────────────────────────────
        blurred = self.blurrer.blur_faces(frame.copy(), faces_to_blur)

        # ── Save output ──────────────────────────────────────────────
        output_name = f"blurred_{image_path.stem}.jpg"
        output_path = out_dir / output_name
        cv2.imwrite(str(output_path), blurred)
        logger.info("  Saved blurred image → %s", output_path)

        # ── Save debug output ────────────────────────────────────────
        if self.cfg.debug:
            self._save_debug(frame, detections, image_path.stem)

        return str(output_path)

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Batch-process all images in a directory.

        Parameters
        ----------
        input_dir : str
            Directory containing images.
        output_dir : str, optional
            Custom output directory.

        Returns
        -------
        list of str
            Paths to all saved blurred images.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            logger.error("Input directory not found: %s", input_dir)
            return []

        # Gather image files
        image_files = sorted(
            f for f in input_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            logger.warning("No images found in %s", input_dir)
            return []

        logger.info("Batch processing %d images from %s", len(image_files), input_dir)
        results = []
        for img_path in image_files:
            result = self.process_image(str(img_path), output_dir)
            if result:
                results.append(result)

        logger.info("Batch complete: %d/%d images processed", len(results), len(image_files))
        return results

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _save_debug(self, frame, detections, stem):
        """Save a debug visualisation with detection boxes and confidence scores."""
        debug_dir = self.cfg.debug_dir
        debug_dir.mkdir(parents=True, exist_ok=True)

        bboxes = [d.bbox for d in detections]
        confs = [d.confidence for d in detections]

        debug_frame = draw_debug_frame(frame, bboxes, confs)
        debug_path = debug_dir / f"debug_{stem}.jpg"
        cv2.imwrite(str(debug_path), debug_frame)
        logger.info("  Saved debug image → %s", debug_path)
