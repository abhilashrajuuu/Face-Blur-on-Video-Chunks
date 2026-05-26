"""
blur.py — Face Blurring / Anonymisation
========================================

Applies blur to detected face regions.  Two modes are supported:

1. **Gaussian blur** — classic privacy technique.  The kernel size is
   scaled dynamically based on the face bounding-box dimensions so that
   large faces get heavier blur and small faces still look anonymised.

2. **Pixelation** — down-scales the face region to a tiny resolution
   and then up-scales it back, creating a mosaic / pixel-art effect.

The mode is controlled via ``config.blur_mode`` (``"gaussian"`` or
``"pixelate"``).

Both modes operate in-place on the frame for performance, but the
public API returns the modified frame for chaining.
"""

import logging
from typing import List, Union

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

from app.config import cfg
from app.detector import Detection
from app.tracker import TrackedFace
from app.utils import clamp_bbox

logger = logging.getLogger("face_blur.blur")


class FaceBlurrer:
    """
    Apply Gaussian or pixelation blur to face bounding boxes.

    Usage
    -----
    >>> blurrer = FaceBlurrer()
    >>> blurred_frame = blurrer.blur_faces(frame, tracked_faces)
    """

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : Config, optional
            Override the global config.
        """
        self.cfg = config or cfg

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def blur_faces(
        self,
        frame: np.ndarray,
        faces: List[Union[Detection, TrackedFace]],
    ) -> np.ndarray:
        """
        Blur every face in *faces* on the given frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame (modified in-place AND returned).
        faces : list
            Detection or TrackedFace objects with ``.bbox`` attribute.

        Returns
        -------
        np.ndarray
            The frame with faces blurred.
        """
        for face in faces:
            x1, y1, x2, y2 = clamp_bbox(face.bbox, frame.shape[:2])

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract face region-of-interest
            roi = frame[y1:y2, x1:x2]

            if self.cfg.blur_mode == "pixelate":
                blurred_roi = self._pixelate(roi)
            else:
                blurred_roi = self._gaussian_blur(roi)

            # Write back
            frame[y1:y2, x1:x2] = blurred_roi

        return frame

    # ------------------------------------------------------------------
    #  Blur implementations
    # ------------------------------------------------------------------

    def _gaussian_blur(self, roi: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur with a kernel size scaled to the face size.

        The kernel size is derived from the largest dimension of the ROI:
            k = max(w, h) // 2 * 2 + 1   (ensures odd number)
        Minimum kernel is 15 to guarantee visible anonymisation.

        A larger face → larger kernel → heavier blur, so the face
        remains unrecognisable regardless of its apparent size.
        """
        h, w = roi.shape[:2]

        # Dynamic kernel size based on face dimensions
        k = max(w, h) // 2
        k = max(k, 15)                   # minimum kernel size
        k = k if k % 2 == 1 else k + 1   # must be odd

        # Sigma=0 → OpenCV auto-computes sigma from kernel size
        return cv2.GaussianBlur(roi, (k, k), 0)

    def _pixelate(self, roi: np.ndarray) -> np.ndarray:
        """
        Pixelate by down-scaling then up-scaling the face region.

        The block size controls how many pixels each "super-pixel"
        covers.  Smaller block_size → heavier pixelation.

        Steps:
            1. Shrink the ROI to (w/block, h/block) using INTER_LINEAR
            2. Scale it back to original size using INTER_NEAREST
               (nearest-neighbour preserves the blocky look)
        """
        h, w = roi.shape[:2]
        block = self.cfg.pixelate_block_size

        # Down-scale
        small_w = max(1, w // block)
        small_h = max(1, h // block)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # Up-scale back to original size with nearest-neighbour
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
