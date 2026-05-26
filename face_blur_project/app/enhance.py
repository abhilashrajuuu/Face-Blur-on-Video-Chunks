"""
enhance.py — Frame Enhancement / Preprocessing Pipeline
========================================================

Before feeding frames to the face detector, we apply a series of
image-enhancement steps that improve detection accuracy on challenging
surveillance footage:

    1. **Denoise** — reduce sensor / compression noise with
       ``cv2.fastNlMeansDenoisingColored``.
    2. **CLAHE** — Contrast-Limited Adaptive Histogram Equalisation on
       the L channel (LAB colour space) to boost local contrast.
    3. **Gamma correction** — brighten under-exposed (low-light) frames
       by estimating mean luminance and adjusting gamma.
    4. **Sharpen** (optional) — unsharp-mask to recover detail from
       motion-blurred or out-of-focus frames.

Each step can be toggled independently via ``config.py``.

Why these steps matter
----------------------
Surveillance cameras often produce images with:
  • poor lighting (IR or dim corridors)
  • heavy JPEG/H.264 compression artefacts
  • motion blur from low shutter speeds

Deep-learning detectors like RetinaFace are *much* more robust than
classical Haar cascades, but they still benefit from clean input.
These lightweight OpenCV operations add <2 ms per frame on CPU.
"""

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import logging

from app.config import cfg

logger = logging.getLogger("face_blur.enhance")


class FrameEnhancer:
    """
    Stateless image-enhancement pipeline.

    Usage
    -----
    >>> enhancer = FrameEnhancer()
    >>> clean_frame = enhancer.enhance(raw_frame)
    """

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : Config, optional
            Override the global config (useful for testing).
        """
        self.cfg = config or cfg

        # Pre-create the CLAHE object (reusable across frames)
        if self.cfg.enhance_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip_limit,
                tileGridSize=self.cfg.clahe_tile_grid,
            )
        else:
            self._clahe = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full enhancement pipeline on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR image (H×W×3, uint8).

        Returns
        -------
        np.ndarray
            Enhanced BGR image, same shape.
        """
        result = frame.copy()

        # Step 1 — Denoise
        if self.cfg.enhance_denoise:
            result = self._denoise(result)

        # Step 2 — CLAHE contrast enhancement
        if self.cfg.enhance_clahe and self._clahe is not None:
            result = self._apply_clahe(result)

        # Step 3 — Gamma correction for low-light frames
        if self.cfg.enhance_gamma:
            result = self._gamma_correct(result)

        # Step 4 — Optional sharpening
        if self.cfg.enhance_sharpen:
            result = self._sharpen(result)

        return result

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply fast non-local-means denoising (colour variant).

        Parameters are tuned for surveillance-grade noise levels:
          h=6  → filter strength for luminance
          hForColorComponents=6  → filter strength for chrominance
          templateWindowSize=7   → patch size
          searchWindowSize=21    → search area
        """
        return cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=6,
            hColor=6,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel of the LAB colour space.

        This boosts local contrast without over-saturating colours,
        which is critical for CCTV footage with uneven lighting.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE only to the luminance channel
        l_enhanced = self._clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _gamma_correct(self, frame: np.ndarray) -> np.ndarray:
        """
        Automatically brighten dark frames using gamma correction.

        Algorithm:
          1. Convert to grayscale and compute mean luminance.
          2. If mean < threshold (low light), compute a gamma < 1.0
             to brighten the image.
          3. Apply a look-up table for O(1) per-pixel transform.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luminance = np.mean(gray)

        if mean_luminance >= self.cfg.gamma_low_light_threshold:
            # Frame is bright enough — skip correction
            return frame

        # Compute gamma: lower mean → lower gamma → more brightening
        # Clamp gamma to [0.3, 1.0] to avoid extreme brightening
        gamma = max(0.3, mean_luminance / self.cfg.gamma_low_light_threshold)

        logger.debug(
            "Low-light frame detected (mean=%.1f). Applying gamma=%.2f",
            mean_luminance, gamma,
        )

        # Build look-up table
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(frame, table)

    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking to recover detail from blurry frames.

        Unsharp mask = original + alpha * (original − blurred)
        """
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
        # alpha=1.5 gives moderate sharpening; beta=-0.5 subtracts the blur
        sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        return sharpened
