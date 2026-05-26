"""
detector.py — Deep-Learning Face Detection
============================================

Uses **InsightFace** with the ``buffalo_l`` model bundle (which includes
a RetinaFace-based detector) to locate every visible human face in a
frame.

Why InsightFace / RetinaFace instead of Haar Cascades?
------------------------------------------------------
• **Haar cascades** are classical ML (Viola-Jones).  They are fast but
  have *very* high false-negative rates on:
    – side-profile faces  (cascade is frontal-only)
    – partially occluded faces
    – small / distant faces
    – low-light / noisy images
  They also produce many false positives on texture-rich backgrounds.

• **RetinaFace** (deep learning, anchor-based dense regression) was
  specifically designed for "in-the-wild" face detection.  Advantages:
    – State-of-the-art accuracy on WIDER FACE benchmark
    – Handles extreme poses (profile, up/down)
    – Detects very small faces (down to ~10 px)
    – Provides 5-point facial landmarks (useful for alignment)
    – Single forward pass — GPU-accelerated

• **InsightFace ``buffalo_l``** bundles RetinaFace detection + ArcFace
  recognition in one download.  We only use the *detection* part.

Confidence-based filtering
--------------------------
The detector returns every face with its confidence score.  We classify
detections into three tiers (thresholds in ``config.py``):

    HIGH  (> 0.8)   → definitely blur
    MEDIUM (0.4–0.8) → soft detection, needs tracker confirmation
    LOW   (< 0.4)   → ignored unless previously tracked

This prevents both missed faces and over-aggressive false positives.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

# pyrefly: ignore [missing-import]
import numpy as np

from app.config import cfg
from app.utils import pad_bbox

logger = logging.getLogger("face_blur.detector")


# ---------------------------------------------------------------------------
#  Data container for a single detection
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    One detected face.

    Attributes
    ----------
    bbox : tuple of int
        (x1, y1, x2, y2) bounding box in pixel coordinates.
    confidence : float
        Detection confidence in [0, 1].
    landmarks : np.ndarray or None
        5-point facial landmarks (eyes, nose, mouth corners).
    tier : str
        Classification: 'high', 'medium', or 'low'.
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    landmarks: Optional[np.ndarray] = None
    tier: str = "high"


# ---------------------------------------------------------------------------
#  Face Detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Wrapper around InsightFace's FaceAnalysis for face detection.

    Usage
    -----
    >>> detector = FaceDetector()
    >>> detections = detector.detect(bgr_frame)
    """

    def __init__(self, config=None):
        """
        Initialise the InsightFace model.

        Parameters
        ----------
        config : Config, optional
            Override the global config.
        """
        self.cfg = config or cfg

        # Lazy-import insightface so we get a clear error if not installed
        try:
            # pyrefly: ignore [missing-import]
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is required.  Install it with:\n"
                "    pip install insightface onnxruntime-gpu\n"
                "or  pip install insightface onnxruntime  (CPU-only)"
            ) from exc

        logger.info("Initialising InsightFace detector (model=buffalo_l) …")

        # Create the FaceAnalysis app
        # 'buffalo_l' = large model bundle with RetinaFace + ArcFace
        # We only need the detection part (det_size controls input resolution)
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=self._get_providers(),
        )

        # Prepare with detection size — larger = more accurate but slower
        # 640×640 is the standard; works well for surveillance
        self._app.prepare(ctx_id=0 if self.cfg.device == "cuda" else -1,
                          det_size=(640, 640))

        logger.info("InsightFace detector ready (device=%s)", self.cfg.device)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect all faces in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR image (H×W×3, uint8).

        Returns
        -------
        list of Detection
            Detected faces with bounding boxes, confidences, and landmarks.
        """
        # InsightFace expects BGR (which OpenCV already provides)
        faces = self._app.get(frame)

        detections: List[Detection] = []
        for face in faces:
            # face.bbox is a numpy array [x1, y1, x2, y2]
            raw_bbox = face.bbox.astype(int)
            conf = float(face.det_score)

            # Classify confidence tier
            tier = self._classify_confidence(conf)

            # Apply safety padding and clamp to frame bounds
            padded = pad_bbox(
                tuple(raw_bbox),
                self.cfg.bbox_padding,
                frame.shape[:2],
            )

            detections.append(Detection(
                bbox=padded,
                confidence=conf,
                landmarks=face.kps if hasattr(face, "kps") else None,
                tier=tier,
            ))

        logger.debug("Detected %d faces (frame shape=%s)", len(detections), frame.shape[:2])
        return detections

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _classify_confidence(self, conf: float) -> str:
        """Map a confidence score to a tier string."""
        if conf >= self.cfg.confidence_high:
            return "high"
        elif conf >= self.cfg.confidence_medium:
            return "medium"
        else:
            return "low"

    def _get_providers(self) -> list:
        """
        Return the ONNX Runtime execution providers in priority order.

        If CUDA is available we prefer CUDAExecutionProvider, otherwise
        fall back to CPU.
        """
        if self.cfg.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
