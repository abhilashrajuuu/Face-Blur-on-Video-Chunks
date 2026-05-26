"""
utils.py — Shared Utility Functions
====================================

Helper functions used across multiple modules:
  • logging setup
  • directory creation
  • bounding-box manipulation (pad, clamp)
  • debug frame annotation
  • device detection
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

from app.config import cfg


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return the root project logger.

    Logs go to both stdout and a file (outputs/debug/pipeline.log).

    Parameters
    ----------
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("face_blur")
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Fix for Windows console UnicodeEncodeError with emojis (✓, →, etc)
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (inside debug output directory)
    log_dir = cfg.debug_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
#  Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all required output / dataset directories if they don't exist."""
    dirs = [
        cfg.datasets_videos_dir,
        cfg.datasets_images_dir,
        cfg.datasets_downloaded_dir,
        cfg.blurred_videos_dir,
        cfg.blurred_images_dir,
        cfg.debug_dir,
        cfg.sample_inputs_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  Bounding-box utilities
# ---------------------------------------------------------------------------

def clamp_bbox(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Clamp a bounding box (x1, y1, x2, y2) so it stays within frame bounds.

    Parameters
    ----------
    bbox : tuple of int
        (x1, y1, x2, y2) pixel coordinates.
    frame_shape : tuple of int
        (height, width) of the frame.

    Returns
    -------
    tuple of int
        Clamped (x1, y1, x2, y2).
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    return (x1, y1, x2, y2)


def pad_bbox(
    bbox: Tuple[int, int, int, int],
    padding: float,
    frame_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Expand a bounding box by a fractional *padding* on each side,
    then clamp to frame bounds.

    Parameters
    ----------
    bbox : tuple of int
        (x1, y1, x2, y2).
    padding : float
        Fraction of box width/height to add (e.g. 0.15 → 15 %).
    frame_shape : tuple of int
        (height, width).

    Returns
    -------
    tuple of int
        Padded and clamped (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    return clamp_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame_shape)


# ---------------------------------------------------------------------------
#  Debug annotation
# ---------------------------------------------------------------------------

def draw_debug_frame(
    frame: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    confidences: Optional[List[float]] = None,
    tracker_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Draw detection boxes, confidence scores, and tracker IDs on a frame copy.

    This is used to generate the debug visualisation saved to outputs/debug/.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame.
    bboxes : list of (x1, y1, x2, y2)
        Face bounding boxes.
    confidences : list of float, optional
        Per-box detection confidence.
    tracker_ids : list of int, optional
        Per-box tracker ID.

    Returns
    -------
    np.ndarray
        Annotated copy of the frame (original is not modified).
    """
    debug = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Choose colour: green for high-confidence, yellow for medium, red for low
        conf = confidences[i] if confidences else 1.0
        if conf >= cfg.confidence_high:
            colour = (0, 255, 0)       # green
        elif conf >= cfg.confidence_medium:
            colour = (0, 255, 255)     # yellow
        else:
            colour = (0, 0, 255)       # red

        # Draw rectangle
        cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)

        # Build label string
        parts: List[str] = []
        if tracker_ids and i < len(tracker_ids):
            parts.append(f"ID:{tracker_ids[i]}")
        if confidences and i < len(confidences):
            parts.append(f"{confidences[i]:.2f}")
        label = " | ".join(parts)

        if label:
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                debug,
                (int(x1), int(y1) - th - 8),
                (int(x1) + tw + 4, int(y1)),
                colour,
                -1,
            )
            cv2.putText(
                debug, label,
                (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

    return debug


# ---------------------------------------------------------------------------
#  Device helper
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return the compute device string ('cuda' or 'cpu')."""
    return cfg.device
