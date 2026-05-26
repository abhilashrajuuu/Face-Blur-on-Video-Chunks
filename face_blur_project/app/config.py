"""
config.py — Central Configuration for Face Blur Pipeline
=========================================================

All tunable parameters live here. Other modules import from this file
so that every knob is in one place.  Modify values below or override
them at runtime via CLI flags / environment variables.

Design Rationale
----------------
Using a frozen dataclass gives us:
  • immutability after construction (catches accidental mutations)
  • type hints for IDE auto-complete
  • easy serialisation to dict / JSON for logging
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# torch is optional at import time so that `--help` works before
# pip install.  We try to import it and fall back gracefully.
try:
    # pyrefly: ignore [missing-import]
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Auto-detect compute device
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, else 'cpu'."""
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
#  Project root — resolved relative to THIS file so it works from any CWD
# ---------------------------------------------------------------------------

# This points to  face_blur_project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
#  Main configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Central configuration container for the face blur pipeline."""

    # ── Blur settings ─────────────────────────────────────────────────────
    # "gaussian" applies a dynamically-sized Gaussian kernel.
    # "pixelate" down-scales the face region then up-scales it back.
    blur_mode: Literal["gaussian", "pixelate"] = "gaussian"

    # Base kernel size for Gaussian blur (actual size is scaled by face dim).
    gaussian_kernel_base: int = 51

    # Block size for pixelation (smaller = heavier pixelation).
    pixelate_block_size: int = 10

    # ── Detection confidence thresholds ───────────────────────────────────
    # Confidence above this → definitely blur the face.
    confidence_high: float = 0.8

    # Confidence between medium and high → rely on tracker / history.
    confidence_medium: float = 0.4

    # Below confidence_medium → ignore unless the face was tracked before.

    # ── Bounding-box padding ──────────────────────────────────────────────
    # Fractional padding added around each detected face box.
    # 0.15 means 15 % of the box width/height is added on each side.
    bbox_padding: float = 0.15

    # ── Temporal smoothing ────────────────────────────────────────────────
    # After a tracked face disappears, keep blurring for this many frames
    # to avoid revealing faces during brief detection gaps.
    temporal_smoothing_frames: int = 5

    # ── Tracker (ByteTrack via supervision) ───────────────────────────────
    # Minimum detection score for ByteTrack to consider activating a track.
    tracker_activation_threshold: float = 0.25

    # Frames a lost track is kept alive before being dropped.
    tracker_lost_buffer: int = 30

    # Minimum consecutive detections before a track is confirmed.
    tracker_min_consecutive: int = 2

    # ── Enhancement toggles ───────────────────────────────────────────────
    enhance_denoise: bool = True
    enhance_clahe: bool = True
    enhance_gamma: bool = True
    enhance_sharpen: bool = False          # off by default; enable for blurry footage

    # CLAHE clip-limit and tile grid size
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple = (8, 8)

    # Gamma correction target mean luminance (below this → brighten)
    gamma_low_light_threshold: float = 80.0

    # ── Device ────────────────────────────────────────────────────────────
    device: str = field(default_factory=_detect_device)

    # ── Debug ─────────────────────────────────────────────────────────────
    debug: bool = False

    # ── Paths ─────────────────────────────────────────────────────────────
    project_root: Path = PROJECT_ROOT

    @property
    def datasets_dir(self) -> Path:
        return self.project_root / "datasets"

    @property
    def datasets_videos_dir(self) -> Path:
        return self.project_root / "datasets" / "videos"

    @property
    def datasets_images_dir(self) -> Path:
        return self.project_root / "datasets" / "images"

    @property
    def datasets_downloaded_dir(self) -> Path:
        return self.project_root / "datasets" / "downloaded"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def blurred_videos_dir(self) -> Path:
        return self.project_root / "outputs" / "blurred_videos"

    @property
    def blurred_images_dir(self) -> Path:
        return self.project_root / "outputs" / "blurred_images"

    @property
    def debug_dir(self) -> Path:
        return self.project_root / "outputs" / "debug"

    @property
    def sample_inputs_dir(self) -> Path:
        return self.project_root / "sample_inputs"


# ---------------------------------------------------------------------------
#  Singleton-style default config (importable as `from app.config import cfg`)
# ---------------------------------------------------------------------------

cfg = Config()
