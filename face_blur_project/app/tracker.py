"""
tracker.py — Multi-Object Face Tracking with ByteTrack
=======================================================

Wraps the **ByteTrack** implementation from the ``supervision`` library
to maintain persistent face identities across video frames.

Why tracking matters for face blurring
--------------------------------------
Without tracking, the detector runs independently on each frame.  This
causes two problems:

1. **Flickering** — a face that's detected on frame N, missed on frame
   N+1, and detected again on frame N+2 will flicker between blurred
   and un-blurred.  This can reveal a person's identity in a single
   un-blurred frame.

2. **Uncertain detections** — a medium-confidence detection (0.4–0.8)
   on its own is risky to blur (could be a false positive).  But if the
   tracker shows that box has been consistently detected for 10 frames,
   we can blur it with high confidence.

ByteTrack key ideas
-------------------
ByteTrack uses a two-stage association strategy:
  • First match *high*-confidence detections to existing tracks.
  • Then match *low*-confidence detections to remaining unmatched tracks.
This recovers partially occluded or blurry faces that other trackers
would drop.  Perfect for surveillance footage.

Temporal smoothing
------------------
After a tracked face disappears, we keep its last known bounding box
and continue blurring for ``temporal_smoothing_frames`` additional
frames.  This prevents brief "flash" reveals.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
import numpy as np

from app.config import cfg
from app.detector import Detection

logger = logging.getLogger("face_blur.tracker")


# ---------------------------------------------------------------------------
#  Data container for a tracked face
# ---------------------------------------------------------------------------

@dataclass
class TrackedFace:
    """
    A face detection augmented with a persistent tracker ID and
    temporal-smoothing metadata.
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    tracker_id: int
    tier: str = "high"
    landmarks: Optional[np.ndarray] = None
    frames_since_seen: int = 0          # 0 = just detected this frame


# ---------------------------------------------------------------------------
#  Face Tracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """
    Wraps ``supervision.ByteTrack`` and adds temporal smoothing.

    Usage
    -----
    >>> tracker = FaceTracker()
    >>> for frame in video:
    ...     detections = detector.detect(frame)
    ...     tracked = tracker.update(detections, frame.shape[:2])
    ...     blurred = blurrer.blur_faces(frame, tracked)
    """

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : Config, optional
            Override the global config.
        """
        self.cfg = config or cfg

        # Lazy-import supervision
        try:
            # pyrefly: ignore [missing-import]
            import supervision as sv
        except ImportError as exc:
            raise ImportError(
                "The 'supervision' library is required for tracking.\n"
                "    pip install supervision"
            ) from exc

        # Initialise ByteTrack from the supervision library
        # NOTE: ByteTrack is deprecated in supervision>=0.28 in favour of
        # the standalone 'trackers' package.  We suppress the warning here;
        # migrate to `from trackers import ByteTrackTracker` when upgrading.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.cfg.tracker_activation_threshold,
                lost_track_buffer=self.cfg.tracker_lost_buffer,
                minimum_matching_threshold=0.8,
                frame_rate=30,
                minimum_consecutive_frames=self.cfg.tracker_min_consecutive,
            )

        # ── Temporal smoothing state ──────────────────────────────────
        # Maps tracker_id → TrackedFace from the most recent detection.
        # When a track disappears, we keep it here and increment
        # frames_since_seen each frame until it exceeds the threshold.
        self._last_seen: Dict[int, TrackedFace] = {}

        logger.info(
            "ByteTrack tracker initialised (activation=%.2f, buffer=%d, "
            "min_consecutive=%d, temporal_smooth=%d frames)",
            self.cfg.tracker_activation_threshold,
            self.cfg.tracker_lost_buffer,
            self.cfg.tracker_min_consecutive,
            self.cfg.temporal_smoothing_frames,
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
    ) -> List[TrackedFace]:
        """
        Feed new detections into the tracker and return tracked faces.

        Confidence-based logic
        ----------------------
        • HIGH-tier detections are always passed to the tracker.
        • MEDIUM-tier detections are passed to the tracker (ByteTrack's
          two-stage matching will confirm or reject them).
        • LOW-tier detections are only included if they match an
          *existing* track (checked after the tracker runs).

        After the tracker runs, we apply temporal smoothing to continue
        blurring recently lost tracks.

        Parameters
        ----------
        detections : list of Detection
            Raw detections from the detector.
        frame_shape : tuple of int
            (height, width) of the current frame.

        Returns
        -------
        list of TrackedFace
            Faces to blur this frame, with persistent tracker IDs.
        """
        # pyrefly: ignore [missing-import]
        import supervision as sv

        # ── Separate detections by tier ──────────────────────────────
        high_medium = [d for d in detections if d.tier in ("high", "medium")]
        low = [d for d in detections if d.tier == "low"]

        # ── Build a supervision Detections object ────────────────────
        if high_medium:
            xyxy = np.array([d.bbox for d in high_medium], dtype=np.float32)
            confs = np.array([d.confidence for d in high_medium], dtype=np.float32)
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=confs,
            )
        else:
            sv_dets = sv.Detections.empty()

        # ── Run ByteTrack ────────────────────────────────────────────
        tracked_sv = self._tracker.update_with_detections(sv_dets)

        # ── Build result list ────────────────────────────────────────
        active_ids = set()
        results: List[TrackedFace] = []

        if tracked_sv.tracker_id is not None:
            for i in range(len(tracked_sv)):
                tid = int(tracked_sv.tracker_id[i])
                bbox = tuple(tracked_sv.xyxy[i].astype(int))
                conf = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 1.0

                tf = TrackedFace(
                    bbox=bbox,
                    confidence=conf,
                    tracker_id=tid,
                    tier="high" if conf >= self.cfg.confidence_high else "medium",
                    frames_since_seen=0,
                )
                results.append(tf)
                active_ids.add(tid)

                # Update last-seen map
                self._last_seen[tid] = tf

        # ── Handle LOW-tier detections that match existing tracks ────
        # If a low-confidence detection overlaps significantly with a
        # known track, we include it (the tracker already handles most
        # of this, but this is a safety net).
        for d in low:
            for tid, prev in self._last_seen.items():
                if tid in active_ids:
                    continue
                if self._iou(d.bbox, prev.bbox) > 0.3:
                    # Reuse the tracked ID
                    tf = TrackedFace(
                        bbox=d.bbox,
                        confidence=d.confidence,
                        tracker_id=tid,
                        tier="low",
                        frames_since_seen=0,
                    )
                    results.append(tf)
                    active_ids.add(tid)
                    self._last_seen[tid] = tf
                    break

        # ── Temporal smoothing: keep recently lost tracks ────────────
        for tid, prev in list(self._last_seen.items()):
            if tid in active_ids:
                continue  # still active, skip

            prev.frames_since_seen += 1
            if prev.frames_since_seen <= self.cfg.temporal_smoothing_frames:
                # Continue blurring at the last known position
                results.append(prev)
                active_ids.add(tid)
                logger.debug(
                    "Temporal smooth: track %d frame %d/%d",
                    tid, prev.frames_since_seen,
                    self.cfg.temporal_smoothing_frames,
                )
            else:
                # Track has been gone too long — remove
                del self._last_seen[tid]

        logger.debug(
            "Tracker update: %d active tracks, %d smoothed",
            len([r for r in results if r.frames_since_seen == 0]),
            len([r for r in results if r.frames_since_seen > 0]),
        )
        return results

    def reset(self) -> None:
        """Reset the tracker state (e.g. when switching to a new video)."""
        self._last_seen.clear()
        # Re-create the ByteTrack instance
        # pyrefly: ignore [missing-import]
        import supervision as sv
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.cfg.tracker_activation_threshold,
                lost_track_buffer=self.cfg.tracker_lost_buffer,
                minimum_matching_threshold=0.8,
                frame_rate=30,
                minimum_consecutive_frames=self.cfg.tracker_min_consecutive,
            )
        logger.info("Tracker state reset.")

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        """
        Compute Intersection-over-Union between two boxes (x1, y1, x2, y2).
        """
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
