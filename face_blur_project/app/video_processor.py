"""
video_processor.py — End-to-End Video Face-Blurring Pipeline
=============================================================

Orchestrates the full pipeline for video files:
    read frames → enhance → detect → track → blur → write MP4

Key features:
  • Preserves original FPS and resolution
  • Uses ByteTrack for persistent face tracking
  • Applies temporal smoothing to prevent flicker
  • Saves debug frames with detection boxes / tracker IDs
  • Muxes original audio track back into the output using ffmpeg
  • Progress bar via tqdm

Output format is always MP4 (H.264 via OpenCV or ffmpeg fallback).
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from tqdm import tqdm

from app.config import cfg
from app.detector import FaceDetector
from app.enhance import FrameEnhancer
from app.tracker import FaceTracker
from app.blur import FaceBlurrer
from app.utils import draw_debug_frame, ensure_dirs

logger = logging.getLogger("face_blur.video_processor")


class VideoProcessor:
    """
    End-to-end video face-blurring processor.

    Usage
    -----
    >>> proc = VideoProcessor()
    >>> proc.process_video("sample_inputs/sample_video.mp4")
    """

    def __init__(self, config=None):
        """
        Initialise all pipeline components.

        Parameters
        ----------
        config : Config, optional
            Override the global config.
        """
        self.cfg = config or cfg

        # Pipeline components
        self.enhancer = FrameEnhancer(self.cfg)
        self.detector = FaceDetector(self.cfg)
        self.tracker = FaceTracker(self.cfg)
        self.blurrer = FaceBlurrer(self.cfg)

        ensure_dirs()
        logger.info(
            "VideoProcessor initialised (blur_mode=%s, device=%s)",
            self.cfg.blur_mode, self.cfg.device,
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Process a single video file: read → enhance → detect → track →
        blur → write MP4 → mux audio.

        Parameters
        ----------
        video_path : str
            Path to the input video.
        output_dir : str, optional
            Custom output directory.  Defaults to ``outputs/blurred_videos/``.

        Returns
        -------
        str or None
            Path to the saved blurred MP4, or None on failure.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error("Video not found: %s", video_path)
            return None

        out_dir = Path(output_dir) if output_dir else self.cfg.blurred_videos_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Open input video ─────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            return None

        # Read video metadata
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Processing video: %s (%dx%d @ %.1f FPS, %d frames)",
            video_path.name, width, height, fps, total_frames,
        )

        # ── Prepare output video writer ──────────────────────────────
        # We write to a temp file first, then mux audio and move
        output_name = f"blurred_{video_path.stem}.mp4"
        final_output = out_dir / output_name

        # Use mp4v codec (widely supported H.264 alternative)
        # If H.264 is available via ffmpeg, we'll re-encode later
        temp_output = out_dir / f"_temp_{output_name}"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))

        if not writer.isOpened():
            logger.error("Failed to create video writer for %s", temp_output)
            cap.release()
            return None

        # ── Reset tracker for this video ─────────────────────────────
        self.tracker.reset()

        # ── Debug writer (optional) ──────────────────────────────────
        debug_writer = None
        if self.cfg.debug:
            debug_path = self.cfg.debug_dir / f"debug_{video_path.stem}.mp4"
            debug_writer = cv2.VideoWriter(
                str(debug_path), fourcc, fps, (width, height)
            )

        # ── Frame processing loop ────────────────────────────────────
        frame_idx = 0
        total_faces = 0

        pbar = tqdm(total=total_frames, desc=f"Blurring {video_path.name}", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Step 1 — Enhance
            enhanced = self.enhancer.enhance(frame)

            # Step 2 — Detect
            detections = self.detector.detect(enhanced)

            # Step 3 — Track (handles confidence tiers + temporal smoothing)
            tracked_faces = self.tracker.update(detections, frame.shape[:2])
            total_faces += len(tracked_faces)

            # Step 4 — Blur
            blurred = self.blurrer.blur_faces(frame.copy(), tracked_faces)

            # Step 5 — Write output frame
            writer.write(blurred)

            # Step 6 — Write debug frame (if enabled)
            if debug_writer is not None:
                bboxes = [tf.bbox for tf in tracked_faces]
                confs = [tf.confidence for tf in tracked_faces]
                tids = [tf.tracker_id for tf in tracked_faces]
                debug_frame = draw_debug_frame(frame, bboxes, confs, tids)
                debug_writer.write(debug_frame)

            frame_idx += 1
            pbar.update(1)

        pbar.close()

        # ── Clean up ─────────────────────────────────────────────────
        cap.release()
        writer.release()
        if debug_writer is not None:
            debug_writer.release()

        logger.info(
            "Processed %d frames, blurred %d face instances",
            frame_idx, total_faces,
        )

        # ── Mux audio from original video ────────────────────────────
        self._mux_audio(video_path, temp_output, final_output)

        # ── Clean up temp file ───────────────────────────────────────
        if temp_output.exists() and final_output.exists():
            temp_output.unlink()

        logger.info("Saved blurred video → %s", final_output)
        return str(final_output)

    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
    ) -> list:
        """
        Batch-process all MP4/AVI/MKV/MOV videos in a directory.

        Parameters
        ----------
        input_dir : str
            Directory containing video files.
        output_dir : str, optional
            Custom output directory.

        Returns
        -------
        list of str
            Paths to all saved blurred videos.
        """
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        input_dir = Path(input_dir)

        if not input_dir.is_dir():
            logger.error("Input directory not found: %s", input_dir)
            return []

        video_files = sorted(
            f for f in input_dir.iterdir()
            if f.suffix.lower() in video_exts
        )

        if not video_files:
            logger.warning("No video files found in %s", input_dir)
            return []

        logger.info("Batch processing %d videos from %s", len(video_files), input_dir)
        results = []
        for vpath in video_files:
            result = self.process_video(str(vpath), output_dir)
            if result:
                results.append(result)

        logger.info("Batch complete: %d/%d videos processed", len(results), len(video_files))
        return results

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _mux_audio(self, original: Path, video_only: Path, output: Path) -> None:
        """
        Use ffmpeg to copy the audio stream from *original* into *video_only*
        and write the result to *output*.

        If ffmpeg is not available or the original has no audio, we simply
        rename the video-only file to the output path.
        """
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            logger.warning(
                "ffmpeg not found on PATH — output will have no audio. "
                "Install ffmpeg to preserve audio tracks."
            )
            # Just rename temp → final
            if output.exists():
                output.unlink()
            video_only.rename(output)
            return

        try:
            cmd = [
                ffmpeg_bin,
                "-y",                               # overwrite output
                "-i", str(video_only),               # blurred video (no audio)
                "-i", str(original),                 # original video (has audio)
                "-c:v", "copy",                      # copy video stream as-is
                "-c:a", "aac",                       # re-encode audio to AAC
                "-map", "0:v:0",                     # video from first input
                "-map", "1:a:0?",                    # audio from second input (optional)
                "-shortest",                         # match shortest stream
                str(output),
            ]
            logger.info("Muxing audio with ffmpeg …")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )
            if result.returncode != 0:
                logger.warning(
                    "ffmpeg audio mux failed (rc=%d). Output will have no audio.\n%s",
                    result.returncode, result.stderr.decode(errors="replace")[:500],
                )
                # Fall back: just rename
                if output.exists():
                    output.unlink()
                video_only.rename(output)
        except FileNotFoundError:
            logger.warning("ffmpeg binary not found — skipping audio mux.")
            if output.exists():
                output.unlink()
            video_only.rename(output)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out — skipping audio mux.")
            if output.exists():
                output.unlink()
            video_only.rename(output)
