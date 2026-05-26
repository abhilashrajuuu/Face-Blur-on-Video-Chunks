"""
download_dataset.py — Automatic Sample Dataset Downloader
==========================================================

Downloads publicly available surveillance-style videos and images
for testing the face blur pipeline.

Sources:
  • Videos — Pexels (free stock footage, CC0-like license)
  • Images — Unsplash and WIDER FACE sample thumbnails

Downloads are saved to:
    datasets/videos/       ← sample MP4 videos
    datasets/images/       ← sample images with faces
    datasets/downloaded/   ← raw downloads before processing

A copy of the first video and first image is also placed in
``sample_inputs/`` for quick CLI testing.

Usage:
    python scripts/download_dataset.py
"""

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
VIDEOS_DIR = DATASETS_DIR / "videos"
IMAGES_DIR = DATASETS_DIR / "images"
DOWNLOADED_DIR = DATASETS_DIR / "downloaded"
SAMPLE_INPUTS_DIR = PROJECT_ROOT / "sample_inputs"

# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_dataset")


# ---------------------------------------------------------------------------
#  Sample URLs
# ---------------------------------------------------------------------------

# Each entry: (filename, url, description)
# These are publicly accessible Pexels videos (free to use)
SAMPLE_VIDEOS: List[Tuple[str, str, str]] = [
    (
        "crowd_walking.mp4",
        "https://videos.pexels.com/video-files/3195440/3195440-uhd_2560_1440_25fps.mp4",
        "People walking in a crowd — multiple faces, varied angles",
    ),
    (
        "street_scene.mp4",
        "https://videos.pexels.com/video-files/2829195/2829195-hd_1920_1080_30fps.mp4",
        "Street scene with pedestrians — urban surveillance style",
    ),
    (
        "crosswalk_crowd.mp4",
        "https://videos.pexels.com/video-files/855029/855029-hd_1920_1080_30fps.mp4",
        "Crowded crosswalk — medium shot, many moving faces (approx 12s)",
    ),
    (
        "station_crowd.mp4",
        "https://videos.pexels.com/video-files/3028965/3028965-uhd_2560_1440_24fps.mp4",
        "Busy train station — people moving in all directions (approx 13s)",
    ),
    (
        "pedestrians_fast.mp4",
        "https://videos.pexels.com/video-files/3121459/3121459-hd_1920_1080_24fps.mp4",
        "Pedestrians walking — side/frontal views, good for tracking (approx 14s)",
    ),
]

# Publicly available images with human faces (Unsplash / direct links)
SAMPLE_IMAGES: List[Tuple[str, str, str]] = [
    (
        "group_people_01.jpg",
        "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=800&q=80",
        "Group of people — multiple faces at various angles",
    ),
    (
        "crowd_02.jpg",
        "https://images.unsplash.com/photo-1517457373958-b7bdd4587205?w=800&q=80",
        "Crowd scene — many small faces, surveillance-like perspective",
    ),
    (
        "portrait_side_03.jpg",
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80",
        "Side profile portrait — tests profile detection",
    ),
    (
        "low_light_04.jpg",
        "https://images.unsplash.com/photo-1516627145497-ae6968895b74?w=800&q=80",
        "Low-light scene with faces — tests enhancement pipeline",
    ),
    (
        "office_meeting_05.jpg",
        "https://images.unsplash.com/photo-1515187029135-18ee286d815b?w=800&q=80",
        "Office meeting — multiple faces, partial occlusions",
    ),
    (
        "busy_crossing_08.jpg",
        "https://images.unsplash.com/photo-1521295121783-8a321d551ad2?w=800&q=80",
        "Busy city crossing — surveillance angle",
    ),
]


# ---------------------------------------------------------------------------
#  Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, description: str = "") -> bool:
    """
    Download a file from *url* to *dest*.

    Uses urllib with a progress indicator.  Returns True on success.
    """
    if dest.exists():
        logger.info("  ✓ Already exists: %s", dest.name)
        return True

    logger.info("  ↓ Downloading: %s", description or dest.name)
    logger.info("    URL: %s", url[:100] + ("…" if len(url) > 100 else ""))

    try:
        # Add a User-Agent header (some servers reject bare urllib)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            chunk_size = 1024 * 64  # 64 KB chunks
            downloaded = 0

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r    Progress: {pct:5.1f}%  ({downloaded // 1024} KB)", end="", flush=True)

            print()  # newline after progress
        logger.info("  ✓ Saved: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True

    except Exception as e:
        logger.warning("  ✗ Failed to download %s: %s", dest.name, e)
        if dest.exists():
            dest.unlink()
        return False


def convert_to_mp4(input_path: Path, output_path: Path) -> bool:
    """
    Convert a video file to MP4 using ffmpeg.

    Returns True on success.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not found — cannot convert %s", input_path.name)
        return False

    try:
        cmd = [
            ffmpeg, "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=300, check=True)
        logger.info("  ✓ Converted to MP4: %s", output_path.name)
        return True
    except Exception as e:
        logger.warning("  ✗ Conversion failed: %s", e)
        return False


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download all sample datasets."""
    logger.info("=" * 60)
    logger.info("  Sample Dataset Downloader")
    logger.info("=" * 60)

    # Create directories
    for d in [VIDEOS_DIR, IMAGES_DIR, DOWNLOADED_DIR, SAMPLE_INPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Download videos ──────────────────────────────────────────────
    logger.info("\n📹 Downloading sample videos (%d) …", len(SAMPLE_VIDEOS))
    first_video = None
    for filename, url, desc in SAMPLE_VIDEOS:
        dest = DOWNLOADED_DIR / filename
        success = download_file(url, dest, desc)

        if success:
            # Ensure it's MP4
            if dest.suffix.lower() != ".mp4":
                mp4_dest = VIDEOS_DIR / (dest.stem + ".mp4")
                convert_to_mp4(dest, mp4_dest)
            else:
                # Copy to videos dir
                target = VIDEOS_DIR / filename
                if not target.exists():
                    shutil.copy2(dest, target)

            if first_video is None:
                first_video = VIDEOS_DIR / filename

    # ── Download images ──────────────────────────────────────────────
    logger.info("\n🖼️  Downloading sample images (%d) …", len(SAMPLE_IMAGES))
    first_image = None
    for filename, url, desc in SAMPLE_IMAGES:
        dest = DOWNLOADED_DIR / filename
        success = download_file(url, dest, desc)

        if success:
            # Copy to images dir
            target = IMAGES_DIR / filename
            if not target.exists():
                shutil.copy2(dest, target)

            if first_image is None:
                first_image = IMAGES_DIR / filename

    # ── Copy samples to sample_inputs/ ───────────────────────────────
    logger.info("\n📂 Setting up sample_inputs/ …")

    if first_video and first_video.exists():
        sample_video = SAMPLE_INPUTS_DIR / "sample_video.mp4"
        if not sample_video.exists():
            shutil.copy2(first_video, sample_video)
            logger.info("  ✓ Copied %s → sample_inputs/sample_video.mp4", first_video.name)

    if first_image and first_image.exists():
        sample_image = SAMPLE_INPUTS_DIR / "sample_image.jpg"
        if not sample_image.exists():
            shutil.copy2(first_image, sample_image)
            logger.info("  ✓ Copied %s → sample_inputs/sample_image.jpg", first_image.name)

    # ── Summary ──────────────────────────────────────────────────────
    n_videos = len(list(VIDEOS_DIR.glob("*.mp4")))
    n_images = len(list(IMAGES_DIR.iterdir()))

    logger.info("\n" + "=" * 60)
    logger.info("  Download complete!")
    logger.info("  Videos : %d files in datasets/videos/", n_videos)
    logger.info("  Images : %d files in datasets/images/", n_images)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
