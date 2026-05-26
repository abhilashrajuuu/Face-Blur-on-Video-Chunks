# =============================================================================
#  Dockerfile — Face Blur on Surveillance Videos & Images
# =============================================================================
#  Build:
#      docker build -t face-blur .
#
#  Run (process a video):
#      docker run --rm -v $(pwd)/sample_inputs:/data/input \
#                       -v $(pwd)/outputs:/data/output \
#                       face-blur --video /data/input/sample_video.mp4 \
#                                 --output-dir /data/output
#
#  Run (Streamlit web UI):
#      docker run --rm -p 8501:8501 face-blur streamlit run streamlit_app.py
# =============================================================================

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ──────────────────────────────────────────────────────
COPY . .

# ── Create output directories ───────────────────────────────────────────────
RUN mkdir -p datasets/videos datasets/images datasets/downloaded \
             outputs/blurred_videos outputs/blurred_images outputs/debug \
             sample_inputs

# ── Default entrypoint ──────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]

# ── Expose Streamlit port (for web UI mode) ──────────────────────────────────
EXPOSE 8501
