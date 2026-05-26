"""
streamlit_app.py — Web UI for Face Blur Pipeline
==================================================

A Streamlit-based web interface for interactive face blurring.

Features:
  • Upload a video or image via drag-and-drop
  • Choose blur mode (Gaussian / Pixelate) from the sidebar
  • Adjust confidence threshold with a slider
  • Toggle debug visualisation
  • View results inline
  • Download the processed output

Run with:
    streamlit run streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np  # used for bytearray conversion

from app.config import Config
from app.utils import ensure_dirs


# ---------------------------------------------------------------------------
#  Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Face Blur — Surveillance Privacy Tool",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
#  Sidebar — settings
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

blur_mode = st.sidebar.selectbox(
    "Blur Mode",
    ["gaussian", "pixelate"],
    index=0,
    help="Gaussian applies a smooth blur; Pixelate creates a mosaic effect.",
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (HIGH tier)",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="Detections above this score are definitely blurred.",
)

debug_mode = st.sidebar.checkbox(
    "Show Debug Overlay",
    value=False,
    help="Display detection boxes, confidence scores, and tracker IDs.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tech Stack:** InsightFace · ByteTrack · OpenCV\n\n"
    "Built with ❤️ using Streamlit"
)


# ---------------------------------------------------------------------------
#  Build config from sidebar values
# ---------------------------------------------------------------------------

@st.cache_resource
def get_config(blur_mode_val, confidence_val, debug_val):
    """Build a Config instance from sidebar parameters."""
    config = Config(
        blur_mode=blur_mode_val,
        confidence_high=confidence_val,
        debug=debug_val,
    )
    return config


# ---------------------------------------------------------------------------
#  Main UI
# ---------------------------------------------------------------------------

st.title("🔒 Face Blur — Surveillance Privacy Tool")
st.markdown(
    "Upload a **video** or **image** to automatically detect and blur "
    "every visible human face using deep-learning-based detection."
)

tab_image, tab_video = st.tabs(["🖼️ Image", "📹 Video"])


# ---------------------------------------------------------------------------
#  Image tab
# ---------------------------------------------------------------------------

with tab_image:
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        # Read uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if input_img is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Process
            with st.spinner("Detecting and blurring faces …"):
                config = Config(
                    blur_mode=blur_mode,
                    confidence_high=confidence_threshold,
                    debug=debug_mode,
                )

                from app.enhance import FrameEnhancer
                from app.detector import FaceDetector
                from app.blur import FaceBlurrer
                from app.utils import draw_debug_frame

                enhancer = FrameEnhancer(config)
                detector = FaceDetector(config)
                blurrer = FaceBlurrer(config)

                enhanced = enhancer.enhance(input_img)
                detections = detector.detect(enhanced)

                # For images, blur HIGH and MEDIUM
                faces_to_blur = [d for d in detections if d.tier in ("high", "medium")]
                blurred = blurrer.blur_faces(input_img.copy(), faces_to_blur)

            with col2:
                st.subheader(f"Blurred ({len(faces_to_blur)} faces)")
                st.image(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Debug overlay
            if debug_mode and detections:
                st.subheader("🔍 Debug: Detection Overlay")
                bboxes = [d.bbox for d in detections]
                confs = [d.confidence for d in detections]
                debug_frame = draw_debug_frame(input_img, bboxes, confs)
                st.image(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Download button
            success, encoded = cv2.imencode(".jpg", blurred)
            if success:
                st.download_button(
                    label="⬇️ Download Blurred Image",
                    data=encoded.tobytes(),
                    file_name=f"blurred_{uploaded_image.name}",
                    mime="image/jpeg",
                )

            # Detection stats
            st.markdown("### Detection Summary")
            st.markdown(
                f"- **Total detections:** {len(detections)}\n"
                f"- **High confidence (>{confidence_threshold:.0%}):** "
                f"{sum(1 for d in detections if d.tier == 'high')}\n"
                f"- **Medium confidence:** "
                f"{sum(1 for d in detections if d.tier == 'medium')}\n"
                f"- **Low confidence (ignored):** "
                f"{sum(1 for d in detections if d.tier == 'low')}\n"
            )
        else:
            st.error("Failed to decode uploaded image.")


# ---------------------------------------------------------------------------
#  Video tab
# ---------------------------------------------------------------------------

with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mkv", "mov"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🚀 Process Video", key="process_video_btn"):
            # Save uploaded video to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            with st.spinner("Processing video — this may take a while …"):
                config = Config(
                    blur_mode=blur_mode,
                    confidence_high=confidence_threshold,
                    debug=debug_mode,
                )

                from app.video_processor import VideoProcessor

                # Use a temp output dir
                output_dir = Path(tempfile.mkdtemp())
                processor = VideoProcessor(config)
                result = processor.process_video(tmp_path, str(output_dir))

            if result:
                st.success("✓ Video processed successfully!")
                st.video(result)

                # Download button
                with open(result, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Blurred Video (MP4)",
                        data=f.read(),
                        file_name=f"blurred_{uploaded_video.name}",
                        mime="video/mp4",
                    )
            else:
                st.error("✗ Video processing failed. Check the logs.")
