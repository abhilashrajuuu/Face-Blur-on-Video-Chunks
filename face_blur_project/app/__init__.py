"""
Face Blur on Surveillance Videos & Images
==========================================

A production-grade pipeline for detecting and anonymizing human faces
in surveillance-style videos and images using deep learning.

Components:
    - detector:         InsightFace/RetinaFace-based face detection
    - tracker:          ByteTrack multi-object tracking via supervision
    - blur:             Gaussian and pixelation blur modes
    - enhance:          Frame preprocessing (denoise, CLAHE, gamma)
    - video_processor:  End-to-end video processing pipeline
    - image_processor:  End-to-end image processing pipeline
    - config:           Central configuration
    - utils:            Shared utility functions
"""

__version__ = "1.0.0"
__author__ = "Face Blur Project"
