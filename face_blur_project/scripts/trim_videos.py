# pyrefly: ignore [missing-import]
import cv2
from pathlib import Path

def trim_videos_opencv():
    videos_dir = Path(__file__).resolve().parent.parent / "datasets" / "videos"
    
    for video_file in videos_dir.glob("*.mp4"):
        if "_trimmed" in video_file.name:
            continue
            
        temp_out = video_file.with_name(video_file.stem + "_trimmed.mp4")
        
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Could not open {video_file.name}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate exactly 10 seconds worth of frames
        max_frames = int(fps * 10)
        
        if total_frames <= max_frames:
            print(f"Skipping {video_file.name} (already <= 10s)")
            cap.release()
            continue
            
        print(f"Trimming {video_file.name} to 10 seconds ({max_frames} frames)...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(temp_out), fourcc, fps, (width, height))
        
        frames_written = 0
        while frames_written < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1
            
        cap.release()
        writer.release()
        
        if temp_out.exists():
            temp_out.replace(video_file)
            print(f"  Saved as {video_file.name}")

if __name__ == "__main__":
    trim_videos_opencv()
