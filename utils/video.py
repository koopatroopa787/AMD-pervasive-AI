import cv2
import os
def extract_frames(video_path, max_duration=15):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(max_duration * fps)
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path, fps=30):
    if os.path.exists(output_path):
        os.remove(output_path)
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()
