import os
import cv2
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import subprocess

def process_frame(frame, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    sr_image = model.predict(image)
    sr_frame = cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)
    return sr_frame

def upscale_video(input_video_path, output_video_path, model):
    from utils.video import extract_frames, save_video

    frames = extract_frames(input_video_path, max_duration=15)
    upscaled_frames = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        upscaled_frames = list(tqdm(executor.map(process_frame, frames, [model]*len(frames)), total=len(frames), desc="Upscaling frames"))

    save_video(upscaled_frames, output_video_path)

def convert_to_h264(input_path, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    command = f'ffmpeg -i "{input_path}" -c:v libx264 -pix_fmt yuv420p "{output_path}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")
