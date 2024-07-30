import os
import subprocess

def extract_audio(video_path, output_audio_path):
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio_path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")

def convert_audio_format(input_audio_path, output_audio_path):
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)
    command = f'ffmpeg -i "{input_audio_path}" -ac 1 -ar 16000 "{output_audio_path}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")
