"""
Video upscaling utilities with secure subprocess handling.
"""
import os
import cv2
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import subprocess
import logging
from pathlib import Path
from typing import Union, Optional, List

logger = logging.getLogger(__name__)


def process_frame(frame: np.ndarray, model) -> np.ndarray:
    """
    Process a single frame with upscaling model.

    Args:
        frame: Input frame as numpy array (BGR format)
        model: Upscaling model

    Returns:
        Upscaled frame as numpy array (BGR format)
    """
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        sr_image = model.predict(image)
        sr_frame = cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)
        return sr_frame
    except Exception as e:
        logger.error(f"Failed to process frame: {e}")
        raise


def upscale_video(
    input_video_path: Union[str, Path],
    output_video_path: Union[str, Path],
    model,
    max_duration: Optional[int] = 15,
    max_workers: int = 4
) -> None:
    """
    Upscale video using AI model.

    Args:
        input_video_path: Path to input video
        output_video_path: Path for output video
        model: Upscaling model
        max_duration: Maximum duration in seconds (None = no limit)
        max_workers: Number of parallel workers for frame processing

    Raises:
        FileNotFoundError: If input video doesn't exist
        RuntimeError: If upscaling fails
    """
    from utils.video import extract_frames, save_video

    input_video_path = Path(input_video_path)

    if not input_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_video_path}")

    logger.info(f"Starting video upscaling: {input_video_path.name}")
    logger.info(f"Using {max_workers} workers for frame processing")

    try:
        # Extract frames
        frames = extract_frames(input_video_path, max_duration=max_duration)
        logger.info(f"Extracted {len(frames)} frames")

        # Upscale frames in parallel
        upscaled_frames = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            upscaled_frames = list(
                tqdm(
                    executor.map(process_frame, frames, [model] * len(frames)),
                    total=len(frames),
                    desc="Upscaling frames",
                    unit="frame"
                )
            )

        # Save upscaled video
        save_video(upscaled_frames, output_video_path)
        logger.info(f"Video upscaled successfully: {output_video_path}")

    except Exception as e:
        logger.error(f"Video upscaling failed: {e}")
        raise RuntimeError(f"Video upscaling failed: {e}")


def convert_to_h264(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    crf: int = 23,
    preset: str = "medium"
) -> None:
    """
    Convert video to H.264 format for better compatibility.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        crf: Constant Rate Factor (0-51, lower = better quality, default: 23)
        preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast,
                medium, slow, slower, veryslow, default: medium)

    Raises:
        RuntimeError: If FFmpeg conversion fails
        FileNotFoundError: If input video doesn't exist
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    # Remove existing output file
    if output_path.exists():
        output_path.unlink()
        logger.debug(f"Removed existing video file: {output_path}")

    # Use list format for secure subprocess (prevents shell injection)
    command = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
        str(output_path)
    ]

    logger.info(f"Converting video to H.264 (crf={crf}, preset={preset})")

    try:
        result = subprocess.run(
            command,
            shell=False,  # Secure: no shell interpretation
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg H.264 conversion failed: {result.stderr}")
            raise RuntimeError(f"H.264 conversion failed: {result.stderr}")

        logger.info(f"Video converted successfully to {output_path.name}")

    except subprocess.TimeoutExpired:
        logger.error("H.264 conversion timed out after 10 minutes")
        raise RuntimeError("H.264 conversion timed out")
    except Exception as e:
        logger.error(f"Unexpected error during H.264 conversion: {e}")
        raise
