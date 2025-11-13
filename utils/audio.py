"""
Audio processing utilities with secure subprocess handling.
"""
import os
import subprocess
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def extract_audio(video_path: Union[str, Path], output_audio_path: Union[str, Path]) -> None:
    """
    Extract audio from video file using FFmpeg.

    Args:
        video_path: Path to input video file
        output_audio_path: Path for output audio file

    Raises:
        RuntimeError: If FFmpeg extraction fails
        FileNotFoundError: If video file doesn't exist
    """
    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)

    # Validate input
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Remove existing output file
    if output_audio_path.exists():
        output_audio_path.unlink()
        logger.debug(f"Removed existing audio file: {output_audio_path}")

    # Use list format for secure subprocess (prevents shell injection)
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:a", "0",
        "-map", "a",
        str(output_audio_path)
    ]

    logger.info(f"Extracting audio from {video_path.name}")

    try:
        result = subprocess.run(
            command,
            shell=False,  # Secure: no shell interpretation
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")

        logger.info(f"Audio extracted successfully to {output_audio_path.name}")

    except subprocess.TimeoutExpired:
        logger.error("Audio extraction timed out after 5 minutes")
        raise RuntimeError("Audio extraction timed out")
    except Exception as e:
        logger.error(f"Unexpected error during audio extraction: {e}")
        raise


def convert_audio_format(
    input_audio_path: Union[str, Path],
    output_audio_path: Union[str, Path],
    sample_rate: int = 16000,
    channels: int = 1
) -> None:
    """
    Convert audio format using FFmpeg.

    Args:
        input_audio_path: Path to input audio file
        output_audio_path: Path for output audio file
        sample_rate: Target sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1 for mono)

    Raises:
        RuntimeError: If FFmpeg conversion fails
        FileNotFoundError: If input audio file doesn't exist
    """
    input_audio_path = Path(input_audio_path)
    output_audio_path = Path(output_audio_path)

    # Validate input
    if not input_audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_audio_path}")

    # Remove existing output file
    if output_audio_path.exists():
        output_audio_path.unlink()
        logger.debug(f"Removed existing audio file: {output_audio_path}")

    # Use list format for secure subprocess
    command = [
        "ffmpeg",
        "-i", str(input_audio_path),
        "-ac", str(channels),
        "-ar", str(sample_rate),
        str(output_audio_path)
    ]

    logger.info(f"Converting audio format: {channels}ch @ {sample_rate}Hz")

    try:
        result = subprocess.run(
            command,
            shell=False,  # Secure: no shell interpretation
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg audio conversion failed: {result.stderr}")
            raise RuntimeError(f"Audio conversion failed: {result.stderr}")

        logger.info(f"Audio converted successfully to {output_audio_path.name}")

    except subprocess.TimeoutExpired:
        logger.error("Audio conversion timed out after 5 minutes")
        raise RuntimeError("Audio conversion timed out")
    except Exception as e:
        logger.error(f"Unexpected error during audio conversion: {e}")
        raise
