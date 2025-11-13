"""
Video upscaling utilities with secure subprocess handling and enhanced quality controls.
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
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VideoCodec(Enum):
    """Supported video codecs"""
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"


class QualityPreset(Enum):
    """Video quality presets with predefined settings"""
    ULTRA = "ultra"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CUSTOM = "custom"


@dataclass
class VideoEncodingSettings:
    """Video encoding settings"""
    codec: VideoCodec = VideoCodec.H264
    crf: int = 23
    preset: str = "medium"
    video_bitrate: Optional[str] = None  # e.g., "5M", "10M"
    audio_bitrate: str = "192k"
    audio_codec: str = "aac"
    pixel_format: str = "yuv420p"

    @classmethod
    def from_quality_preset(cls, preset: QualityPreset, codec: VideoCodec = VideoCodec.H264) -> 'VideoEncodingSettings':
        """
        Create encoding settings from quality preset.

        Args:
            preset: Quality preset
            codec: Video codec to use

        Returns:
            VideoEncodingSettings instance
        """
        preset_configs = {
            QualityPreset.ULTRA: {
                VideoCodec.H264: {"crf": 18, "preset": "slow", "video_bitrate": "20M", "audio_bitrate": "320k"},
                VideoCodec.H265: {"crf": 22, "preset": "slow", "video_bitrate": "15M", "audio_bitrate": "320k"},
            },
            QualityPreset.HIGH: {
                VideoCodec.H264: {"crf": 20, "preset": "medium", "video_bitrate": "10M", "audio_bitrate": "256k"},
                VideoCodec.H265: {"crf": 24, "preset": "medium", "video_bitrate": "8M", "audio_bitrate": "256k"},
            },
            QualityPreset.MEDIUM: {
                VideoCodec.H264: {"crf": 23, "preset": "medium", "video_bitrate": "5M", "audio_bitrate": "192k"},
                VideoCodec.H265: {"crf": 28, "preset": "medium", "video_bitrate": "4M", "audio_bitrate": "192k"},
            },
            QualityPreset.LOW: {
                VideoCodec.H264: {"crf": 28, "preset": "fast", "video_bitrate": "2M", "audio_bitrate": "128k"},
                VideoCodec.H265: {"crf": 32, "preset": "fast", "video_bitrate": "1.5M", "audio_bitrate": "128k"},
            },
        }

        config = preset_configs.get(preset, {}).get(codec, preset_configs[QualityPreset.MEDIUM][VideoCodec.H264])

        return cls(
            codec=codec,
            crf=config["crf"],
            preset=config["preset"],
            video_bitrate=config["video_bitrate"],
            audio_bitrate=config["audio_bitrate"]
        )


def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get video information using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    import json

    video_path = Path(video_path)

    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)

            # Extract video stream info
            video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})

            return {
                'duration': float(data.get('format', {}).get('duration', 0)),
                'size_bytes': int(data.get('format', {}).get('size', 0)),
                'bitrate': int(data.get('format', {}).get('bit_rate', 0)),
                'width': video_stream.get('width', 0),
                'height': video_stream.get('height', 0),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'codec': video_stream.get('codec_name', 'unknown')
            }
    except Exception as e:
        logger.warning(f"Failed to get video info: {e}")

    return {}


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


def encode_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    settings: Optional[VideoEncodingSettings] = None,
    quality_preset: Optional[QualityPreset] = None,
    copy_audio: bool = False
) -> Dict[str, Any]:
    """
    Encode video with enhanced quality and bitrate controls.

    Args:
        input_path: Path to input video
        output_path: Path for output video
        settings: Custom encoding settings (overrides quality_preset)
        quality_preset: Quality preset to use (ignored if settings provided)
        copy_audio: Copy audio stream without re-encoding

    Returns:
        Dictionary with encoding statistics

    Raises:
        RuntimeError: If FFmpeg encoding fails
        FileNotFoundError: If input video doesn't exist
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    # Get input video info
    input_info = get_video_info(input_path)

    # Use settings or create from preset
    if settings is None:
        preset = quality_preset or QualityPreset.MEDIUM
        settings = VideoEncodingSettings.from_quality_preset(preset)

    # Remove existing output file
    if output_path.exists():
        output_path.unlink()
        logger.debug(f"Removed existing video file: {output_path}")

    # Build FFmpeg command
    command = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", settings.codec.value,
        "-pix_fmt", settings.pixel_format,
    ]

    # Add CRF or bitrate
    if settings.video_bitrate:
        command.extend(["-b:v", settings.video_bitrate])
        command.extend(["-maxrate", settings.video_bitrate])
        command.extend(["-bufsize", f"{int(settings.video_bitrate.rstrip('MmKk')) * 2}M"])
    else:
        command.extend(["-crf", str(settings.crf)])

    # Add preset (only for x264/x265)
    if settings.codec in [VideoCodec.H264, VideoCodec.H265]:
        command.extend(["-preset", settings.preset])

    # Audio encoding
    if copy_audio:
        command.extend(["-c:a", "copy"])
    else:
        command.extend([
            "-c:a", settings.audio_codec,
            "-b:a", settings.audio_bitrate
        ])

    # Codec-specific optimizations
    if settings.codec == VideoCodec.H264:
        command.extend([
            "-profile:v", "high",
            "-level", "4.1",
            "-movflags", "+faststart"  # Enable streaming
        ])
    elif settings.codec == VideoCodec.H265:
        command.extend([
            "-tag:v", "hvc1",  # Better compatibility
            "-movflags", "+faststart"
        ])

    command.append(str(output_path))

    # Log encoding info
    codec_name = settings.codec.name
    quality_info = f"bitrate={settings.video_bitrate}" if settings.video_bitrate else f"crf={settings.crf}"
    logger.info(f"Encoding video: {codec_name}, {quality_info}, preset={settings.preset}")
    logger.info(f"Audio: {settings.audio_codec} @ {settings.audio_bitrate}")

    try:
        import time
        start_time = time.time()

        result = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout for large files
        )

        encoding_time = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"FFmpeg encoding failed: {result.stderr}")
            raise RuntimeError(f"Video encoding failed: {result.stderr}")

        # Get output video info
        output_info = get_video_info(output_path)

        # Calculate statistics
        stats = {
            'encoding_time': encoding_time,
            'input_size_mb': input_info.get('size_bytes', 0) / (1024 * 1024),
            'output_size_mb': output_info.get('size_bytes', 0) / (1024 * 1024),
            'compression_ratio': 0,
            'input_bitrate': input_info.get('bitrate', 0),
            'output_bitrate': output_info.get('bitrate', 0),
            'resolution': f"{output_info.get('width', 0)}x{output_info.get('height', 0)}"
        }

        if stats['input_size_mb'] > 0:
            stats['compression_ratio'] = stats['output_size_mb'] / stats['input_size_mb']

        logger.info(f"Encoding complete in {encoding_time:.1f}s")
        logger.info(f"Size: {stats['input_size_mb']:.1f}MB → {stats['output_size_mb']:.1f}MB "
                   f"({stats['compression_ratio']:.2%} of original)")

        return stats

    except subprocess.TimeoutExpired:
        logger.error("Video encoding timed out after 30 minutes")
        raise RuntimeError("Video encoding timed out")
    except Exception as e:
        logger.error(f"Unexpected error during video encoding: {e}")
        raise


def convert_to_h264(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    crf: int = 23,
    preset: str = "medium"
) -> None:
    """
    Convert video to H.264 format (legacy function for backward compatibility).

    Args:
        input_path: Path to input video
        output_path: Path for output video
        crf: Constant Rate Factor (0-51, lower = better quality, default: 23)
        preset: Encoding preset

    Raises:
        RuntimeError: If FFmpeg conversion fails
        FileNotFoundError: If input video doesn't exist
    """
    settings = VideoEncodingSettings(
        codec=VideoCodec.H264,
        crf=crf,
        preset=preset
    )
    encode_video(input_path, output_path, settings=settings)
