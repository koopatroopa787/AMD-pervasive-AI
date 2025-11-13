"""
Input validation and sanitization utilities.
"""
import re
import logging
from pathlib import Path
from typing import Union, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_valid_youtube_url(url: str) -> bool:
    """
    Validate YouTube URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid YouTube URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False

    youtube_patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtu\.be/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/embed/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/v/[\w-]+',
    ]

    return any(re.match(pattern, url) for pattern in youtube_patterns)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename to remove dangerous characters.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')

    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_file"

    # Truncate if too long
    if len(sanitized) > max_length:
        # Preserve extension if possible
        parts = sanitized.rsplit('.', 1)
        if len(parts) == 2:
            name, ext = parts
            max_name_len = max_length - len(ext) - 1
            sanitized = name[:max_name_len] + '.' + ext
        else:
            sanitized = sanitized[:max_length]

    return sanitized


def validate_file_size(file_path: Union[str, Path], max_size_mb: int = 500) -> tuple[bool, Optional[str]]:
    """
    Validate file size.

    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB

    Returns:
        (is_valid, error_message)
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            return False, f"File not found: {file_path}"

        size_mb = file_path.stat().st_size / (1024 * 1024)

        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"

        return True, None

    except Exception as e:
        return False, f"Error checking file size: {e}"


def validate_video_file(file_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
    """
    Validate video file format.

    Args:
        file_path: Path to video file

    Returns:
        (is_valid, error_message)
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        return False, "File not found"

    # Check extension
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'}
    if file_path.suffix.lower() not in allowed_extensions:
        return False, f"Invalid video format. Allowed: {', '.join(allowed_extensions)}"

    # Check file size (default max 500MB)
    is_valid, error = validate_file_size(file_path, max_size_mb=500)
    if not is_valid:
        return False, error

    return True, None


def validate_audio_file(file_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
    """
    Validate audio file format.

    Args:
        file_path: Path to audio file

    Returns:
        (is_valid, error_message)
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        return False, "File not found"

    # Check extension
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
    if file_path.suffix.lower() not in allowed_extensions:
        return False, f"Invalid audio format. Allowed: {', '.join(allowed_extensions)}"

    # Check file size (default max 100MB)
    is_valid, error = validate_file_size(file_path, max_size_mb=100)
    if not is_valid:
        return False, error

    return True, None


def validate_language_code(lang_code: str, supported_languages: list) -> tuple[bool, Optional[str]]:
    """
    Validate language code.

    Args:
        lang_code: Language code to validate
        supported_languages: List of supported language codes

    Returns:
        (is_valid, error_message)
    """
    if not lang_code:
        return False, "Language code is required"

    if lang_code not in supported_languages:
        return False, f"Unsupported language: {lang_code}. Supported: {', '.join(supported_languages)}"

    return True, None


def sanitize_text_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text input.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace('\x00', '')

    # Normalize whitespace
    text = ' '.join(text.split())

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")

    return text


def validate_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL is required"

    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False, "Invalid URL format"

        if result.scheme not in ['http', 'https']:
            return False, "Only HTTP/HTTPS URLs are allowed"

        return True, None

    except Exception as e:
        return False, f"Invalid URL: {e}"
