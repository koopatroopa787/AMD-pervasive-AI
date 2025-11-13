"""
Audio transcription utilities using Vosk speech recognition.
"""
import wave
import json
import logging
from pathlib import Path
from typing import Union
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)


def transcribe_audio_vosk(audio_path: Union[str, Path]) -> str:
    """
    Transcribe audio file using Vosk offline speech recognition.

    Args:
        audio_path: Path to audio file (WAV format, 16kHz, mono)

    Returns:
        Transcribed text

    Raises:
        FileNotFoundError: If audio file or model doesn't exist
        RuntimeError: If transcription fails
    """
    from config import get_config

    audio_path = Path(audio_path)

    # Validate audio file
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Get model path from config
    config = get_config()
    model_path = config.paths.vosk_model_path

    # Validate model path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Vosk model not found at: {model_path}\n"
            f"Download from: https://alphacephei.com/vosk/models\n"
            f"Extract to: {model_path}"
        )

    try:
        logger.info(f"Loading Vosk model from: {model_path}")
        model = Model(str(model_path))

        logger.info(f"Opening audio file: {audio_path.name}")
        wf = wave.open(str(audio_path), "rb")

        # Validate audio format
        if wf.getnchannels() != 1:
            wf.close()
            raise RuntimeError("Audio must be mono (1 channel)")

        sample_rate = wf.getframerate()
        logger.info(f"Audio sample rate: {sample_rate} Hz")

        rec = KaldiRecognizer(model, sample_rate)

        # Transcribe audio
        logger.info("Starting transcription")
        transcript = ""
        frame_count = 0

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            frame_count += 1
            if rec.AcceptWaveform(data):
                result = rec.Result()
                text = json.loads(result).get('text', '')
                transcript += text + " "

                if frame_count % 100 == 0:
                    logger.debug(f"Processed {frame_count} frames, {len(transcript)} chars")

        # Get final result
        final_result = rec.FinalResult()
        text = json.loads(final_result).get('text', '')
        transcript += text

        wf.close()

        transcript = transcript.strip()
        logger.info(f"Transcription complete: {len(transcript)} characters, {len(transcript.split())} words")

        return transcript

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")
