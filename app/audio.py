"""
Local WAV playback.

Uses `sounddevice` + `soundfile`, which are lightweight, cross-platform
(Windows/macOS/Linux), and require no external system player -- a good
fit for a fully local, offline voice assistant.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioPlaybackError(Exception):
    """Raised when a generated WAV file cannot be played."""


def play_wav(path: Path) -> None:
    """Play a WAV file from disk and block until playback finishes."""
    if not path.exists():
        raise AudioPlaybackError(f"Audio file not found: {path}")

    try:
        import soundfile as sf
        import sounddevice as sd
    except ImportError as exc:
        raise AudioPlaybackError(
            "The 'sounddevice' and 'soundfile' packages are required for "
            "playback. Install them with `pip install sounddevice soundfile`."
        ) from exc

    try:
        data, samplerate = sf.read(str(path), dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
    except Exception as exc:
        raise AudioPlaybackError(f"Failed to play '{path}': {exc}") from exc
