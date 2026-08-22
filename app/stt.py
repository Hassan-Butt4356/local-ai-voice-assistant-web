"""
Speech-to-text using FunAudioLLM's SenseVoice model (via the `funasr`
package). Loaded once and reused, same pattern as PocketTTSEngine.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class STTError(Exception):
    """Raised when speech-to-text transcription fails."""


class SenseVoiceEngine:
    """Loads SenseVoice once, then transcribes WAV/MP3 files to text."""

    def __init__(self, model_id: str, device: str = "cpu", language: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.language = language
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise STTError(
                "The 'funasr' package is not installed. Install it with "
                "`pip install funasr`."
            ) from exc

        try:
            logger.info("Loading SenseVoice model '%s' on %s...", self.model_id, self.device)
            self._model = AutoModel(
                model=self.model_id,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
                disable_update=True,
            )
        except Exception as exc:
            raise STTError(f"Failed to load SenseVoice model: {exc}") from exc
        logger.info("SenseVoice ready.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe an audio file (any common format) to plain text."""
        if not self.is_loaded:
            raise STTError("STT engine not loaded yet; call load() first.")
        if not audio_path.exists():
            raise STTError(f"Audio file not found: {audio_path}")

        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as exc:
            raise STTError("Failed to import funasr postprocess utils.") from exc

        try:
            result = self._model.generate(
                input=str(audio_path),
                cache={},
                language=self.language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
            )
        except Exception as exc:
            raise STTError(f"Transcription failed: {exc}") from exc

        if not result:
            raise STTError("SenseVoice returned no result.")

        text = rich_transcription_postprocess(result[0]["text"]).strip()
        if not text:
            raise STTError("Transcription produced empty text (silence or unclear audio?).")
        return text
