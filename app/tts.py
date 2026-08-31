"""
Wrapper around Kyutai's Pocket TTS Python API.

Reference: https://github.com/kyutai-labs/pocket-tts

The model and the chosen voice's "state" are both expensive-ish to
load, so this module loads them exactly once (in `PocketTTSEngine.load`)
and reuses them for every subsequent utterance.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Raised when speech synthesis fails."""


class PocketTTSEngine:
    """
    Loads a Pocket TTS model + voice once, then converts text to WAV
    files on demand.
    """

    def __init__(self, voice: str, language: str = "english") -> None:
        self.voice = voice
        self.language = language
        self._model = None
        self._voice_state = None
        self._sample_rate: int | None = None

    def load(self) -> None:
        """
        Load the Pocket TTS model and prepare the configured voice.
        Must be called once before `synthesize()`. Safe to call again
        (it's a no-op) if already loaded.
        """
        if self._model is not None:
            return

        try:
            from pocket_tts import TTSModel
        except ImportError as exc:
            raise TTSError(
                "The 'pocket-tts' package is not installed. Install it with "
                "`pip install pocket-tts` (see "
                "https://github.com/kyutai-labs/pocket-tts)."
            ) from exc

        try:
            logger.info("Loading Pocket TTS model (language=%s)...", self.language)
            self._model = TTSModel.load_model(language=self.language)
        except TypeError:
            # Older/newer versions of the API may not accept `language`
            # as a kwarg on load_model(); fall back to the no-arg form.
            try:
                self._model = TTSModel.load_model()
            except Exception as exc:
                raise TTSError(f"Failed to load Pocket TTS model: {exc}") from exc
        except Exception as exc:
            raise TTSError(f"Failed to load Pocket TTS model: {exc}") from exc

        try:
            logger.info("Preparing voice '%s'...", self.voice)
            self._voice_state = self._model.get_state_for_audio_prompt(self.voice)
        except Exception as exc:
            raise TTSError(
                f"Failed to load voice '{self.voice}'. Make sure it's a valid "
                f"preset name, a local WAV path, or an 'hf://...' voice "
                f"reference. Original error: {exc}"
            ) from exc

        self._sample_rate = self._model.sample_rate
        logger.info("Pocket TTS ready (sample rate: %s Hz).", self._sample_rate)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._voice_state is not None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise TTSError("TTS engine not loaded yet; call load() first.")
        return self._sample_rate

    def synthesize(self, text: str, output_path: Path) -> Path:
        """
        Convert `text` to speech and save it as a WAV file at
        `output_path`. Returns the path for convenience.
        """
        if not self.is_loaded:
            raise TTSError("TTS engine not loaded yet; call load() first.")
        if not text or not text.strip():
            raise TTSError("Cannot synthesize empty text.")

        try:
            import scipy.io.wavfile
        except ImportError as exc:
            raise TTSError(
                "The 'scipy' package is required to write WAV files. "
                "Install it with `pip install scipy`."
            ) from exc

        try:
            audio = self._model.generate_audio(self._voice_state, text)
        except Exception as exc:
            raise TTSError(f"Speech generation failed: {exc}") from exc

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            scipy.io.wavfile.write(str(output_path), self.sample_rate, audio.numpy())
        except Exception as exc:
            raise TTSError(f"Failed to write WAV file to '{output_path}': {exc}") from exc

        return output_path


    def clone_voice_from_file(self, audio_path: Path, voice_name: str, voices_dir: Path) -> Path:
        """
        Process a recorded/uploaded audio sample into a reusable voice
        embedding (safetensors) and save it under `voices_dir`. Returns
        the path to the saved embedding.

        Uses Pocket TTS's official cloning flow:
            get_state_for_audio_prompt(audio_path) -> export_model_state(...)
        """
        if not self.is_loaded:
            raise TTSError("TTS engine not loaded yet; call load() first.")
        if not audio_path.exists():
            raise TTSError(f"Audio file not found: {audio_path}")

        try:
            from pocket_tts import export_model_state
        except ImportError as exc:
            raise TTSError(
                "This version of pocket-tts does not expose export_model_state; "
                "upgrade with `pip install -U pocket-tts`."
            ) from exc

        try:
            logger.info("Processing voice sample '%s' for cloning...", audio_path.name)
            voice_state = self._model.get_state_for_audio_prompt(str(audio_path))
        except Exception as exc:
            raise TTSError(
                f"Failed to process voice sample. Make sure it's a clear, "
                f"single-speaker recording of a few seconds. Original error: {exc}"
            ) from exc

        voices_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c for c in voice_name if c.isalnum() or c in ("-", "_")).strip()
        safe_name = safe_name or "custom_voice"
        dest = voices_dir / f"{safe_name}.safetensors"

        try:
            export_model_state(voice_state, str(dest))
        except Exception as exc:
            raise TTSError(f"Failed to save cloned voice: {exc}") from exc

        logger.info("Cloned voice saved to '%s'.", dest)
        return dest

    def set_active_voice(self, voice_ref: str) -> None:
        """
        Switch the currently active voice used by synthesize(). `voice_ref`
        can be a preset name (e.g. "alba"), a local .wav path, or a local
        .safetensors path (a previously cloned/exported voice).
        """
        if not self.is_loaded:
            raise TTSError("TTS engine not loaded yet; call load() first.")

        try:
            self._voice_state = self._model.get_state_for_audio_prompt(voice_ref)
            self.voice = voice_ref
        except Exception as exc:
            raise TTSError(f"Failed to switch to voice '{voice_ref}': {exc}") from exc