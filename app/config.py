"""
Configuration handling for the Local AI Voice Assistant.

All tunable settings (Ollama model/host, TTS voice/language, output
locations, etc.) are read from environment variables, which can be
supplied via a `.env` file in the project root. This keeps the code
free of hard-coded values and makes the assistant easy to reconfigure
without touching any Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when the application configuration is invalid."""


@dataclass(frozen=True)
class Config:
    """Immutable container for all runtime configuration."""

    # --- Ollama (LLM) settings ---
    ollama_model: str
    ollama_host: str
    system_prompt: str

    # --- Pocket TTS settings ---
    tts_voice: str
    tts_language: str

    # --- SenseVoice (STT) settings ---
    stt_model: str
    stt_device: str
    stt_language_code: str

    # --- Audio / output settings ---
    output_dir: Path
    output_filename: str
    sample_rate_override: int | None

    # --- CLI behaviour ---
    exit_commands: tuple[str, ...]

    @staticmethod
    def load(env_file: str | os.PathLike | None = ".env") -> "Config":
        """
        Load configuration from environment variables / a .env file.

        Parameters
        ----------
        env_file:
            Path to a dotenv file to load before reading variables.
            Defaults to a `.env` file in the current working directory.
            Missing files are silently ignored (so the app still runs
            using real environment variables or built-in defaults).
        """
        if env_file:
            load_dotenv(dotenv_path=env_file, override=False)

        output_dir = Path(os.getenv("OUTPUT_DIR", "./output")).expanduser()

        sample_rate_raw = os.getenv("TTS_SAMPLE_RATE_OVERRIDE", "").strip()
        sample_rate_override = int(sample_rate_raw) if sample_rate_raw else None

        exit_commands_raw = os.getenv("EXIT_COMMANDS", "exit,quit,bye,:q")
        exit_commands = tuple(
            cmd.strip().lower() for cmd in exit_commands_raw.split(",") if cmd.strip()
        )

        config = Config(
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2").strip(),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").strip(),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "You are a helpful, concise voice assistant. Keep replies short "
                "and conversational since they will be read aloud.",
            ),
            tts_voice=os.getenv("TTS_VOICE", "alba").strip(),
            tts_language=os.getenv("TTS_LANGUAGE", "english").strip(),
            stt_model=os.getenv("STT_MODEL", "iic/SenseVoiceSmall").strip(),
            stt_device=os.getenv("STT_DEVICE", "cpu").strip(),
            stt_language_code=os.getenv("STT_LANGUAGE_CODE", "auto").strip(),
            output_dir=output_dir,
            output_filename=os.getenv("OUTPUT_FILENAME", "response.wav").strip(),
            sample_rate_override=sample_rate_override,
            exit_commands=exit_commands,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Raise ConfigError if any setting is unusable."""
        if not self.ollama_model:
            raise ConfigError("OLLAMA_MODEL must not be empty.")
        if not self.ollama_host:
            raise ConfigError("OLLAMA_HOST must not be empty.")
        if not self.tts_voice:
            raise ConfigError("TTS_VOICE must not be empty.")
        if not self.tts_language:
            raise ConfigError("TTS_LANGUAGE must not be empty.")
        if not self.output_filename.lower().endswith(".wav"):
            raise ConfigError("OUTPUT_FILENAME must end with '.wav'.")
        if not self.exit_commands:
            raise ConfigError("EXIT_COMMANDS must contain at least one command.")

    @property
    def output_path(self) -> Path:
        """Full path to the WAV file that gets (re)written each turn."""
        return self.output_dir / self.output_filename
