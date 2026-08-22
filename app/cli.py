"""
Command-line interface for the Local AI Voice Assistant.

Workflow per turn:
    user text -> Ollama (LLM reply) -> Pocket TTS (WAV) -> playback
"""

from __future__ import annotations

import logging
import sys

from app.audio import AudioPlaybackError, play_wav
from app.config import Config, ConfigError
from app.llm import LLMError, OllamaClient
from app.tts import PocketTTSEngine, TTSError

logger = logging.getLogger(__name__)

BANNER = """\
============================================
  Local AI Voice Assistant (Ollama + Pocket TTS)
============================================
Type your message and press Enter.
Type 'exit' (or 'quit' / 'bye') to stop.
"""


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep third-party libraries quiet unless something goes wrong.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def run() -> int:
    """
    Run the interactive CLI loop.

    Returns a process exit code (0 = success, non-zero = startup failure).
    """
    _configure_logging()
    print(BANNER)

    try:
        config = Config.load()
    except ConfigError as exc:
        print(f"[Config error] {exc}")
        return 1

    llm = OllamaClient(
        model=config.ollama_model,
        host=config.ollama_host,
        system_prompt=config.system_prompt,
    )
    tts = PocketTTSEngine(voice=config.tts_voice, language=config.tts_language)

    print(f"Connecting to Ollama at {config.ollama_host} (model: {config.ollama_model})...")
    try:
        llm.check_connection()
    except LLMError as exc:
        print(f"[Ollama error] {exc}")
        return 1
    print("Ollama is ready.")

    print(f"Loading Pocket TTS (voice: {config.tts_voice}, language: {config.tts_language})...")
    try:
        tts.load()
    except TTSError as exc:
        print(f"[TTS error] {exc}")
        return 1
    print("Pocket TTS is ready.\n")

    _chat_loop(config, llm, tts)
    return 0


def _chat_loop(config: Config, llm: OllamaClient, tts: PocketTTSEngine) -> None:
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if not user_text:
            continue

        if user_text.lower() in config.exit_commands:
            print("Goodbye!")
            return

        try:
            reply = llm.generate_reply(user_text)
        except LLMError as exc:
            print(f"[LLM error] {exc}")
            continue

        print(f"Assistant: {reply}")

        try:
            wav_path = tts.synthesize(reply, config.output_path)
        except TTSError as exc:
            print(f"[TTS error] {exc}")
            continue

        try:
            play_wav(wav_path)
        except AudioPlaybackError as exc:
            print(f"[Playback error] {exc}")
            print(f"(Speech was still saved to: {wav_path})")


def main() -> None:
    sys.exit(run())
