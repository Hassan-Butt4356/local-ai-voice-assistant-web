"""
Web server for the Local AI Voice Assistant.

Serves web/index.html and exposes POST /api/chat which accepts EITHER:
  - form field "text"  -> sent straight to Ollama
  - form file  "audio" -> transcribed with SenseVoice first, then sent to Ollama

Either way: Ollama generates a reply -> Pocket TTS synthesizes it -> the
WAV is returned to the browser for playback.

Run with:
    python server.py
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

# Ensure a literally-named "ffmpeg.exe" is on PATH (needed by funasr/librosa
# to decode webm/opus audio recorded by the browser). imageio-ffmpeg bundles
# a static binary but under a versioned filename (e.g. ffmpeg-win64-vX.exe),
# which funasr's hardcoded "ffmpeg" subprocess call won't find — so we copy
# it once into a local folder under the exact name "ffmpeg.exe" and add that
# folder to PATH. No system-wide ffmpeg install required.
import imageio_ffmpeg

_ffmpeg_bin_dir = Path(__file__).parent / ".ffmpeg_bin"
_ffmpeg_bin_dir.mkdir(exist_ok=True)
_ffmpeg_exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
_ffmpeg_dst = _ffmpeg_bin_dir / _ffmpeg_exe_name
if not _ffmpeg_dst.exists():
    shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), _ffmpeg_dst)
    if os.name != "nt":
        _ffmpeg_dst.chmod(0o755)
os.environ["PATH"] = str(_ffmpeg_bin_dir) + os.pathsep + os.environ.get("PATH", "")

from flask import Flask, jsonify, request, send_from_directory

from app.config import Config, ConfigError
from app.llm import LLMError, OllamaClient
from app.stt import STTError, SenseVoiceEngine
from app.tts import PocketTTSEngine, TTSError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="web", static_url_path="")

# --- Load everything once at startup ---
config = Config.load()
llm = OllamaClient(model=config.ollama_model, host=config.ollama_host, system_prompt=config.system_prompt)
tts = PocketTTSEngine(voice=config.tts_voice, language=config.tts_language)
stt = SenseVoiceEngine(
    model_id=config.stt_model,
    device=config.stt_device,
    language=config.stt_language_code,
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def _startup() -> None:
    logger.info("Connecting to Ollama...")
    llm.check_connection()
    logger.info("Loading Pocket TTS...")
    tts.load()
    logger.info("Loading SenseVoice STT...")
    stt.load()
    logger.info("All models ready.")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/output/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(config.output_dir.resolve(), filename)


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        user_text = None

        if "audio" in request.files and request.files["audio"].filename:
            audio_file = request.files["audio"]
            temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{audio_file.filename}"
            audio_file.save(temp_path)
            try:
                user_text = stt.transcribe(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
        else:
            user_text = (request.form.get("text") or "").strip()

        if not user_text:
            return jsonify({"error": "No text or audio provided."}), 400

        reply = llm.generate_reply(user_text)

        # unique filename per request so concurrent users don't clobber each other
        wav_name = f"{uuid.uuid4().hex}.wav"
        wav_path = tts.synthesize(reply, config.output_dir / wav_name)

        return jsonify({
            "transcribed_text": user_text,
            "reply": reply,
            "audio_url": f"/output/{wav_path.name}",
        })

    except (LLMError, TTSError, STTError) as exc:
        logger.exception("Request failed")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500


if __name__ == "__main__":
    try:
        _startup()
    except (ConfigError, LLMError, TTSError, STTError) as exc:
        print(f"[Startup error] {exc}")
        raise SystemExit(1)

    app.run(host="0.0.0.0", port=5000, debug=False)