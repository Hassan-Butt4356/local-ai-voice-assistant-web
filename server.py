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

import subprocess


def convert_to_wav(src_path: Path) -> Path:
    """
    Convert any browser-recorded audio (typically WebM/Opus) to a plain
    16-bit PCM WAV file using ffmpeg, so libraries that only support WAV
    (like Pocket TTS's audio loader) can read it.
    """
    dst_path = src_path.with_suffix(".wav")
    result = subprocess.run(
        [str(_ffmpeg_dst), "-y", "-i", str(src_path), "-ar", "24000", "-ac", "1", str(dst_path)],
        capture_output=True,
    )
    if result.returncode != 0 or not dst_path.exists():
        raise RuntimeError(
            f"ffmpeg failed to convert audio: {result.stderr.decode(errors='ignore')[:500]}"
        )
    return dst_path


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
config.voices_dir.mkdir(exist_ok=True, parents=True)

# Built-in Pocket TTS voice presets (for the dropdown in the UI)
PRESET_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]


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


@app.route("/api/voices", methods=["GET"])
def list_voices():
    """List built-in presets and previously cloned custom voices."""
    custom = sorted(p.stem for p in config.voices_dir.glob("*.safetensors"))
    return jsonify({"presets": PRESET_VOICES, "custom": custom, "active": tts.voice})


@app.route("/api/clone-voice", methods=["POST"])
def clone_voice():
    """
    Accepts an audio recording ("audio" file) and a "voice_name" field,
    turns it into a reusable voice embedding, and immediately switches
    the assistant to speak in that voice for subsequent /api/chat calls.
    """
    try:
        if "audio" not in request.files or not request.files["audio"].filename:
            return jsonify({"error": "No audio file provided."}), 400

        voice_name = (request.form.get("voice_name") or "my_voice").strip()
        audio_file = request.files["audio"]
        temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{audio_file.filename}"
        audio_file.save(temp_path)

        wav_path = None
        try:
            wav_path = convert_to_wav(temp_path)
            dest = tts.clone_voice_from_file(wav_path, voice_name, config.voices_dir)
        finally:
            temp_path.unlink(missing_ok=True)
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)

        tts.set_active_voice(str(dest))

        return jsonify({"voice_name": dest.stem, "message": f"Voice '{dest.stem}' cloned and activated."})

    except TTSError as exc:
        logger.exception("Voice cloning failed")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500


@app.route("/api/use-voice", methods=["POST"])
def use_voice():
    """Switch the active TTS voice to a preset name or a saved custom voice."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        voice_ref = (data.get("voice") or "").strip()
        if not voice_ref:
            return jsonify({"error": "No voice specified."}), 400

        # Resolve a bare custom-voice name to its saved .safetensors file
        candidate = config.voices_dir / f"{voice_ref}.safetensors"
        target = str(candidate) if candidate.exists() else voice_ref

        tts.set_active_voice(target)
        return jsonify({"message": f"Active voice set to '{voice_ref}'.", "active": voice_ref})

    except TTSError as exc:
        logger.exception("Voice switch failed")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500


@app.route("/api/tts", methods=["POST"])
def text_to_speech():
    """
    Convert arbitrary pasted text directly to speech — no LLM involved.
    Accepts JSON {"text": "..."} or a form field "text".
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or request.form.get("text") or "").strip()

        if not text:
            return jsonify({"error": "No text provided."}), 400

        wav_name = f"{uuid.uuid4().hex}.wav"
        wav_path = tts.synthesize(text, config.output_dir / wav_name)

        return jsonify({
            "text": text,
            "audio_url": f"/output/{wav_path.name}",
            "audio_filename": wav_path.name,
        })

    except TTSError as exc:
        logger.exception("TTS request failed")
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error")
        return jsonify({"error": f"Unexpected error: {exc}"}), 500

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