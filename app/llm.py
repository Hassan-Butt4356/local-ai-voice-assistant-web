"""
Thin wrapper around a locally running Ollama server.

Keeps conversation history in memory and exposes a single
`generate_reply()` method so the rest of the app doesn't need to know
anything about the Ollama client library or HTTP details.
"""

from __future__ import annotations

import logging
from typing import List, TypedDict

import ollama
from ollama import ResponseError

logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMError(Exception):
    """Raised when the local LLM (Ollama) cannot produce a reply."""


class OllamaClient:
    """Wraps an Ollama chat model running locally, with conversation memory."""

    def __init__(self, model: str, host: str, system_prompt: str) -> None:
        self.model = model
        self.host = host
        self.system_prompt = system_prompt
        self._client = ollama.Client(host=host)
        self._history: List[ChatMessage] = []
        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})

    def check_connection(self) -> None:
        """
        Verify the Ollama server is reachable and the configured model
        is available. Raises LLMError with a helpful message otherwise.
        """
        try:
            models_response = self._client.list()
        except Exception as exc:  # connection refused, DNS error, etc.
            raise LLMError(
                f"Could not reach Ollama at '{self.host}'. Is `ollama serve` "
                f"running? Original error: {exc}"
            ) from exc

        available = {
            getattr(m, "model", None) or getattr(m, "name", None)
            for m in getattr(models_response, "models", [])
        }
        # Ollama model names may or may not include a ':tag' suffix.
        base_names = {name.split(":")[0] for name in available if name}
        if self.model not in available and self.model.split(":")[0] not in base_names:
            raise LLMError(
                f"Model '{self.model}' was not found on the Ollama server "
                f"at '{self.host}'. Pull it first with:\n"
                f"    ollama pull {self.model}"
            )

    def generate_reply(self, user_text: str) -> str:
        """
        Send `user_text` to the model, append both turns to the running
        conversation history, and return the assistant's reply text.
        """
        if not user_text or not user_text.strip():
            raise LLMError("Cannot generate a reply for empty input.")

        self._history.append({"role": "user", "content": user_text})
        try:
            response = self._client.chat(model=self.model, messages=self._history)
        except ResponseError as exc:
            # Roll back the user turn we optimistically added, so a failed
            # call doesn't corrupt the conversation history.
            self._history.pop()
            raise LLMError(f"Ollama returned an error: {exc}") from exc
        except Exception as exc:
            self._history.pop()
            raise LLMError(f"Failed to contact Ollama: {exc}") from exc

        reply_text = (response.get("message", {}) or {}).get("content", "").strip()
        if not reply_text:
            self._history.pop()
            raise LLMError("Ollama returned an empty response.")

        self._history.append({"role": "assistant", "content": reply_text})
        return reply_text

    def reset(self) -> None:
        """Clear conversation history, keeping the system prompt if set."""
        self._history = []
        if self.system_prompt:
            self._history.append({"role": "system", "content": self.system_prompt})
