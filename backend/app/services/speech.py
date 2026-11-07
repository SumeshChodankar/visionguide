"""Speech-to-text via OpenAI Whisper.

Text-to-speech runs on the phone (expo-speech -> iOS AVSpeechSynthesizer /
Android TextToSpeech): zero network latency, works offline, and respects the
user's preferred system voice and rate, which matters to screen-reader users.
"""
from __future__ import annotations

import logging

from ..config import Settings

log = logging.getLogger("visionguide.speech")

# Biases Whisper toward the vocabulary we care about (room codes, venues).
WHISPER_PROMPT = ("Navigation commands for a blind user. Take me to Room 443B. Room 205A. Gate 4. "
                  "Building C. Find the elevator. Guide me to reception. Where am I? What is in front of me?")


class WhisperService:
    """Whisper when an OpenAI key is set; otherwise Gemini transcribes the audio
    (so tap-to-talk works in Expo Go with only the Gemini key)."""

    def __init__(self, settings: Settings, gemini=None):
        self.settings = settings
        self.gemini = gemini
        self.whisper = bool(settings.openai_api_key)
        self.enabled = self.whisper or bool(gemini and getattr(gemini, "enabled", False))
        self.engine = "whisper" if self.whisper else ("gemini" if self.enabled else None)
        self._client = None
        if self.whisper:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=15.0)

    async def transcribe(self, audio: bytes, filename: str = "speech.m4a") -> str:
        if self.whisper:
            r = await self._client.audio.transcriptions.create(
                model=self.settings.whisper_model, file=(filename, audio), language="en", prompt=WHISPER_PROMPT,
            )
            return (r.text or "").strip()
        if self.enabled:
            return await self.gemini.transcribe(audio, filename, WHISPER_PROMPT)
        raise RuntimeError("No speech-to-text configured (OPENAI_API_KEY or GEMINI_API_KEY); use /command with text")
