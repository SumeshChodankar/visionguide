"""Gemini calls.

  * Vision loop (GEMINI_MODEL, a Flash model): one structured multimodal call
    per frame. Latency matters more than depth here, so thinking is minimised
    and outputs are capped.
  * Brain (BRAIN_MODEL, Gemini 4 Argon by default): tool-calling turns, see
    agents/brain.py.
 Every call has a timeout; callers
treat `None` as "vision unavailable this tick" and fall back to YOLO + OCR.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, ValidationError

from ..config import Settings

log = logging.getLogger("visionguide.gemini")

Side = Literal["left", "ahead", "right"]


class GLandmark(BaseModel):
    type: str
    side: Side
    distance_m: float | None = None
    description: str = ""


class GHazard(BaseModel):
    type: str
    severity: Literal["low", "medium", "high"]
    side: Side
    distance_m: float | None = None
    description: str = ""


class GScene(BaseModel):
    summary: str
    environment: Literal["indoor", "outdoor", "unknown"]
    landmarks: list[GLandmark]
    hazards: list[GHazard]
    sign_texts: list[str]
    floor_indicator: str | None = None
    destination_visible: bool = False
    destination_evidence: str | None = None
    suggested_guidance: str | None = None


SCENE_SYSTEM = """You are the eyes of a blind pedestrian. You receive one photo from a phone held at chest height, camera facing the direction the person is walking.

Return JSON only, matching the schema. Rules:
- SAFETY FIRST. List every hazard in or near the walking path: wet floor / spill, wet-floor sign, obstacle, chair or furniture in path, open door (especially doors opening toward the person), people close by, construction barrier, stairs going down or up, curb, uneven or broken surface, low-hanging object, glass door, vehicle, bicycle. severity=high if it is directly ahead within ~3 m or could cause a fall.
- side is relative to the camera: left third, ahead (middle), right third. distance_m is your best estimate in metres (null if unsure).
- landmarks: entrance, exit, reception, elevator, stairs, escalator, corridor, door, restroom, sign, information desk, gate, turnstile, building name. Include open/closed state for doors in description.
- sign_texts: transcribe EXACTLY every legible sign, room plate, building name, floor indicator and arrow sign. Include arrows as words, e.g. "Rooms 400-450 → right".
- floor_indicator: the current floor if an elevator display or floor sign shows it, else null.
- destination_visible: true only if you can see the specific destination named in the context (e.g. the room plate "443B", or the named building's entrance). Put what you saw in destination_evidence.
- summary: at most 2 short sentences in plain spoken English, most important first. No markdown, no "the image shows".
- suggested_guidance: one short actionable instruction toward the goal (e.g. "Elevator at 1 o'clock, 3 meters."), or null.
Never invent text you cannot read."""


class GeminiService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.gemini_api_key)
        self._client = None
        self._thinking_unsupported: dict[str, bool] = {}
        if self.enabled:
            from google import genai  # imported lazily so tests run without the SDK configured

            http_options = None
            if settings.gemini_base_url:
                from google.genai import types

                http_options = types.HttpOptions(base_url=settings.gemini_base_url)
            self._client = genai.Client(api_key=settings.gemini_api_key, http_options=http_options)

    def _thinking_off(self, model: str, level: str) -> bool:
        """Has this model rejected this thinking level? Tracked per (model, level): a model that
        refuses "minimal" may still accept "low", so one refusal mustn't disable thinking everywhere."""
        return bool(self._thinking_unsupported.get(f"{model}:{level.lower()}")
                    or self._thinking_unsupported.get(f"{model}:*"))

    def _thinking(self, model: str, level: str):
        """Gemini 2.x: thinking_budget=0 disables thinking. Gemini 3+: thinking_level."""
        from google.genai import types

        if self._thinking_off(model, level):
            return None
        if model.startswith("gemini-2"):
            return types.ThinkingConfig(thinking_budget=0)
        return types.ThinkingConfig(thinking_level=level.upper())

    def _config(self, system: str, *, schema=None, json_mode: bool = False, max_tokens: int = 700,
                temperature: float = 0.2, model: str | None = None):
        from google.genai import types

        m = model or self.settings.gemini_model
        if self._thinking_off(m, self.settings.vision_thinking_level):
            # Without our thinking setting the model thinks by default, and thinking tokens count
            # against max_output_tokens: answers were cut mid-sentence. Leave room for both.
            max_tokens = max(max_tokens * 4, 2048)
        kwargs = dict(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=self._thinking(model or self.settings.gemini_model, self.settings.vision_thinking_level),
        )
        if schema is not None or json_mode:
            kwargs["response_mime_type"] = "application/json"
        if schema is not None:
            kwargs["response_schema"] = schema
        return types.GenerateContentConfig(**kwargs)

    async def analyze_scene(self, jpeg: bytes, context: str) -> GScene | None:
        if not self.enabled:
            return None
        from google.genai import types

        contents = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), f"Context: {context}"]
        try:
            resp = await self._generate(self.settings.gemini_model, contents,
                                        lambda: self._config(SCENE_SYSTEM, schema=GScene),
                                        self.settings.gemini_timeout_s)
            return _parse_scene(resp.text)
        except asyncio.TimeoutError:
            log.warning("Gemini scene analysis timed out")
        except Exception as e:  # noqa: BLE001 - never let vision take the request down
            log.warning("Gemini scene analysis failed: %s", e)
        return None

    async def text(self, system: str, prompt: str, *, json_mode: bool = False, max_tokens: int = 300,
                   timeout: float | None = None) -> str | None:
        if not self.enabled:
            return None
        try:
            resp = await self._generate(
                self.settings.gemini_model, prompt,
                lambda: self._config(system, json_mode=json_mode, max_tokens=max_tokens, temperature=0.4),
                timeout or self.settings.gemini_timeout_s)
            return (resp.text or "").strip() or None
        except Exception as e:  # noqa: BLE001
            log.warning("Gemini text call failed: %s", e)
            return None


    async def image_json(self, image: bytes, prompt: str, system: str, timeout: float = 60.0) -> dict | None:
        """Image + prompt -> parsed JSON object (used once per floor plan)."""
        if not self.enabled:
            return None
        from google.genai import types

        try:
            resp = await self._generate(
                self.settings.brain_model, [types.Part.from_bytes(data=image, mime_type="image/jpeg"), prompt],
                lambda: self._config(system, json_mode=True, max_tokens=2048, temperature=0.1,
                                     model=self.settings.brain_model), timeout)
            raw = (resp.text or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`").removeprefix("json").strip()
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as e:  # noqa: BLE001
            log.warning("Gemini image_json failed: %s", type(e).__name__ if not str(e) else e)
            return None

    async def image_text(self, image: bytes, prompt: str, system: str, timeout: float = 30.0) -> str | None:
        """Image + question -> short plain-text answer (e.g. a question about a floor plan)."""
        if not self.enabled:
            return None
        from google.genai import types

        try:
            resp = await self._generate(
                self.settings.brain_model, [types.Part.from_bytes(data=image, mime_type="image/jpeg"), prompt],
                lambda: self._config(system, max_tokens=600, temperature=0.2, model=self.settings.brain_model), timeout)
            return (resp.text or "").strip() or None
        except Exception as e:  # noqa: BLE001
            log.warning("Gemini image_text failed: %s", type(e).__name__ if not str(e) else e)
            return None

    AUDIO_MIME = {"m4a": "audio/mp4", "mp4": "audio/mp4", "aac": "audio/aac", "wav": "audio/wav",
                  "mp3": "audio/mp3", "ogg": "audio/ogg", "webm": "audio/webm", "3gp": "audio/3gpp", "flac": "audio/flac"}

    async def transcribe(self, audio: bytes, filename: str, vocabulary: str = "") -> str:
        """Speech-to-text with Gemini (used when no OpenAI/Whisper key is configured)."""
        from google.genai import types

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "m4a"
        mime = self.AUDIO_MIME.get(ext, "audio/mp4")
        system = ("You are a speech-to-text engine. Return ONLY the exact words spoken, in English, with no "
                  "quotes, labels or commentary. If nothing intelligible is spoken, return an empty string. "
                  f"Likely vocabulary: {vocabulary}")
        last: Exception | None = None
        for m in dict.fromkeys([mime, "audio/aac"]):  # m4a is AAC in an MP4 box; retry as raw AAC if refused
            try:
                resp = await self._generate(
                    self.settings.gemini_model,
                    [types.Part.from_bytes(data=audio, mime_type=m), "Transcribe this audio."],
                    lambda: self._config(system, max_tokens=200, temperature=0.0), 15.0)
                text = (resp.text or "").strip().strip('"')
                return "" if text.lower() in ("", "empty string", "(silence)", "[silence]") else text
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("Gemini transcription with %s failed: %s", m, e)
        raise RuntimeError(f"Gemini transcription failed: {last}")

    async def maps_grounded(self, prompt: str, lat: float, lng: float, *, timeout: float = 30.0,
                            system: str | None = None) -> tuple[str | None, list[str]]:
        """Ask Gemini with the Google Maps grounding tool, centred on the user.
        Returns (answer text, titles of the Google Maps places it cited)."""
        if not self.enabled:
            return None, []
        from google.genai import types

        model = self.settings.area_grounding_model or self.settings.brain_model

        def make_config():
            return types.GenerateContentConfig(
                system_instruction=system,
                tools=[types.Tool(google_maps=types.GoogleMaps())],
                tool_config=types.ToolConfig(retrieval_config=types.RetrievalConfig(
                    lat_lng=types.LatLng(latitude=lat, longitude=lng))),
                thinking_config=self._thinking(model, "low"),
                max_output_tokens=800,
                temperature=0.2,
            )

        try:
            resp = await self._generate(model, prompt, make_config, timeout)
        except Exception as e:  # noqa: BLE001
            log.warning("Maps-grounded Gemini call failed: %s", type(e).__name__ if not str(e) else e)
            return None, []
        sources: list[str] = []
        try:
            gm = resp.candidates[0].grounding_metadata
            for ch in (gm.grounding_chunks or []) if gm else []:
                title = getattr(getattr(ch, "maps", None), "title", None) or getattr(getattr(ch, "web", None), "title", None)
                if title and title not in sources:
                    sources.append(title)
        except Exception:  # noqa: BLE001
            pass
        return (resp.text or "").strip() or None, sources

    async def _generate(self, model: str, contents, make_config, timeout: float):
        """generate_content with a timeout; if the model rejects our thinking
        settings (they differ between Gemini generations), retry once without."""
        try:
            return await asyncio.wait_for(
                self._client.aio.models.generate_content(model=model, contents=contents, config=make_config()),
                timeout=timeout)
        except asyncio.TimeoutError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "thinking" not in msg.lower():
                raise
            # "Thinking level MINIMAL is not supported for this model" -> remember (model, minimal) only.
            m = re.search(r"thinking level\s+(\w+)", msg, re.I)
            key = f"{model}:{m.group(1).lower()}" if m else f"{model}:*"
            if not self._thinking_unsupported.get(key):
                log.info("Model %s rejected thinking setting (%s); retrying without it", model, key.split(":")[1])
            self._thinking_unsupported[key] = True
            # Always retry once: calls already in flight were built before the flag was set.
            return await asyncio.wait_for(
                self._client.aio.models.generate_content(model=model, contents=contents, config=make_config()),
                timeout=timeout)

    async def generate_with_tools(self, *, system: str, contents: list, declarations: list, timeout: float):
        """One brain step: the model either answers or asks for tool calls.
        Automatic function calling is disabled; the Brain runs tools itself so it
        can time them, cap the number of steps and keep safety rules outside the LLM."""
        from google.genai import types

        model = self.settings.brain_model

        def make_config():
            return types.GenerateContentConfig(
                system_instruction=system,
                tools=[types.Tool(function_declarations=declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=self._thinking(model, self.settings.brain_thinking_level),
                max_output_tokens=1024,
            )

        return await self._generate(model, contents, make_config, timeout)


def _parse_scene(raw: str | None) -> GScene | None:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        return GScene.model_validate_json(raw)
    except ValidationError:
        try:  # be lenient: fill defaults for missing lists
            data = json.loads(raw)
            data.setdefault("landmarks", [])
            data.setdefault("hazards", [])
            data.setdefault("sign_texts", [])
            data.setdefault("environment", "unknown")
            data.setdefault("summary", "")
            return GScene.model_validate(data)
        except Exception:  # noqa: BLE001
            log.warning("Unparseable Gemini scene output: %.200s", raw)
            return None
