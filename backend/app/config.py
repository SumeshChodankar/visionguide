from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables / .env (see .env.example)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- AI providers ---------------------------------------------------
    gemini_api_key: str = ""
    # Optional proxy in front of the Gemini API (e.g. a hackathon key proxy). Blank = Google directly.
    gemini_base_url: str = ""
    # Fast model for the per-frame safety/perception loop (structured JSON, no tools).
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_s: float = 7.0
    vision_thinking_level: str = "minimal"   # Gemini 3.x; 2.x models use thinking_budget=0 instead

    # Brain: tool-calling model that handles voice turns and planning.
    #   agent_mode=brain -> Gemini decides which tools (agents) to call
    #   agent_mode=rules -> deterministic intent parser only (no LLM in the loop)
    agent_mode: str = "brain"
    brain_model: str = "gemini-4-argon"
    brain_thinking_level: str = "low"        # low keeps voice turns responsive; raise for harder planning
    brain_max_steps: int = 6                 # max model<->tool round trips per user turn
    brain_timeout_s: float = 25.0

    openai_api_key: str = ""  # Whisper speech-to-text
    whisper_model: str = "whisper-1"

    # --- Google Maps Platform --------------------------------------------
    google_maps_api_key: str = ""
    # New Cloud projects cannot enable the legacy Directions API (legacy since
    # March 2025), so Routes API computeRoutes is the default. Set true only if
    # your project already has the legacy Directions API enabled.
    use_legacy_directions: bool = False

    # --- Area map agent (prefetches the user's surroundings in the background) ---
    area_prefetch: bool = True
    area_radius_m: int = 350              # nearby-places search radius
    area_refresh_m: float = 150.0         # re-fetch once the user has moved this far from the last centre
    area_max_age_s: float = 900.0         # ...or when the map is older than this
    area_grounding_model: str = ""        # Gemini model for the Google Maps-grounded briefing ("" = brain_model)
    # OpenStreetMap (Overpass + Nominatim) fills gaps when a Google Maps API is not enabled on the key.
    osm_fallback: bool = True

    # --- Your own building data (floor plans, photos): backend/buildings/<id>/building.json ---
    buildings_dir: str = "buildings"

    # --- Local CV models ---------------------------------------------------
    enable_yolo: bool = True
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    camera_hfov_deg: float = 60.0
    camera_vfov_deg: float = 70.0  # portrait phone

    enable_ocr: bool = True
    ocr_languages: str = "en"
    ocr_gpu: bool = False
    ocr_min_conf: float = 0.35

    # --- Behaviour tuning ---------------------------------------------------
    speech_cooldown_s: float = 10.0       # don't repeat the same sentence sooner
    critical_cooldown_s: float = 4.0      # hazards can repeat sooner
    step_advance_radius_m: float = 15.0
    preannounce_m: float = 35.0
    arrival_radius_m: float = 30.0        # switch from outdoor to entrance search
    offroute_threshold_m: float = 45.0
    reroute_min_interval_s: float = 30.0
    ground_floor_is_one: bool = True      # US-style numbering for 2-digit rooms

    # --- Infra ---------------------------------------------------------------
    db_path: str = "visionguide.db"
    cors_origins: str = "*"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
