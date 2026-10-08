"""
Centralized application configuration.

All configuration is read from environment variables (optionally loaded from
a .env file). Nothing in this file contains secrets - real values must be
supplied via the environment. See ../../.env.example for the full list of
supported variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = backend/
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, all overridable via environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    APP_NAME: str = "AI Multimodal Tutor"
    ENV: Literal["development", "test", "production"] = "development"
    DEBUG: bool = True

    # --- Database ---
    DATABASE_URL: str = Field(
        default=f"sqlite:///{(BACKEND_ROOT / 'data' / 'tutor.db').as_posix()}"
    )

    # --- Storage ---
    UPLOAD_DIR: str = str(BACKEND_ROOT / "data" / "uploads")
    PROCESSED_DIR: str = str(BACKEND_ROOT / "data" / "processed")
    MAX_UPLOAD_SIZE_MB: int = 8
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = Field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"]
    )

    # --- CORS ---
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    # --- Model providers ---
    # "ml" (classical scikit-learn symbol recognizer - the default),
    # "demo" (hash-matched fixtures, no model), and optional baselines:
    # "trocr" / "donut" (deep-learning HF models, need requirements-dl.txt),
    # "vision_api" (Claude or OpenAI-compatible vision endpoint).
    DEFAULT_OCR_PROVIDER: Literal["ml", "demo", "trocr", "donut", "vision_api"] = "ml"

    # Which vision-language backend to use when provider == vision_api.
    VISION_API_BACKEND: Literal["anthropic", "openai_compatible", "none"] = "none"
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"
    OPENAI_COMPATIBLE_BASE_URL: str | None = None
    OPENAI_COMPATIBLE_API_KEY: str | None = None
    OPENAI_COMPATIBLE_MODEL: str | None = None

    # --- Local HF model configuration ---
    TROCR_MODEL_NAME: str = "microsoft/trocr-base-handwritten"
    DONUT_MODEL_NAME: str = "naver-clova-ix/donut-base"
    HF_DEVICE: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    HF_CACHE_DIR: str | None = None

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    @field_validator("CORS_ORIGINS", "ALLOWED_UPLOAD_EXTENSIONS", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def ensure_dirs(self) -> None:
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.DATABASE_URL.replace("sqlite:///", "")).parent.mkdir(
            parents=True, exist_ok=True
        )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
