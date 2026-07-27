"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    """Return a required environment variable or raise."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _float_env(name: str, default: float) -> float:
    """Parse a float environment variable with a default."""
    raw = os.getenv(name)
    return float(raw) if raw else default


def _int_env(name: str, default: int) -> int:
    """Parse an int environment variable with a default."""
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the Good News Digest service."""

    database_url: str
    newsapi_key: str
    anthropic_api_key: str
    sendgrid_api_key: str
    sendgrid_from_email: str
    digest_recipient_email: str
    similarity_threshold: float
    sentiment_threshold: float
    digest_hour: int
    # V1: single-user Pacific time. Future: per-user timezone preference.
    digest_timezone: str
    fetch_window_hours: int


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings(
        database_url=_require_env("DATABASE_URL"),
        newsapi_key=os.getenv("NEWSAPI_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        sendgrid_api_key=os.getenv("SENDGRID_API_KEY", ""),
        sendgrid_from_email=os.getenv("SENDGRID_FROM_EMAIL", ""),
        digest_recipient_email=os.getenv("DIGEST_RECIPIENT_EMAIL", ""),
        similarity_threshold=_float_env("SIMILARITY_THRESHOLD", 0.85),
        sentiment_threshold=_float_env("SENTIMENT_THRESHOLD", 0.6),
        digest_hour=_int_env("DIGEST_HOUR", 7),
        digest_timezone=os.getenv("DIGEST_TIMEZONE", "America/Los_Angeles"),
        fetch_window_hours=_int_env("FETCH_WINDOW_HOURS", 24),
    )
