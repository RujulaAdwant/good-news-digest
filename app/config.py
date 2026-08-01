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


def _bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with a default.

    Accepts true/1/yes/on (case-insensitive); anything else is False when set.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional_float_env(name: str) -> float | None:
    """Parse an optional float env var; empty/unset returns None."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the Good News Digest service."""

    database_url: str
    thenewsapi_key: str
    anthropic_api_key: str
    sendgrid_api_key: str
    sendgrid_from_email: str
    digest_recipient_email: str
    similarity_threshold: float
    sentiment_threshold: float
    # When set, digest picks prefer scores near this value (Glass tone) instead
    # of maximizing positivity. None keeps legacy ORDER BY sentiment DESC.
    # Starting point for Glass: 0.65 with SENTIMENT_THRESHOLD around 0.52.
    sentiment_target: float | None
    digest_size: int
    # Starting point: below dedup (0.85) so near-dupes stay out, but same-topic
    # stories (e.g. two climate wins) are treated as overlapping for the digest.
    topic_diversity_threshold: float
    # Starting point: require article closer to good-news prototypes than
    # corporate/IR by at least this cosine margin (sim_good - sim_corporate).
    relevance_margin: float
    digest_hour: int
    # V1: single-user Pacific time. Future: per-user timezone preference.
    digest_timezone: str
    fetch_window_hours: int
    # Dev/test only: when True, DELETE /digest may clear email_sent_at so you
    # can recompile and resend the same calendar day. Keep False in production.
    allow_digest_reset: bool
    # Reserved for a future hosted header banner; email HTML does not use it yet.
    digest_banner_url: str


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings(
        database_url=_require_env("DATABASE_URL"),
        thenewsapi_key=os.getenv("THENEWSAPI_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        sendgrid_api_key=os.getenv("SENDGRID_API_KEY", ""),
        sendgrid_from_email=os.getenv("SENDGRID_FROM_EMAIL", ""),
        digest_recipient_email=os.getenv("DIGEST_RECIPIENT_EMAIL", ""),
        similarity_threshold=_float_env("SIMILARITY_THRESHOLD", 0.85),
        sentiment_threshold=_float_env("SENTIMENT_THRESHOLD", 0.6),
        sentiment_target=_optional_float_env("SENTIMENT_TARGET"),
        digest_size=_int_env("DIGEST_SIZE", 5),
        topic_diversity_threshold=_float_env("TOPIC_DIVERSITY_THRESHOLD", 0.70),
        relevance_margin=_float_env("RELEVANCE_MARGIN", 0.05),
        digest_hour=_int_env("DIGEST_HOUR", 7),
        digest_timezone=os.getenv("DIGEST_TIMEZONE", "America/Los_Angeles"),
        fetch_window_hours=_int_env("FETCH_WINDOW_HOURS", 24),
        allow_digest_reset=_bool_env("ALLOW_DIGEST_RESET", False),
        digest_banner_url=os.getenv("DIGEST_BANNER_URL", ""),
    )
