"""Shared pytest fixtures."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from app.config import Settings, get_settings


@pytest.fixture
def test_settings() -> Settings:
    """Settings with a fixed timezone and fetch window for deterministic tests."""
    return Settings(
        database_url="postgresql://digest:digest@localhost:5432/good_news_digest",
        newsapi_key="test-newsapi-key",
        anthropic_api_key="",
        sendgrid_api_key="",
        sendgrid_from_email="",
        digest_recipient_email="",
        similarity_threshold=0.85,
        sentiment_threshold=0.6,
        digest_hour=7,
        digest_timezone="America/Los_Angeles",
        fetch_window_hours=24,
    )


@pytest.fixture
def recent_published_at() -> datetime:
    """UTC timestamp safely inside the default 24-hour fetch window."""
    return datetime.now(UTC) - timedelta(hours=2)


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Iterator[None]:
    """Clear cached settings between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
