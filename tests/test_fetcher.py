"""Tests for fetcher parsing and filtering logic."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from app.config import Settings
from app.fetcher import (
    _extract_rss_body,
    _extract_thenewsapi_body,
    fetch_articles,
    fetch_from_thenewsapi,
    first_paragraph,
    text_for_nlp,
    within_fetch_window,
)
from db.articles import ArticleRecord


def test_first_paragraph_returns_first_non_empty_block() -> None:
    """first_paragraph should ignore blank lines between paragraphs."""
    body = "First paragraph.\n\nSecond paragraph."
    assert first_paragraph(body) == "First paragraph."


def test_text_for_nlp_combines_title_and_first_paragraph() -> None:
    """NLP text should include both title and the first paragraph."""
    body = "Scientists made progress.\n\nMore details here."
    result = text_for_nlp("Breakthrough", body)
    assert result == "Breakthrough\n\nScientists made progress."


def test_within_fetch_window_rejects_missing_and_old_dates(test_settings: Settings) -> None:
    """Articles without dates or outside the window should be excluded."""
    recent = datetime.now(UTC) - timedelta(hours=1)
    old = datetime.now(UTC) - timedelta(hours=30)

    assert within_fetch_window(recent, settings=test_settings) is True
    assert within_fetch_window(old, settings=test_settings) is False
    assert within_fetch_window(None, settings=test_settings) is False


def test_extract_thenewsapi_body_prefers_description() -> None:
    """Description should win over the shorter snippet field."""
    article = {
        "description": "A hopeful science update with useful context.",
        "snippet": "A hopeful science update...",
    }
    assert (
        _extract_thenewsapi_body(article)
        == "A hopeful science update with useful context."
    )


def test_extract_rss_body_uses_summary_when_present() -> None:
    """RSS summary should be used when full content is unavailable."""
    entry = {"summary": "Good news snippet."}
    assert _extract_rss_body(entry) == "Good news snippet."


@patch("app.fetcher.fetch_from_rss")
@patch("app.fetcher.fetch_from_thenewsapi")
def test_fetch_articles_applies_time_filter(
    mock_thenewsapi: MagicMock,
    mock_rss: MagicMock,
    test_settings: Settings,
    recent_published_at: datetime,
) -> None:
    """fetch_articles should drop items outside the configured window."""
    recent = ArticleRecord(
        title="Recent",
        url="https://example.com/recent",
        published_at=recent_published_at,
    )
    old = ArticleRecord(
        title="Old",
        url="https://example.com/old",
        published_at=recent_published_at - timedelta(days=2),
    )
    mock_thenewsapi.return_value = [recent, old]
    mock_rss.return_value = []

    result = fetch_articles(test_settings)

    assert len(result) == 1
    assert result[0].title == "Recent"


@patch("app.fetcher._get_thenewsapi")
def test_fetch_from_thenewsapi_parses_articles(
    mock_get: MagicMock,
    test_settings: Settings,
    recent_published_at: datetime,
) -> None:
    """TheNewsAPI responses should be normalized into ArticleRecord objects."""
    mock_get.return_value = {
        "meta": {"found": 1, "returned": 1, "limit": 50, "page": 1},
        "data": [
            {
                "uuid": "abc-123",
                "title": "Scientists discover faster DNA sequencing",
                "description": "A new technique could accelerate medical research.",
                "snippet": "A new technique could...",
                "url": "https://example.com/dna",
                "published_at": recent_published_at.isoformat().replace("+00:00", "Z"),
                "source": "example.com",
                "categories": ["science"],
            }
        ],
    }

    articles = fetch_from_thenewsapi(test_settings)

    assert len(articles) == 1
    assert articles[0].title == "Scientists discover faster DNA sequencing"
    assert articles[0].full_text == "A new technique could accelerate medical research."
    assert articles[0].source == "example.com"
