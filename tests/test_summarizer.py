"""Unit tests for Claude summarization (API mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.exceptions import SummarizationError
from app.summarizer import summarize_articles, summarize_with_claude
from db.articles import ArticleRecord


def _article() -> ArticleRecord:
    """Sample article eligible for summarization."""
    return ArticleRecord(
        id=3,
        title="Community garden feeds neighborhood",
        url="https://example.com/garden",
        full_text="Volunteers grew produce for local families.",
    )


@patch("app.summarizer.anthropic.Anthropic")
def test_summarize_with_claude_returns_text(mock_anthropic_cls: MagicMock) -> None:
    """Successful Claude response should return joined text blocks."""
    mock_client = mock_anthropic_cls.return_value
    text_block = MagicMock(type="text", text="A hopeful community story.")
    mock_client.messages.create.return_value = MagicMock(content=[text_block])

    summary = summarize_with_claude(_article(), api_key="test-key")

    assert summary == "A hopeful community story."
    mock_client.messages.create.assert_called_once()


@patch("app.summarizer.time.sleep")
@patch("app.summarizer.anthropic.Anthropic")
def test_summarize_with_claude_retries_then_raises(
    mock_anthropic_cls: MagicMock,
    _mock_sleep: MagicMock,
) -> None:
    """After max retries, SummarizationError should be raised."""
    mock_client = mock_anthropic_cls.return_value
    mock_client.messages.create.side_effect = RuntimeError("boom")

    with pytest.raises(SummarizationError):
        summarize_with_claude(_article(), api_key="test-key")

    assert mock_client.messages.create.call_count == 3  # 1 try + 2 retries


@patch("app.summarizer.articles_db.update_summaries", return_value=1)
@patch("app.summarizer.summarize_with_claude", return_value="Short hopeful summary.")
@patch("app.summarizer.articles_db.get_articles_needing_summary")
def test_summarize_articles_persists_results(
    mock_get: MagicMock,
    mock_claude: MagicMock,
    mock_update: MagicMock,
    test_settings: Settings,
) -> None:
    """summarize_articles should only persist successful Claude outputs."""
    settings = Settings(
        database_url=test_settings.database_url,
        newsapi_key=test_settings.newsapi_key,
        anthropic_api_key="test-anthropic-key",
        sendgrid_api_key="",
        sendgrid_from_email="",
        digest_recipient_email="",
        similarity_threshold=test_settings.similarity_threshold,
        sentiment_threshold=test_settings.sentiment_threshold,
        digest_hour=test_settings.digest_hour,
        digest_timezone=test_settings.digest_timezone,
        fetch_window_hours=test_settings.fetch_window_hours,
    )
    mock_get.return_value = [_article()]

    updated = summarize_articles(settings)

    assert updated == 1
    mock_update.assert_called_once_with([(3, "Short hopeful summary.")])
    mock_claude.assert_called_once()
