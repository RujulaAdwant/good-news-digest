"""Unit tests for Claude summarization (API mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.exceptions import SummarizationError
from app.summarizer import (
    is_usable_summary,
    summarize_articles,
    summarize_with_claude,
)
from db.articles import ArticleRecord


def _article(*, summary: str | None = None, full_text: str | None = None) -> ArticleRecord:
    """Sample article eligible for summarization."""
    return ArticleRecord(
        id=3,
        title="Community garden feeds neighborhood",
        url="https://example.com/garden",
        full_text=full_text
        or (
            "Volunteers grew produce for local families across several blocks "
            "and donated the harvest to a nearby food pantry every weekend."
        ),
        sentiment_score=0.92,
        summary=summary,
    )


def _settings_with_key(test_settings: Settings) -> Settings:
    """Copy fixture settings with a fake Anthropic key."""
    return Settings(
        database_url=test_settings.database_url,
        thenewsapi_key=test_settings.thenewsapi_key,
        anthropic_api_key="test-anthropic-key",
        sendgrid_api_key="",
        sendgrid_from_email="",
        digest_recipient_email="",
        similarity_threshold=test_settings.similarity_threshold,
        sentiment_threshold=test_settings.sentiment_threshold,
        sentiment_target=test_settings.sentiment_target,
        digest_size=test_settings.digest_size,
        topic_diversity_threshold=test_settings.topic_diversity_threshold,
        relevance_margin=test_settings.relevance_margin,
        digest_hour=test_settings.digest_hour,
        digest_timezone=test_settings.digest_timezone,
        fetch_window_hours=test_settings.fetch_window_hours,
        allow_digest_reset=test_settings.allow_digest_reset,
        digest_banner_url=test_settings.digest_banner_url,
    )


def test_is_usable_summary_rejects_refusal_prose() -> None:
    """Claude meta-refusals must not count as digest summaries."""
    assert is_usable_summary("A hopeful community garden story.") is True
    assert is_usable_summary(None) is False
    assert is_usable_summary("INSUFFICIENT_CONTENT") is False
    assert (
        is_usable_summary(
            "I'm sorry, but the article you've shared doesn't contain enough "
            "substantive content to summarize."
        )
        is False
    )
    assert is_usable_summary("I'm unable to summarize this article because...") is False


@patch("app.summarizer.anthropic.Anthropic")
def test_summarize_with_claude_returns_text(mock_anthropic_cls: MagicMock) -> None:
    """Successful Claude response should return joined text blocks."""
    mock_client = mock_anthropic_cls.return_value
    text_block = MagicMock(type="text", text="A hopeful community story.")
    mock_client.messages.create.return_value = MagicMock(content=[text_block])

    summary = summarize_with_claude(_article(), api_key="test-key")

    assert summary == "A hopeful community story."
    mock_client.messages.create.assert_called_once()


@patch("app.summarizer.anthropic.Anthropic")
def test_summarize_with_claude_rejects_refusal(
    mock_anthropic_cls: MagicMock,
) -> None:
    """Refusal / insufficient-content replies must raise, not be returned."""
    mock_client = mock_anthropic_cls.return_value
    text_block = MagicMock(
        type="text",
        text=(
            "I'm sorry, but the article you've shared doesn't contain enough "
            "substantive content to summarize."
        ),
    )
    mock_client.messages.create.return_value = MagicMock(content=[text_block])

    with pytest.raises(SummarizationError, match="non-summary refusal"):
        summarize_with_claude(_article(), api_key="test-key")


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
@patch("app.summarizer.select_articles_for_digest")
def test_summarize_articles_only_claude_digest_picks(
    mock_select: MagicMock,
    mock_claude: MagicMock,
    mock_update: MagicMock,
    test_settings: Settings,
) -> None:
    """summarize_articles should only Claude selected picks missing summaries."""
    settings = _settings_with_key(test_settings)
    already_summarized = _article(summary="Cached summary.")
    needs_summary = ArticleRecord(
        id=4,
        title="Solar co-op powers town",
        url="https://example.com/solar",
        full_text=(
            "A co-op installed panels for free and cut neighborhood power bills "
            "for dozens of households over the first year of operation."
        ),
        sentiment_score=0.88,
        summary=None,
    )
    mock_select.return_value = [already_summarized, needs_summary]

    updated = summarize_articles(settings)

    assert updated == 1
    mock_select.assert_called_once_with(
        settings,
        digest_size=settings.digest_size * 2,
    )
    mock_claude.assert_called_once_with(
        needs_summary,
        api_key="test-anthropic-key",
    )
    mock_update.assert_called_once_with([(4, "Short hopeful summary.")])


@patch("app.summarizer.articles_db.update_summaries", return_value=0)
@patch("app.summarizer.articles_db.clear_summaries", return_value=1)
@patch("app.summarizer.summarize_with_claude")
@patch("app.summarizer.select_articles_for_digest")
def test_summarize_articles_clears_cached_refusals(
    mock_select: MagicMock,
    mock_claude: MagicMock,
    mock_clear: MagicMock,
    mock_update: MagicMock,
    test_settings: Settings,
) -> None:
    """Cached refusal text should be scrubbed and not re-sent to digests."""
    settings = _settings_with_key(test_settings)
    thin = ArticleRecord(
        id=5,
        title="DJ Cuppy looks good",
        url="https://example.com/cuppy",
        full_text="She looks good.",
        sentiment_score=0.99,
        summary=(
            "I'm sorry, but the article you've shared doesn't contain enough "
            "substantive content to summarize."
        ),
    )
    mock_select.return_value = [thin]

    updated = summarize_articles(settings)

    assert updated == 0
    mock_clear.assert_called_once_with([5])
    mock_claude.assert_not_called()
    mock_update.assert_not_called()


@patch("app.summarizer.select_articles_for_digest", return_value=[])
def test_summarize_articles_skips_when_no_picks(
    _mock_select: MagicMock,
    test_settings: Settings,
) -> None:
    """Empty digest selection should write zero summaries."""
    settings = _settings_with_key(test_settings)

    assert summarize_articles(settings) == 0
