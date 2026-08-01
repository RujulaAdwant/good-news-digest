"""Unit tests for digest HTML formatting and SendGrid delivery."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.emailer import format_digest_html, format_digest_subject, send_digest
from app.exceptions import EmailDeliveryError
from db.articles import ArticleRecord
from db.digests import DigestRecord


def _settings(**overrides: object) -> Settings:
    """Build settings with email credentials for send tests."""
    base = {
        "database_url": "postgresql://digest:digest@localhost:5432/good_news_digest",
        "thenewsapi_key": "",
        "anthropic_api_key": "",
        "sendgrid_api_key": "sg-test",
        "sendgrid_from_email": "digest@example.com",
        "digest_recipient_email": "you@example.com",
        "similarity_threshold": 0.85,
        "sentiment_threshold": 0.6,
        "sentiment_target": None,
        "digest_size": 5,
        "topic_diversity_threshold": 0.70,
        "relevance_margin": 0.05,
        "digest_hour": 7,
        "digest_timezone": "America/Los_Angeles",
        "fetch_window_hours": 24,
        "allow_digest_reset": False,
        "digest_banner_url": "https://cdn.example.com/cloud-banner.jpg",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_format_digest_subject_includes_date() -> None:
    """Subject should include the digest calendar date."""
    subject = format_digest_subject(date(2026, 7, 28))
    assert "The Glass Digest" in subject
    assert "July" in subject
    assert "2026" in subject


def test_format_digest_html_escapes_and_links() -> None:
    """HTML body should link the title and escape user-controlled text."""
    articles = [
        ArticleRecord(
            id=1,
            title='Win <script>alert("x")</script>',
            url="https://example.com/a?q=1&b=2",
            source="Good Source",
            summary="A hopeful <b>story</b>.",
            sentiment_score=0.9,
        )
    ]
    html = format_digest_html(articles, date(2026, 7, 28))

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert 'href="https://example.com/a?q=1&amp;b=2"' in html
    assert "Good Source" in html
    assert "&lt;b&gt;story&lt;/b&gt;" in html
    assert "<img" not in html
    assert "01 ·" in html
    assert "#0b57d0" not in html
    assert "The Glass Digest" in html


def test_format_digest_html_includes_articles_without_summary() -> None:
    """Stories without a usable summary still appear; only the blurb is omitted."""
    articles = [
        ArticleRecord(
            id=1,
            title="Has summary",
            url="https://example.com/ok",
            source="Good Source",
            summary="A hopeful community story.",
        ),
        ArticleRecord(
            id=2,
            title="No summary",
            url="https://example.com/missing",
            source="Other",
            summary=None,
        ),
        ArticleRecord(
            id=3,
            title="Refusal",
            url="https://example.com/refuse",
            source="Other",
            summary="I'm sorry, but I cannot summarize this.",
        ),
    ]
    html = format_digest_html(articles, date(2026, 7, 28))

    assert "Has summary" in html
    assert "A hopeful community story." in html
    assert "No summary" in html
    assert "Refusal" in html
    assert "I'm sorry" not in html
    assert "Summary unavailable" not in html
    assert "01 ·" in html
    assert "03 ·" in html


def test_format_digest_html_empty_pool() -> None:
    """Empty digests should still render a polite body."""
    html = format_digest_html([], date(2026, 7, 28))
    assert "Check back tomorrow" in html
    assert "<img" not in html


@patch("app.emailer.digests_db.mark_email_sent")
@patch("app.emailer.SendGridAPIClient")
@patch("app.emailer.articles_db.get_articles_by_ids")
@patch("app.emailer.digests_db.get_digest_by_date")
def test_send_digest_calls_sendgrid_and_stamps(
    mock_get_digest: MagicMock,
    mock_get_articles: MagicMock,
    mock_sg_cls: MagicMock,
    mock_mark_sent: MagicMock,
) -> None:
    """send_digest should send HTML and stamp email_sent_at."""
    digest_date = date(2026, 7, 28)
    mock_get_digest.return_value = DigestRecord(
        id=9,
        date=digest_date,
        article_ids=[1],
        email_sent_at=None,
    )
    mock_get_articles.return_value = [
        ArticleRecord(
            id=1,
            title="Hopeful headline",
            url="https://example.com/hope",
            source="Example",
            summary="A short summary.",
        )
    ]
    mock_sg_cls.return_value.send.return_value = MagicMock(status_code=202)
    sent_at = datetime(2026, 7, 28, 14, 0, tzinfo=UTC)
    mock_mark_sent.return_value = DigestRecord(
        id=9,
        date=digest_date,
        article_ids=[1],
        email_sent_at=sent_at,
    )

    result = send_digest(_settings(), digest_date=digest_date)

    assert result.email_sent_at == sent_at
    mock_sg_cls.assert_called_once_with("sg-test")
    mock_sg_cls.return_value.send.assert_called_once()
    mock_mark_sent.assert_called_once()


def test_send_digest_requires_config() -> None:
    """Missing SendGrid env should raise before any API call."""
    with pytest.raises(EmailDeliveryError, match="Missing email config"):
        send_digest(_settings(sendgrid_api_key=""))


@patch("app.emailer.digests_db.get_digest_by_date", return_value=None)
def test_send_digest_requires_compiled_row(_mock_get: MagicMock) -> None:
    """Send without a prior compile should fail clearly."""
    with pytest.raises(EmailDeliveryError, match="No compiled digest"):
        send_digest(_settings(), digest_date=date(2026, 7, 28))
