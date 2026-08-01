"""Tests for FastAPI endpoints."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from db.articles import ArticleRecord

client = TestClient(app)


def test_get_health_returns_ok() -> None:
    """GET /health should report liveness for deploy probes."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.main.articles_db.save_articles")
@patch("app.main.fetcher.fetch_articles")
def test_post_fetch_returns_counts(
    mock_fetch_articles: object,
    mock_save_articles: object,
) -> None:
    """POST /fetch should return fetched, inserted, and skipped counts."""
    mock_fetch_articles.return_value = [
        ArticleRecord(
            title="Good headline",
            url="https://example.com/good",
            published_at=datetime.now(UTC) - timedelta(hours=1),
        ),
        ArticleRecord(
            title="Duplicate headline",
            url="https://example.com/duplicate",
            published_at=datetime.now(UTC) - timedelta(hours=2),
        ),
    ]
    mock_save_articles.return_value = (1, 1)

    response = client.post("/fetch")

    assert response.status_code == 201
    assert response.json() == {"fetched": 2, "inserted": 1, "skipped": 1}


@patch("app.main.articles_db.get_articles")
def test_get_articles_returns_stored_rows(mock_get_articles: object) -> None:
    """GET /articles should serialize stored article records."""
    created_at = datetime.now(UTC)
    mock_get_articles.return_value = [
        ArticleRecord(
            id=1,
            title="Stored headline",
            url="https://example.com/stored",
            source="Example Source",
            published_at=created_at - timedelta(hours=1),
            full_text="Snippet text",
            created_at=created_at,
        )
    ]

    response = client.get("/articles?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["title"] == "Stored headline"
    assert payload[0]["url"] == "https://example.com/stored"
    assert payload[0]["is_duplicate"] is False


@patch("app.main.summarizer.summarize_articles", return_value=2)
@patch("app.main.sentiment.score_articles", return_value=5)
@patch("app.main.deduplicator.flag_duplicates", return_value=1)
def test_post_process_runs_all_stages(
    _mock_dedup: object,
    _mock_sentiment: object,
    _mock_summarize: object,
) -> None:
    """POST /process should aggregate counts from each NLP stage."""
    response = client.post("/process")

    assert response.status_code == 200
    assert response.json() == {
        "duplicates_flagged": 1,
        "sentiment_scored": 5,
        "summarized": 2,
        "stage_errors": [],
    }


@patch("app.main.summarizer.summarize_articles", return_value=0)
@patch("app.main.sentiment.score_articles", return_value=3)
@patch("app.main.deduplicator.flag_duplicates")
def test_post_process_continues_after_stage_failure(
    mock_dedup: object,
    _mock_sentiment: object,
    _mock_summarize: object,
) -> None:
    """A failed dedup stage should be recorded without blocking later stages."""
    from app.exceptions import DuplicateDetectionError

    mock_dedup.side_effect = DuplicateDetectionError("boom")

    response = client.post("/process")

    assert response.status_code == 200
    body = response.json()
    assert body["duplicates_flagged"] == 0
    assert body["sentiment_scored"] == 3
    assert body["summarized"] == 0
    assert body["stage_errors"] == ["deduplicate"]


@patch("app.main.digest.compile_digest")
def test_post_digest_returns_compiled_preview(mock_compile: object) -> None:
    """POST /digest should return selected article previews without sending."""
    from datetime import date

    from app.digest import CompiledDigest

    mock_compile.return_value = CompiledDigest(
        digest_id=7,
        digest_date=date(2026, 7, 28),
        articles=[
            ArticleRecord(
                id=11,
                title="Hopeful story",
                url="https://example.com/hope",
                source="Example",
                sentiment_score=0.91,
                summary="A short summary.",
            )
        ],
    )

    response = client.post("/digest")

    assert response.status_code == 201
    body = response.json()
    assert body["digest_id"] == 7
    assert body["digest_date"] == "2026-07-28"
    assert body["article_count"] == 1
    assert body["article_ids"] == [11]
    assert body["articles"][0]["title"] == "Hopeful story"


@patch("app.main.emailer.send_digest")
@patch("app.main.get_settings")
def test_post_send_returns_delivery_metadata(
    mock_settings: object,
    mock_send: object,
) -> None:
    """POST /send should stamp and report SendGrid delivery."""
    from datetime import UTC, date, datetime

    from app.config import Settings
    from db.digests import DigestRecord

    mock_settings.return_value = Settings(
        database_url="postgresql://digest:digest@localhost:5432/good_news_digest",
        thenewsapi_key="",
        anthropic_api_key="",
        sendgrid_api_key="sg",
        sendgrid_from_email="from@example.com",
        digest_recipient_email="you@example.com",
        similarity_threshold=0.85,
        sentiment_threshold=0.6,
        sentiment_target=None,
        digest_size=5,
        topic_diversity_threshold=0.7,
        relevance_margin=0.05,
        digest_hour=7,
        digest_timezone="America/Los_Angeles",
        fetch_window_hours=24,
        allow_digest_reset=False,
        digest_banner_url="",
    )
    sent_at = datetime(2026, 7, 28, 14, 0, tzinfo=UTC)
    mock_send.return_value = DigestRecord(
        id=7,
        date=date(2026, 7, 28),
        article_ids=[11],
        email_sent_at=sent_at,
    )

    response = client.post("/send")

    assert response.status_code == 200
    body = response.json()
    assert body["digest_id"] == 7
    assert body["article_count"] == 1
    assert body["recipient"] == "you@example.com"
    assert body["email_sent_at"].startswith("2026-07-28")


@patch("app.main.digest.reset_digest")
def test_delete_digest_returns_unlock_metadata(mock_reset: object) -> None:
    """DELETE /digest should report that today's send lock was cleared."""
    from datetime import date

    from app.digest import DigestResetResult

    mock_reset.return_value = DigestResetResult(
        digest_id=7,
        digest_date=date(2026, 7, 28),
        email_was_sent=True,
        articles_unstamped=3,
    )

    response = client.delete("/digest")

    assert response.status_code == 200
    body = response.json()
    assert body["digest_id"] == 7
    assert body["digest_date"] == "2026-07-28"
    assert body["email_was_sent"] is True
    assert body["articles_unstamped"] == 3
    assert "unlocked" in body["message"].lower()


@patch("app.main.digest.reset_digest")
def test_delete_digest_forbidden_when_disabled(mock_reset: object) -> None:
    """DELETE /digest should 403 when ALLOW_DIGEST_RESET is false."""
    from app.exceptions import DigestResetError

    mock_reset.side_effect = DigestResetError(
        "Digest reset is disabled; set ALLOW_DIGEST_RESET=true to enable"
    )

    response = client.delete("/digest")

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


@patch("app.main.digest.reset_digest")
def test_delete_digest_not_found(mock_reset: object) -> None:
    """DELETE /digest should 404 when no digest exists for today."""
    from app.exceptions import DigestResetError

    mock_reset.side_effect = DigestResetError("No digest found for 2026-07-28")

    response = client.delete("/digest")

    assert response.status_code == 404
