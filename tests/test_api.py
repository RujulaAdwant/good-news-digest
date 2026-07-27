"""Tests for FastAPI endpoints."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from db.articles import ArticleRecord

client = TestClient(app)


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
