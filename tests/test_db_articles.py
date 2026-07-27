"""Integration tests for article persistence."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from db.articles import ArticleRecord, get_articles, save_articles
from db.connection import get_connection


@pytest.fixture
def unique_article() -> ArticleRecord:
    """Build a unique article for database integration tests."""
    suffix = uuid4().hex
    return ArticleRecord(
        title=f"Test article {suffix}",
        url=f"https://example.com/{suffix}",
        source="Test Source",
        published_at=datetime.now(UTC) - timedelta(hours=1),
        full_text="A positive development in testing.",
    )


def test_save_articles_inserts_new_rows(unique_article: ArticleRecord) -> None:
    """save_articles should insert a new article and return inserted=1."""
    inserted, skipped = save_articles([unique_article])

    assert inserted == 1
    assert skipped == 0


def test_save_articles_skips_duplicate_urls(unique_article: ArticleRecord) -> None:
    """Duplicate URLs should be ignored via ON CONFLICT DO NOTHING."""
    first_inserted, first_skipped = save_articles([unique_article])
    duplicate = ArticleRecord(
        title="Different title",
        url=unique_article.url,
        source="Other Source",
        published_at=unique_article.published_at,
        full_text="Different body",
    )
    second_inserted, second_skipped = save_articles([duplicate])

    assert first_inserted == 1
    assert first_skipped == 0
    assert second_inserted == 0
    assert second_skipped == 1


def test_get_articles_filters_by_source(unique_article: ArticleRecord) -> None:
    """get_articles should support exact source filtering."""
    save_articles([unique_article])

    matches = get_articles(source=unique_article.source, limit=10)
    non_matches = get_articles(source="Definitely Not This Source", limit=10)

    assert any(row.url == unique_article.url for row in matches)
    assert all(row.url != unique_article.url for row in non_matches)


@pytest.fixture(autouse=True)
def cleanup_test_articles(unique_article: ArticleRecord) -> None:
    """Remove test rows after each database test."""
    yield
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM articles WHERE url = %s", (unique_article.url,))
