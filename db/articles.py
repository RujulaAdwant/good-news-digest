"""Database operations for the articles table."""

import logging
from dataclasses import dataclass
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_batch

from db.connection import get_connection

logger = logging.getLogger(__name__)

_ARTICLE_COLUMNS = """
    id, title, url, source, published_at, full_text,
    sentiment_score, summary, is_duplicate, digest_date, created_at
"""

_INSERT_ARTICLE = """
    INSERT INTO articles (title, url, source, published_at, full_text)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (url) DO NOTHING
    RETURNING id
"""

_SELECT_ARTICLES = f"""
    SELECT {_ARTICLE_COLUMNS}
    FROM articles
    {{where_clause}}
    ORDER BY published_at DESC NULLS LAST, created_at DESC
    LIMIT %s OFFSET %s
"""

_SELECT_IN_WINDOW = f"""
    SELECT {_ARTICLE_COLUMNS}
    FROM articles
    WHERE published_at >= %s
    ORDER BY id ASC
"""

_SELECT_FOR_SENTIMENT = f"""
    SELECT {_ARTICLE_COLUMNS}
    FROM articles
    WHERE published_at >= %s
      AND is_duplicate = FALSE
      AND sentiment_score IS NULL
    ORDER BY id ASC
"""

_SELECT_FOR_SUMMARY = f"""
    SELECT {_ARTICLE_COLUMNS}
    FROM articles
    WHERE published_at >= %s
      AND is_duplicate = FALSE
      AND sentiment_score >= %s
      AND summary IS NULL
    ORDER BY id ASC
"""

_MARK_DUPLICATES = """
    UPDATE articles
    SET is_duplicate = TRUE
    WHERE id = ANY(%s)
"""

_UPDATE_SENTIMENT = """
    UPDATE articles
    SET sentiment_score = %s
    WHERE id = %s
"""

_UPDATE_SUMMARY = """
    UPDATE articles
    SET summary = %s
    WHERE id = %s
"""


@dataclass(frozen=True)
class ArticleRecord:
    """Article fields used for inserts and reads."""

    title: str
    url: str
    source: str | None = None
    published_at: datetime | None = None
    full_text: str | None = None
    id: int | None = None
    sentiment_score: float | None = None
    summary: str | None = None
    is_duplicate: bool = False
    digest_date: datetime | None = None
    created_at: datetime | None = None


def _row_to_article(row: tuple[object, ...]) -> ArticleRecord:
    """Map a SELECT row to an ArticleRecord."""
    return ArticleRecord(
        id=row[0],  # type: ignore[arg-type]
        title=row[1],  # type: ignore[arg-type]
        url=row[2],  # type: ignore[arg-type]
        source=row[3],  # type: ignore[arg-type]
        published_at=row[4],  # type: ignore[arg-type]
        full_text=row[5],  # type: ignore[arg-type]
        sentiment_score=row[6],  # type: ignore[arg-type]
        summary=row[7],  # type: ignore[arg-type]
        is_duplicate=row[8],  # type: ignore[arg-type]
        digest_date=row[9],  # type: ignore[arg-type]
        created_at=row[10],  # type: ignore[arg-type]
    )


def save_articles(articles: list[ArticleRecord]) -> tuple[int, int]:
    """Insert articles, skipping rows with duplicate URLs.

    Args:
        articles: Article records to persist.

    Returns:
        A tuple of (inserted_count, skipped_count).

    Raises:
        psycopg2.Error: If the database operation fails.
    """
    if not articles:
        return 0, 0

    inserted = 0
    skipped = 0

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for article in articles:
                    cur.execute(
                        _INSERT_ARTICLE,
                        (
                            article.title,
                            article.url,
                            article.source,
                            article.published_at,
                            article.full_text,
                        ),
                    )
                    if cur.fetchone():
                        inserted += 1
                    else:
                        skipped += 1
    except psycopg2.Error:
        logger.exception("Failed to save %d articles", len(articles))
        raise

    logger.info(
        "Saved articles: inserted=%d skipped=%d total=%d",
        inserted,
        skipped,
        len(articles),
    )
    return inserted, skipped


def get_articles(
    *,
    source: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ArticleRecord]:
    """Fetch stored articles with optional source filtering.

    Args:
        source: Optional exact source name filter.
        limit: Maximum rows to return (capped by caller).
        offset: Number of rows to skip for pagination.

    Returns:
        Matching article records ordered by recency.

    Raises:
        psycopg2.Error: If the database query fails.
    """
    clauses: list[str] = []
    params: list[object] = []

    if source:
        clauses.append("source = %s")
        params.append(source)

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = _SELECT_ARTICLES.format(where_clause=where_clause)
    params.extend([limit, offset])

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
    except psycopg2.Error:
        logger.exception("Failed to fetch articles")
        raise

    return [_row_to_article(row) for row in rows]


def get_articles_in_window(cutoff: datetime) -> list[ArticleRecord]:
    """Fetch articles published on or after cutoff, oldest id first.

    Used by deduplication so older rows become the canonical copy.

    Args:
        cutoff: Inclusive UTC lower bound on published_at.

    Returns:
        Articles in the time window ordered by id ascending.

    Raises:
        psycopg2.Error: If the database query fails.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_SELECT_IN_WINDOW, (cutoff,))
                rows = cur.fetchall()
    except psycopg2.Error:
        logger.exception("Failed to fetch articles in window cutoff=%s", cutoff)
        raise

    return [_row_to_article(row) for row in rows]


def get_articles_needing_sentiment(cutoff: datetime) -> list[ArticleRecord]:
    """Fetch non-duplicate articles in window that lack a sentiment score.

    Args:
        cutoff: Inclusive UTC lower bound on published_at.

    Returns:
        Articles ready for sentiment scoring.

    Raises:
        psycopg2.Error: If the database query fails.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_SELECT_FOR_SENTIMENT, (cutoff,))
                rows = cur.fetchall()
    except psycopg2.Error:
        logger.exception("Failed to fetch articles needing sentiment")
        raise

    return [_row_to_article(row) for row in rows]


def get_articles_needing_summary(
    cutoff: datetime,
    sentiment_threshold: float,
) -> list[ArticleRecord]:
    """Fetch positive, non-duplicate articles in window without a summary.

    Args:
        cutoff: Inclusive UTC lower bound on published_at.
        sentiment_threshold: Minimum sentiment_score to include (starting point: 0.6).

    Returns:
        Articles eligible for Claude summarization.

    Raises:
        psycopg2.Error: If the database query fails.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_SELECT_FOR_SUMMARY, (cutoff, sentiment_threshold))
                rows = cur.fetchall()
    except psycopg2.Error:
        logger.exception("Failed to fetch articles needing summary")
        raise

    return [_row_to_article(row) for row in rows]


def mark_duplicates(article_ids: list[int]) -> int:
    """Set is_duplicate=TRUE for the given article ids.

    Args:
        article_ids: Primary keys to flag.

    Returns:
        Number of rows updated.

    Raises:
        psycopg2.Error: If the database update fails.
    """
    if not article_ids:
        return 0

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_MARK_DUPLICATES, (article_ids,))
                updated = cur.rowcount
    except psycopg2.Error:
        logger.exception("Failed to mark %d articles as duplicates", len(article_ids))
        raise

    logger.info("Marked %d articles as duplicates", updated)
    return updated


def update_sentiment_scores(scores: list[tuple[int, float]]) -> int:
    """Persist sentiment scores for articles.

    Args:
        scores: List of (article_id, sentiment_score) pairs.

    Returns:
        Number of rows updated.

    Raises:
        psycopg2.Error: If the database update fails.
    """
    if not scores:
        return 0

    # INTERVIEW: understand this — batch UPDATE avoids N+1 round-trips
    params = [(score, article_id) for article_id, score in scores]
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, _UPDATE_SENTIMENT, params)
                updated = len(scores)
    except psycopg2.Error:
        logger.exception("Failed to update %d sentiment scores", len(scores))
        raise

    logger.info("Updated sentiment scores for %d articles", updated)
    return updated


def update_summaries(summaries: list[tuple[int, str]]) -> int:
    """Persist Claude summaries for articles.

    Args:
        summaries: List of (article_id, summary_text) pairs.

    Returns:
        Number of rows updated.

    Raises:
        psycopg2.Error: If the database update fails.
    """
    if not summaries:
        return 0

    params = [(summary, article_id) for article_id, summary in summaries]
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                execute_batch(cur, _UPDATE_SUMMARY, params)
                updated = len(summaries)
    except psycopg2.Error:
        logger.exception("Failed to update %d summaries", len(summaries))
        raise

    logger.info("Updated summaries for %d articles", updated)
    return updated

