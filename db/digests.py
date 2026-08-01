"""Database operations for the digests table."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

import psycopg2

from db.connection import get_connection

logger = logging.getLogger(__name__)

_SELECT_BY_DATE = """
    SELECT id, date, article_ids, email_sent_at
    FROM digests
    WHERE date = %s
"""

_UPSERT_DIGEST = """
    INSERT INTO digests (date, article_ids)
    VALUES (%s, %s)
    ON CONFLICT (date) DO UPDATE
    SET article_ids = EXCLUDED.article_ids
    WHERE digests.email_sent_at IS NULL
    RETURNING id, date, article_ids, email_sent_at
"""

_MARK_SENT = """
    UPDATE digests
    SET email_sent_at = %s
    WHERE date = %s
      AND email_sent_at IS NULL
    RETURNING id, date, article_ids, email_sent_at
"""

_CLEAR_EMAIL_SENT = """
    UPDATE digests
    SET email_sent_at = NULL
    WHERE date = %s
    RETURNING id, date, article_ids, email_sent_at
"""


@dataclass(frozen=True)
class DigestRecord:
    """One daily digest row."""

    id: int
    date: date
    article_ids: list[int]
    email_sent_at: datetime | None = None


def _row_to_digest(row: tuple[object, ...]) -> DigestRecord:
    """Map a SELECT/RETURNING row to a DigestRecord."""
    raw_ids = row[2] or []
    return DigestRecord(
        id=row[0],  # type: ignore[arg-type]
        date=row[1],  # type: ignore[arg-type]
        article_ids=list(raw_ids),  # type: ignore[arg-type]
        email_sent_at=row[3],  # type: ignore[arg-type]
    )


def get_digest_by_date(digest_date: date) -> DigestRecord | None:
    """Fetch the digest for a calendar date, if it exists.

    Args:
        digest_date: Local digest calendar date (UNIQUE key).

    Returns:
        Matching digest, or None.

    Raises:
        psycopg2.Error: If the query fails.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_SELECT_BY_DATE, (digest_date,))
                row = cur.fetchone()
    except psycopg2.Error:
        logger.exception("Failed to fetch digest for date=%s", digest_date)
        raise

    return _row_to_digest(row) if row else None


def upsert_digest(digest_date: date, article_ids: list[int]) -> DigestRecord | None:
    """Insert or replace article_ids for a date, only if email not yet sent.

    ``ON CONFLICT ... WHERE email_sent_at IS NULL`` refuses to overwrite a
    digest that already went out — RETURNING yields no row in that case.

    Args:
        digest_date: Local digest calendar date.
        article_ids: Ordered article primary keys for the email.

    Returns:
        Upserted digest, or None if the day was already emailed.

    Raises:
        psycopg2.Error: If the upsert fails.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_UPSERT_DIGEST, (digest_date, article_ids))
                row = cur.fetchone()
    except psycopg2.Error:
        logger.exception("Failed to upsert digest date=%s", digest_date)
        raise

    if row is None:
        logger.warning(
            "Digest upsert skipped; email already sent date=%s",
            digest_date,
        )
        return None

    digest = _row_to_digest(row)
    logger.info(
        "Upserted digest id=%d date=%s article_count=%d",
        digest.id,
        digest.date,
        len(digest.article_ids),
    )
    return digest


def mark_email_sent(digest_date: date, sent_at: datetime) -> DigestRecord | None:
    """Stamp email_sent_at when SendGrid delivery succeeds.

    Args:
        digest_date: Local digest calendar date.
        sent_at: UTC timestamp of successful send.

    Returns:
        Updated digest, or None if missing / already marked sent.

    Raises:
        psycopg2.Error: If the update fails.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_MARK_SENT, (sent_at, digest_date))
                row = cur.fetchone()
    except psycopg2.Error:
        logger.exception("Failed to mark digest sent date=%s", digest_date)
        raise

    if row is None:
        return None

    digest = _row_to_digest(row)
    logger.info("Marked digest emailed id=%d date=%s", digest.id, digest.date)
    return digest


def clear_email_sent(digest_date: date) -> DigestRecord | None:
    """Clear email_sent_at so the day can be recompiled and resent.

    Dev/test helper used by ``reset_digest``. Production send/compile guards
    still refuse a stamped day unless this unlocks it first.

    Args:
        digest_date: Local digest calendar date.

    Returns:
        Updated digest with ``email_sent_at`` NULL, or None if no row exists.

    Raises:
        psycopg2.Error: If the update fails.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_CLEAR_EMAIL_SENT, (digest_date,))
                row = cur.fetchone()
    except psycopg2.Error:
        logger.exception("Failed to clear email_sent_at date=%s", digest_date)
        raise

    if row is None:
        return None

    digest = _row_to_digest(row)
    logger.info(
        "Cleared email_sent_at for digest id=%d date=%s",
        digest.id,
        digest.date,
    )
    return digest
