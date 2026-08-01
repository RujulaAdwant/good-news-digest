"""Daily APScheduler job for the Good News Digest pipeline.

Runs as a blocking process (separate from uvicorn). Each stage is isolated so
one failure does not skip later stages unless they depend on earlier output
(e.g. send still needs a compiled digest).

Usage:
    python scheduler.py
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from app import deduplicator, digest, emailer, fetcher, sentiment, summarizer
from app.config import get_settings
from app.exceptions import (
    DigestCompileError,
    DuplicateDetectionError,
    EmailDeliveryError,
    SentimentScoringError,
    SummarizationError,
)
from db import articles as articles_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_daily_pipeline() -> None:
    """Execute fetch → NLP → compile → send for today's digest.

    Each stage logs and continues on failure where safe. Compile/send failures
    are logged; send is skipped if compile did not succeed in this run and no
    prior unsent digest exists (send_digest loads from DB).
    """
    logger.info("Daily pipeline starting")

    try:
        fetched = fetcher.fetch_articles()
        inserted, skipped = articles_db.save_articles(fetched)
        logger.info(
            "Fetch complete fetched=%d inserted=%d skipped=%d",
            len(fetched),
            inserted,
            skipped,
        )
    except Exception:
        logger.exception("Fetch stage failed")

    try:
        flagged = deduplicator.flag_duplicates()
        logger.info("Dedup complete flagged=%d", flagged)
    except DuplicateDetectionError:
        logger.exception("Dedup stage failed")

    try:
        scored = sentiment.score_articles()
        logger.info("Sentiment complete scored=%d", scored)
    except SentimentScoringError:
        logger.exception("Sentiment stage failed")

    try:
        summarized = summarizer.summarize_articles()
        logger.info("Summarize complete summarized=%d", summarized)
    except SummarizationError:
        logger.exception("Summarize stage failed")

    try:
        compiled = digest.compile_digest()
        logger.info(
            "Compile complete digest_id=%d articles=%d",
            compiled.digest_id,
            len(compiled.articles),
        )
    except DigestCompileError:
        logger.exception("Compile stage failed")

    try:
        sent = emailer.send_digest()
        logger.info(
            "Send complete digest_id=%d date=%s",
            sent.id,
            sent.date,
        )
    except EmailDeliveryError:
        logger.exception("Send stage failed")

    logger.info("Daily pipeline finished")


def main() -> None:
    """Start the blocking scheduler for DIGEST_HOUR in DIGEST_TIMEZONE."""
    settings = get_settings()
    scheduler = BlockingScheduler(timezone=settings.digest_timezone)
    trigger = CronTrigger(
        hour=settings.digest_hour,
        minute=0,
        timezone=settings.digest_timezone,
    )
    scheduler.add_job(run_daily_pipeline, trigger=trigger, id="daily_digest")
    logger.info(
        "Scheduler started hour=%d timezone=%s",
        settings.digest_hour,
        settings.digest_timezone,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
