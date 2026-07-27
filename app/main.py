"""FastAPI application entrypoint."""

import logging
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from app import deduplicator, fetcher, sentiment, summarizer
from app.exceptions import (
    DuplicateDetectionError,
    SentimentScoringError,
    SummarizationError,
)
from app.schemas import ArticleResponse, FetchResponse, ProcessResponse
from db import articles as articles_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Good News Digest",
    description="Backend service for fetching, filtering, and delivering positive news.",
    version="0.1.0",
)


@app.post("/fetch", response_model=FetchResponse, status_code=201)
def trigger_fetch() -> FetchResponse:
    """Fetch articles from all sources and store new rows in PostgreSQL."""
    try:
        fetched_articles = fetcher.fetch_articles()
        inserted, skipped = articles_db.save_articles(fetched_articles)
    except Exception as exc:
        logger.exception("Fetch pipeline failed")
        raise HTTPException(status_code=500, detail="Fetch pipeline failed") from exc

    return FetchResponse(
        fetched=len(fetched_articles),
        inserted=inserted,
        skipped=skipped,
    )


@app.post("/process", response_model=ProcessResponse)
def trigger_process() -> ProcessResponse:
    """Run deduplication → sentiment → summarization on windowed articles.

    Each stage is isolated: a failure in one is recorded in ``stage_errors``
    and later stages still run.
    """
    duplicates_flagged = 0
    sentiment_scored = 0
    summarized = 0
    stage_errors: list[str] = []

    try:
        duplicates_flagged = deduplicator.flag_duplicates()
    except DuplicateDetectionError:
        logger.exception("Deduplication stage failed")
        stage_errors.append("deduplicate")

    try:
        sentiment_scored = sentiment.score_articles()
    except SentimentScoringError:
        logger.exception("Sentiment stage failed")
        stage_errors.append("sentiment")

    try:
        summarized = summarizer.summarize_articles()
    except SummarizationError:
        logger.exception("Summarization stage failed")
        stage_errors.append("summarize")

    return ProcessResponse(
        duplicates_flagged=duplicates_flagged,
        sentiment_scored=sentiment_scored,
        summarized=summarized,
        stage_errors=stage_errors,
    )


@app.get("/articles", response_model=list[ArticleResponse])
def list_articles(
    source: Annotated[str | None, Query(description="Filter by exact source name")] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Maximum rows to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Pagination offset")] = 0,
) -> list[ArticleResponse]:
    """Return stored articles with optional filtering and pagination."""
    try:
        records = articles_db.get_articles(source=source, limit=limit, offset=offset)
    except Exception as exc:
        logger.exception("Failed to list articles")
        raise HTTPException(status_code=500, detail="Failed to list articles") from exc

    return [ArticleResponse.model_validate(record.__dict__) for record in records]
