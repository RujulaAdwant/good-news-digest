"""FastAPI application entrypoint."""

import logging
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from app import deduplicator, digest, emailer, fetcher, sentiment, summarizer
from app.config import get_settings
from app.exceptions import (
    DigestCompileError,
    DigestResetError,
    DuplicateDetectionError,
    EmailDeliveryError,
    SentimentScoringError,
    SummarizationError,
)
from app.schemas import (
    ArticleResponse,
    DigestArticleSummary,
    DigestResetResponse,
    DigestResponse,
    FetchResponse,
    ProcessResponse,
    SendResponse,
)
from app.summarizer import is_usable_summary
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


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for local runs and Railway health checks."""
    return {"status": "ok"}


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
    """Run deduplication → sentiment → digest-pick summarization.

    Summarization only covers the final digest selection (top sentiment,
    topic-diverse), not every positive article.

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


@app.post("/digest", response_model=DigestResponse, status_code=201)
def trigger_digest() -> DigestResponse:
    """Compile today's digest without sending email.

    Selects topic-diverse positive stories, upserts the ``digests`` row,
    and stamps ``articles.digest_date``. Usable Claude summaries are
    included in the preview when present; missing ones are ``null``.
    Prefer ``POST /process`` (summarize) before this, then ``POST /send``
    after inspecting.
    """
    try:
        compiled = digest.compile_digest()
    except DigestCompileError as exc:
        logger.exception("Digest compile failed")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Digest compile failed unexpectedly")
        raise HTTPException(status_code=500, detail="Digest compile failed") from exc

    return DigestResponse(
        digest_id=compiled.digest_id,
        digest_date=compiled.digest_date,
        article_count=len(compiled.articles),
        article_ids=[a.id for a in compiled.articles if a.id is not None],
        articles=[
            DigestArticleSummary(
                id=article.id,  # type: ignore[arg-type]
                title=article.title,
                url=article.url,
                source=article.source,
                sentiment_score=article.sentiment_score,
                summary=article.summary if is_usable_summary(article.summary) else None,
            )
            for article in compiled.articles
            if article.id is not None
        ],
    )


@app.delete("/digest", response_model=DigestResetResponse)
def reset_digest() -> DigestResetResponse:
    """Unlock today's digest for recompile and resend (dev/test only).

    Clears ``email_sent_at`` and article ``digest_date`` stamps for the
    local calendar day. Requires ``ALLOW_DIGEST_RESET=true``. After reset,
    call ``POST /digest`` then ``POST /send`` again.
    """
    try:
        result = digest.reset_digest()
    except DigestResetError as exc:
        logger.exception("Digest reset failed")
        detail = str(exc)
        if "disabled" in detail:
            status = 403
        elif "No digest found" in detail:
            status = 404
        else:
            status = 400
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        logger.exception("Digest reset failed unexpectedly")
        raise HTTPException(status_code=500, detail="Digest reset failed") from exc

    return DigestResetResponse(
        digest_id=result.digest_id,
        digest_date=result.digest_date,
        email_was_sent=result.email_was_sent,
        articles_unstamped=result.articles_unstamped,
        message="Digest unlocked for recompile and resend",
    )


@app.post("/send", response_model=SendResponse)
def trigger_send() -> SendResponse:
    """Email the compiled digest for today via SendGrid.

    Requires a prior ``POST /digest`` for today's local calendar date.
    """
    settings = get_settings()
    try:
        sent = emailer.send_digest(settings)
    except EmailDeliveryError as exc:
        logger.exception("Digest send failed")
        detail = str(exc)
        status = 404 if "No compiled digest" in detail else 400
        if "already sent" in detail:
            status = 409
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        logger.exception("Digest send failed unexpectedly")
        raise HTTPException(status_code=500, detail="Digest send failed") from exc

    assert sent.email_sent_at is not None
    return SendResponse(
        digest_id=sent.id,
        digest_date=sent.date,
        article_count=len(sent.article_ids),
        email_sent_at=sent.email_sent_at,
        recipient=settings.digest_recipient_email,
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
