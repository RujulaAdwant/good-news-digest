"""Pydantic request/response schemas for the FastAPI layer."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class ArticleResponse(BaseModel):
    """A stored article returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    url: str
    source: str | None = None
    published_at: datetime | None = None
    full_text: str | None = None
    sentiment_score: float | None = None
    summary: str | None = None
    is_duplicate: bool = False
    digest_date: date | None = None
    created_at: datetime


class FetchResponse(BaseModel):
    """Result of a fetch-and-store operation."""

    fetched: int = Field(description="Articles returned by sources after time filtering")
    inserted: int = Field(description="New articles inserted into the database")
    skipped: int = Field(description="Articles skipped due to duplicate URLs")


class ProcessResponse(BaseModel):
    """Result of running the NLP pipeline stages."""

    duplicates_flagged: int = Field(description="Articles marked is_duplicate=True")
    sentiment_scored: int = Field(description="Articles that received a sentiment_score")
    summarized: int = Field(
        description="Digest picks that received a new Claude summary"
    )
    stage_errors: list[str] = Field(
        default_factory=list,
        description="Names of stages that failed (other stages still ran)",
    )


class DigestArticleSummary(BaseModel):
    """Compact article preview returned after compiling a digest."""

    id: int
    title: str
    url: str
    source: str | None = None
    sentiment_score: float | None = None
    summary: str | None = None


class DigestResponse(BaseModel):
    """Result of compiling (but not sending) today's digest."""

    digest_id: int = Field(description="Primary key of the digests row")
    digest_date: date = Field(description="Local calendar date for this digest")
    article_count: int = Field(description="Number of stories selected")
    article_ids: list[int] = Field(description="Ordered article ids in the digest")
    articles: list[DigestArticleSummary] = Field(
        description="Preview of selected stories for inspection before send",
    )


class SendResponse(BaseModel):
    """Result of emailing a compiled digest via SendGrid."""

    digest_id: int
    digest_date: date
    article_count: int = Field(description="Stories included in the email")
    email_sent_at: datetime = Field(description="UTC timestamp stamped after send")
    recipient: str = Field(description="Address the digest was sent to")


class DigestResetResponse(BaseModel):
    """Result of unlocking today's digest for another compile/send cycle."""

    digest_id: int
    digest_date: date
    email_was_sent: bool = Field(
        description="Whether email_sent_at was set before the reset",
    )
    articles_unstamped: int = Field(
        description="Articles that had digest_date cleared",
    )
    message: str = Field(
        description="Human-readable confirmation",
    )
