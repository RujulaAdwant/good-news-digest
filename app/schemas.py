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
    summarized: int = Field(description="Articles that received a Claude summary")
    stage_errors: list[str] = Field(
        default_factory=list,
        description="Names of stages that failed (other stages still ran)",
    )
