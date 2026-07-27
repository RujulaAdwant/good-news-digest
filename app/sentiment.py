"""Sentiment scoring for articles using a pretrained HuggingFace model.

Model: ``distilbert-base-uncased-finetuned-sst-2-english``.

Why this model: DistilBERT is a smaller, faster cousin of BERT. It was
*fine-tuned* on SST-2 (Stanford Sentiment Treebank) — movie-review style
positive/negative labels. We did not train it; we reuse public weights.

Output mapping (Decision 5A):
  pipeline returns {label: POSITIVE|NEGATIVE, score: confidence in that label}.
  We store a single "positive probability":
    POSITIVE → score
    NEGATIVE → 1 - score
  So sentiment_score >= 0.6 means "model leans positive" (threshold is tunable).
"""

from __future__ import annotations

import logging
from typing import Any

from transformers import pipeline

from app.config import Settings, get_settings
from app.exceptions import SentimentScoringError
from app.fetcher import fetch_cutoff, text_for_nlp
from db import articles as articles_db

logger = logging.getLogger(__name__)

_sentiment_pipeline: Any | None = None


def get_sentiment_pipeline() -> Any:
    """Return the shared sentiment pipeline, loading it once.

    Returns:
        HuggingFace text-classification pipeline.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info(
            "Loading sentiment model distilbert-base-uncased-finetuned-sst-2-english"
        )
        # INTERVIEW: understand this — pretrained + fine-tuned, not trained from scratch
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    return _sentiment_pipeline


def positive_probability(label: str, confidence: float) -> float:
    """Convert SST-2 label+confidence into a positive-leaning score in [0, 1].

    Args:
        label: ``POSITIVE`` or ``NEGATIVE`` (case-insensitive).
        confidence: Model confidence in that label, typically in (0.5, 1].

    Returns:
        Estimated P(positive).
    """
    normalized = label.upper()
    if normalized == "POSITIVE":
        return confidence
    if normalized == "NEGATIVE":
        return 1.0 - confidence
    logger.warning("Unexpected sentiment label=%s; treating as 0.5", label)
    return 0.5


def score_text(text: str) -> float:
    """Score a single text string for positive sentiment.

    Args:
        text: Title + first paragraph (or title alone).

    Returns:
        Positive probability in [0, 1].

    Raises:
        SentimentScoringError: If the model call fails.
    """
    try:
        result = get_sentiment_pipeline()(text[:512])[0]
        return positive_probability(result["label"], float(result["score"]))
    except SentimentScoringError:
        raise
    except Exception as exc:
        raise SentimentScoringError("Sentiment model inference failed") from exc


def score_articles(settings: Settings | None = None) -> int:
    """Score unscored, non-duplicate articles in the fetch window.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Number of articles whose ``sentiment_score`` was written.

    Raises:
        SentimentScoringError: If loading candidates or bulk scoring fails hard.
    """
    cfg = settings or get_settings()
    cutoff = fetch_cutoff(cfg)

    try:
        candidates = articles_db.get_articles_needing_sentiment(cutoff)
    except Exception as exc:
        raise SentimentScoringError("Failed to load articles for sentiment") from exc

    if not candidates:
        logger.info("No articles need sentiment scoring")
        return 0

    scores: list[tuple[int, float]] = []
    for article in candidates:
        if article.id is None:
            continue
        try:
            text = text_for_nlp(article.title, article.full_text)
            score = score_text(text)
            scores.append((article.id, score))
            logger.info(
                "Sentiment scored url=%s score=%.3f",
                article.url,
                score,
            )
        except SentimentScoringError:
            # One bad article should not abort the stage.
            logger.exception(
                "Sentiment scoring failed for url=%s stage=sentiment",
                article.url,
            )

    try:
        return articles_db.update_sentiment_scores(scores)
    except Exception as exc:
        raise SentimentScoringError("Failed to persist sentiment scores") from exc
