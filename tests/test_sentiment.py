"""Unit tests for sentiment scoring."""

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.sentiment import positive_probability, score_articles, score_text
from db.articles import ArticleRecord


def test_positive_probability_from_positive_label() -> None:
    """POSITIVE confidence should be stored as-is."""
    assert positive_probability("POSITIVE", 0.91) == pytest.approx(0.91)


def test_positive_probability_from_negative_label() -> None:
    """NEGATIVE confidence should invert to P(positive)."""
    assert positive_probability("NEGATIVE", 0.8) == pytest.approx(0.2)


@patch("app.sentiment.get_sentiment_pipeline")
def test_score_text_maps_pipeline_output(mock_get_pipeline: MagicMock) -> None:
    """score_text should convert pipeline output via positive_probability."""
    mock_get_pipeline.return_value = MagicMock(
        return_value=[{"label": "NEGATIVE", "score": 0.7}]
    )

    assert score_text("Terrible news everywhere.") == pytest.approx(0.3)


@patch("app.sentiment.articles_db.update_sentiment_scores", return_value=1)
@patch("app.sentiment.score_text", return_value=0.82)
@patch("app.sentiment.articles_db.get_articles_needing_sentiment")
def test_score_articles_writes_scores(
    mock_get: MagicMock,
    mock_score: MagicMock,
    mock_update: MagicMock,
    test_settings: Settings,
) -> None:
    """score_articles should score candidates and batch-update the DB."""
    mock_get.return_value = [
        ArticleRecord(
            id=7,
            title="Hopeful breakthrough",
            url="https://example.com/hope",
            full_text="Researchers made progress.",
        )
    ]

    updated = score_articles(test_settings)

    assert updated == 1
    mock_update.assert_called_once_with([(7, 0.82)])
    mock_score.assert_called_once()
