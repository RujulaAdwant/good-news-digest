"""Unit tests for good-news vs junk embedding relevance filtering."""

from unittest.mock import MagicMock, patch

import torch

from app.relevance import (
    filter_good_news_articles,
    is_astrology_content,
    relevance_margin_score,
)
from db.articles import ArticleRecord


def _article(article_id: int, title: str, *, full_text: str | None = None) -> ArticleRecord:
    """Minimal article for relevance tests."""
    return ArticleRecord(
        id=article_id,
        title=title,
        url=f"https://example.com/{article_id}",
        full_text=full_text if full_text is not None else title,
        sentiment_score=0.95,
    )


def test_is_astrology_content_matches_horoscope_tokens() -> None:
    """Horoscope / zodiac headlines should be flagged before embedding."""
    assert is_astrology_content("Your daily horoscope for July", None) is True
    assert is_astrology_content("Zodiac outlook this week", "stars align") is True
    assert is_astrology_content("Community garden opens downtown", None) is False


def test_relevance_margin_score_prefers_closer_prototype_set() -> None:
    """Positive margin when the article is nearer good-news than junk."""
    # Axes: 0=good, 1=corporate, 2=promo
    article = torch.tensor([1.0, 0.0, 0.0])
    good = torch.tensor([[1.0, 0.0, 0.0]])
    corporate = torch.tensor([[0.0, 1.0, 0.0]])
    promotional = torch.tensor([[0.0, 0.0, 1.0]])

    score = relevance_margin_score(
        article,
        good=good,
        corporate=corporate,
        promotional=promotional,
    )

    assert score > 0.5


def test_relevance_margin_score_promo_can_veto() -> None:
    """Promotional similarity alone should drive the margin negative."""
    article = torch.tensor([0.1, 0.0, 0.99])
    good = torch.tensor([[1.0, 0.0, 0.0]])
    corporate = torch.tensor([[0.0, 1.0, 0.0]])
    promotional = torch.tensor([[0.0, 0.0, 1.0]])

    score = relevance_margin_score(
        article,
        good=good,
        corporate=corporate,
        promotional=promotional,
    )

    assert score < 0


@patch("app.relevance._get_prototype_embeddings")
@patch("app.relevance._encode")
def test_filter_good_news_articles_drops_corporate_and_promo(
    mock_encode: MagicMock,
    mock_prototypes: MagicMock,
) -> None:
    """Corporate IR and travel-ad copy should both be filtered out."""
    articles = [
        _article(1, "Community garden feeds neighbors"),
        _article(2, "Strong Q2 performance and solid H1 results"),
        _article(3, "Mollymook hotel is the most impressively designed stay"),
    ]
    # Axes: 0=good, 1=corporate, 2=promo
    mock_encode.return_value = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.1, 0.99, 0.0],
            [0.1, 0.0, 0.99],
        ]
    )
    mock_prototypes.return_value = (
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )

    kept = filter_good_news_articles(articles, margin=0.05)

    assert [a.id for a in kept] == [1]


@patch("app.relevance._get_prototype_embeddings")
@patch("app.relevance._encode")
def test_filter_good_news_articles_drops_horoscope_before_encode(
    mock_encode: MagicMock,
    mock_prototypes: MagicMock,
) -> None:
    """Astrology keyword hits should be dropped without embedding them."""
    articles = [
        _article(1, "Community garden feeds neighbors"),
        _article(2, "Daily horoscope: what the stars say today"),
    ]
    mock_encode.return_value = torch.tensor([[1.0, 0.0, 0.0]])
    mock_prototypes.return_value = (
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )

    kept = filter_good_news_articles(articles, margin=0.05)

    assert [a.id for a in kept] == [1]
    # Only the non-astrology article should be encoded.
    mock_encode.assert_called_once()
    encoded_texts = mock_encode.call_args[0][0]
    assert len(encoded_texts) == 1


@patch("app.relevance._get_prototype_embeddings")
@patch("app.relevance._encode")
def test_filter_good_news_articles_keeps_borderline_good_news(
    mock_encode: MagicMock,
    mock_prototypes: MagicMock,
) -> None:
    """Articles clearly closer to good-news prototypes should pass."""
    articles = [_article(3, "Wildlife recovery in national park")]
    mock_encode.return_value = torch.tensor([[0.9, 0.05, 0.05]])
    mock_prototypes.return_value = (
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )

    kept = filter_good_news_articles(articles, margin=0.05)

    assert [a.id for a in kept] == [3]
