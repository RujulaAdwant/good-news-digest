"""Unit tests for embedding-based deduplication."""

from unittest.mock import MagicMock, patch

import torch

from app.deduplicator import find_duplicate_ids, flag_duplicates
from app.config import Settings
from db.articles import ArticleRecord


def _article(article_id: int, title: str, url: str) -> ArticleRecord:
    """Build a minimal article with a stable id."""
    return ArticleRecord(
        id=article_id,
        title=title,
        url=url,
        full_text="First paragraph about the story.",
    )


@patch("app.deduplicator._encode")
def test_find_duplicate_ids_flags_near_duplicates(mock_encode: MagicMock) -> None:
    """Near-identical embeddings above threshold should flag the newer id."""
    # Three unit vectors: 0 and 1 almost same; 2 orthogonal.
    mock_encode.return_value = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.141, 0.0],  # high cosine with first after normalize-ish
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # Re-normalize so cosine is exact
    mock_encode.return_value = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],  # identical → similarity 1.0
                [0.0, 1.0],  # orthogonal → similarity 0.0
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=1,
    )

    articles = [
        _article(1, "Vaccine breakthrough announced", "https://a.example/1"),
        _article(2, "Scientists announce vaccine breakthrough", "https://b.example/2"),
        _article(3, "Local sports team wins championship", "https://c.example/3"),
    ]

    duplicate_ids = find_duplicate_ids(articles, threshold=0.85)

    assert duplicate_ids == [2]


@patch("app.deduplicator._encode")
def test_find_duplicate_ids_keeps_distinct_articles(mock_encode: MagicMock) -> None:
    """Clearly distinct embeddings should not flag any duplicates."""
    mock_encode.return_value = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=1,
    )
    articles = [
        _article(10, "Climate accord signed", "https://a.example/10"),
        _article(11, "New park opens downtown", "https://b.example/11"),
        _article(12, "Researchers map coral reefs", "https://c.example/12"),
    ]

    assert find_duplicate_ids(articles, threshold=0.85) == []


@patch("app.deduplicator.articles_db.mark_duplicates", return_value=1)
@patch("app.deduplicator.find_duplicate_ids", return_value=[42])
@patch("app.deduplicator.articles_db.get_articles_in_window")
def test_flag_duplicates_persists_ids(
    mock_get: MagicMock,
    mock_find: MagicMock,
    mock_mark: MagicMock,
    test_settings: Settings,
) -> None:
    """flag_duplicates should load the window, find ids, and mark them."""
    mock_get.return_value = [_article(42, "Dup", "https://example.com/dup")]

    updated = flag_duplicates(test_settings)

    assert updated == 1
    mock_find.assert_called_once()
    mock_mark.assert_called_once_with([42])
