"""Unit tests for digest candidate selection and compilation."""

from dataclasses import replace
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.digest import (
    compile_digest,
    pick_diverse_articles,
    rank_digest_candidates,
    reset_digest,
    select_articles_for_digest,
)
from app.exceptions import DigestCompileError, DigestResetError, DigestSelectionError
from db.articles import ArticleRecord
from db.digests import DigestRecord


def _article(
    article_id: int,
    title: str,
    *,
    score: float,
    url: str | None = None,
    summary: str | None = None,
    source: str | None = None,
) -> ArticleRecord:
    """Build a scored article for selection tests."""
    return ArticleRecord(
        id=article_id,
        title=title,
        url=url or f"https://example.com/{article_id}",
        source=source or f"Source {article_id}",
        full_text=title,
        sentiment_score=score,
        summary=summary,
    )


@patch("app.digest._encode")
def test_pick_diverse_articles_caps_at_digest_size(mock_encode: MagicMock) -> None:
    """Selection should stop at digest_size even with many distinct topics."""
    candidates = [
        _article(1, "Climate breakthrough A", score=0.99),
        _article(2, "School lunch program", score=0.95),
        _article(3, "Wildlife recovery", score=0.90),
        _article(4, "Clean energy jobs", score=0.85),
    ]
    # Orthogonal-ish unit vectors so every pair is dissimilar.
    mock_encode.return_value = torch.eye(4)

    picked = pick_diverse_articles(
        candidates,
        digest_size=3,
        diversity_threshold=0.70,
    )

    assert [a.id for a in picked] == [1, 2, 3]


@patch("app.digest._encode")
def test_pick_diverse_articles_skips_overlapping_topics(
    mock_encode: MagicMock,
) -> None:
    """A high-sentiment story too similar to a kept one should be skipped."""
    candidates = [
        _article(1, "Climate win one", score=0.99),
        _article(2, "Climate win two", score=0.98),
        _article(3, "Local library opens", score=0.80),
    ]
    # Article 2 is nearly identical to article 1; article 3 is orthogonal.
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    mock_encode.return_value = embeddings

    picked = pick_diverse_articles(
        candidates,
        digest_size=5,
        diversity_threshold=0.70,
    )

    assert [a.id for a in picked] == [1, 3]


@patch("app.digest._encode")
def test_pick_diverse_articles_prefers_unique_sources(
    mock_encode: MagicMock,
) -> None:
    """Same-outlet stories should yield to a later different source."""
    candidates = [
        _article(1, "GNN story one", score=0.90, source="Good News Network"),
        _article(2, "GNN story two", score=0.88, source="Good News Network"),
        _article(3, "AP recovery", score=0.70, source="Associated Press"),
    ]
    mock_encode.return_value = torch.eye(3)

    picked = pick_diverse_articles(
        candidates,
        digest_size=2,
        diversity_threshold=0.70,
    )

    assert [a.id for a in picked] == [1, 3]
    assert {a.source for a in picked} == {"Good News Network", "Associated Press"}


@patch("app.digest._encode")
def test_pick_diverse_articles_relaxes_source_if_needed(
    mock_encode: MagicMock,
) -> None:
    """If every candidate shares a source, still fill digest_size."""
    candidates = [
        _article(1, "Story A", score=0.90, source="Good News Network"),
        _article(2, "Story B", score=0.85, source="Good News Network"),
        _article(3, "Story C", score=0.80, source="Good News Network"),
    ]
    mock_encode.return_value = torch.eye(3)

    picked = pick_diverse_articles(
        candidates,
        digest_size=3,
        diversity_threshold=0.70,
    )

    assert [a.id for a in picked] == [1, 2, 3]

def test_rank_digest_candidates_max_positive_when_no_target() -> None:
    """With no target, higher sentiment_score should rank first."""
    candidates = [
        _article(1, "Mid", score=0.70),
        _article(2, "High", score=0.95),
        _article(3, "Low", score=0.55),
    ]
    ranked = rank_digest_candidates(candidates, sentiment_target=None)
    assert [a.id for a in ranked] == [2, 1, 3]


def test_rank_digest_candidates_prefers_near_target() -> None:
    """With a Glass target, mid-positive should beat extreme cheerleading."""
    candidates = [
        _article(1, "Cheerleading", score=0.99),
        _article(2, "Glass tone", score=0.66),
        _article(3, "Barely positive", score=0.53),
    ]
    ranked = rank_digest_candidates(candidates, sentiment_target=0.65)
    assert [a.id for a in ranked] == [2, 3, 1]


@patch("app.digest.filter_good_news_articles", side_effect=lambda arts, margin: arts)
@patch("app.digest.articles_db.get_digest_candidates")
def test_select_articles_for_digest_loads_then_picks(
    mock_get: MagicMock,
    mock_filter: MagicMock,
    test_settings,
) -> None:
    """select_articles_for_digest should load, relevance-filter, then greedy pick."""
    settings = test_settings
    mock_get.return_value = [
        _article(1, "Good news A", score=0.9),
        _article(2, "Good news B", score=0.8),
    ]

    with patch("app.digest.pick_diverse_articles") as mock_pick:
        mock_pick.return_value = [mock_get.return_value[0]]
        result = select_articles_for_digest(settings)

    mock_get.assert_called_once()
    mock_filter.assert_called_once()
    mock_pick.assert_called_once()
    # Ranked list is what diversity sees (max-positive when target is None).
    ranked_arg = mock_pick.call_args.args[0]
    assert [a.id for a in ranked_arg] == [1, 2]
    assert len(result) == 1


@patch("app.digest.filter_good_news_articles", side_effect=lambda arts, margin: arts)
@patch("app.digest.articles_db.get_digest_candidates")
@patch("app.digest.pick_diverse_articles")
def test_select_uses_target_ranking_when_configured(
    mock_pick: MagicMock,
    mock_get: MagicMock,
    _mock_filter: MagicMock,
    test_settings,
) -> None:
    """SENTIMENT_TARGET should reorder candidates before diversity."""
    settings = replace(test_settings, sentiment_target=0.65)
    mock_get.return_value = [
        _article(1, "Cheer", score=0.99),
        _article(2, "Glass", score=0.64),
    ]
    mock_pick.return_value = [mock_get.return_value[1]]

    select_articles_for_digest(settings)

    ranked_arg = mock_pick.call_args.args[0]
    assert [a.id for a in ranked_arg] == [2, 1]

@patch("app.digest.articles_db.get_digest_candidates", side_effect=RuntimeError("db"))
def test_select_articles_for_digest_wraps_load_errors(
    _mock_get: MagicMock,
    test_settings,
) -> None:
    """DB load failures should surface as DigestSelectionError."""
    with pytest.raises(DigestSelectionError):
        select_articles_for_digest(test_settings)


@patch("app.digest.articles_db.assign_digest_date", return_value=1)
@patch("app.digest.digests_db.upsert_digest")
@patch("app.digest.select_articles_for_digest")
@patch("app.digest.digests_db.get_digest_by_date", return_value=None)
@patch("app.digest.digest_calendar_date", return_value=date(2026, 7, 28))
def test_compile_digest_persists_selection(
    _mock_date: MagicMock,
    _mock_existing: MagicMock,
    mock_select: MagicMock,
    mock_upsert: MagicMock,
    mock_assign: MagicMock,
    test_settings,
) -> None:
    """compile_digest should upsert digests and stamp article digest_date."""
    selected = [_article(10, "Story", score=0.9, summary="A real summary.")]
    mock_select.return_value = selected
    mock_upsert.return_value = DigestRecord(
        id=3,
        date=date(2026, 7, 28),
        article_ids=[10],
        email_sent_at=None,
    )

    compiled = compile_digest(test_settings)

    assert compiled.digest_id == 3
    assert compiled.digest_date == date(2026, 7, 28)
    assert [a.id for a in compiled.articles] == [10]
    mock_select.assert_called_once_with(test_settings)
    mock_upsert.assert_called_once_with(date(2026, 7, 28), [10])
    mock_assign.assert_called_once_with(date(2026, 7, 28), [10])

@patch("app.digest.digests_db.get_digest_by_date")
@patch("app.digest.digest_calendar_date", return_value=date(2026, 7, 28))
def test_compile_digest_refuses_already_sent(
    _mock_date: MagicMock,
    mock_existing: MagicMock,
    test_settings,
) -> None:
    """Already-emailed digests must not be overwritten."""
    mock_existing.return_value = DigestRecord(
        id=1,
        date=date(2026, 7, 28),
        article_ids=[1],
        email_sent_at=datetime(2026, 7, 28, 12, 0),
    )

    with pytest.raises(DigestCompileError, match="already emailed"):
        compile_digest(test_settings)


def test_reset_digest_refuses_when_disabled(test_settings) -> None:
    """reset_digest must no-op refuse when ALLOW_DIGEST_RESET is false."""
    with pytest.raises(DigestResetError, match="disabled"):
        reset_digest(test_settings)


@patch("app.digest.articles_db.clear_digest_dates", return_value=2)
@patch("app.digest.digests_db.clear_email_sent")
@patch("app.digest.digests_db.get_digest_by_date")
@patch("app.digest.digest_calendar_date", return_value=date(2026, 7, 28))
def test_reset_digest_clears_send_lock(
    _mock_date: MagicMock,
    mock_existing: MagicMock,
    mock_clear_sent: MagicMock,
    mock_clear_stamps: MagicMock,
    test_settings,
) -> None:
    """When enabled, reset clears email_sent_at and article stamps."""
    mock_existing.return_value = DigestRecord(
        id=9,
        date=date(2026, 7, 28),
        article_ids=[1, 2],
        email_sent_at=datetime(2026, 7, 28, 12, 0),
    )
    mock_clear_sent.return_value = DigestRecord(
        id=9,
        date=date(2026, 7, 28),
        article_ids=[1, 2],
        email_sent_at=None,
    )
    enabled = replace(test_settings, allow_digest_reset=True)

    result = reset_digest(enabled)

    assert result.digest_id == 9
    assert result.digest_date == date(2026, 7, 28)
    assert result.email_was_sent is True
    assert result.articles_unstamped == 2
    mock_clear_sent.assert_called_once_with(date(2026, 7, 28))
    mock_clear_stamps.assert_called_once_with(date(2026, 7, 28))


@patch("app.digest.digests_db.get_digest_by_date", return_value=None)
@patch("app.digest.digest_calendar_date", return_value=date(2026, 7, 28))
def test_reset_digest_missing_row(
    _mock_date: MagicMock,
    _mock_existing: MagicMock,
    test_settings,
) -> None:
    """Reset with no digest for today should raise DigestResetError."""
    enabled = replace(test_settings, allow_digest_reset=True)

    with pytest.raises(DigestResetError, match="No digest found"):
        reset_digest(enabled)
