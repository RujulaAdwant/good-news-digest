"""Digest candidate selection and daily digest compilation.

Picks the final stories that will appear in the daily email: non-duplicates
above ``sentiment_threshold``, filtered for *general* good-news relevance
(vs corporate/IR copy and horoscopes), ranked for Glass tone when
``sentiment_target`` is set (prefer scores near the target instead of max
positivity), then diversified by *topic* and *source* so the digest is not
five variants of one story or five items from one outlet.

Articles are selected by sentiment/relevance/diversity — not by whether
Claude already summarized them. Usable summaries are attached when present;
missing ones are simply omitted from the email/API preview.

Topic diversity reuses ``all-MiniLM-L6-v2`` embeddings from the deduplicator.
Greedy rule: walk candidates in rank order; keep a story only if its
max cosine similarity to already-selected stories is below
``topic_diversity_threshold`` (starting point 0.70 — looser than dedup 0.85,
so near-dupes stay out but same-topic coverage is still rejected) and,
preferentially, its source is new. A second pass may repeat sources only
if the digest would otherwise stay under ``digest_size``.

Compilation persists a ``digests`` row and stamps ``articles.digest_date``.
Sending is a separate stage (``app.emailer``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from sentence_transformers import util

from app.config import Settings, get_settings
from app.deduplicator import get_embedding_model
from app.exceptions import DigestCompileError, DigestResetError, DigestSelectionError, RelevanceFilterError
from app.fetcher import fetch_cutoff, text_for_nlp
from app.relevance import filter_good_news_articles
from db import articles as articles_db
from db import digests as digests_db
from db.articles import ArticleRecord

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledDigest:
    """Result of selecting and persisting today's digest."""

    digest_id: int
    digest_date: date
    articles: list[ArticleRecord]
    already_sent: bool = False


def digest_calendar_date(settings: Settings | None = None) -> date:
    """Return today's date in the configured digest timezone.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Local calendar date used as ``digests.date``.
    """
    cfg = settings or get_settings()
    return datetime.now(ZoneInfo(cfg.digest_timezone)).date()


def _encode(texts: list[str]) -> Tensor:
    """Encode texts into L2-normalized embedding vectors.

    Args:
        texts: Strings to embed (title + first paragraph).

    Returns:
        Tensor of shape (len(texts), 384).
    """
    model = get_embedding_model()
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def rank_digest_candidates(
    candidates: list[ArticleRecord],
    *,
    sentiment_target: float | None,
) -> list[ArticleRecord]:
    """Order digest candidates for greedy topic-diversity selection.

    Args:
        candidates: Pool already past the sentiment floor (and usually
            relevance). Order from the DB is sentiment DESC.
        sentiment_target: When set, prefer scores closest to this value
            (Glass / neutrally-positive tone). Distance ties break toward
            the higher score, then lower id. When ``None``, keep max-positive
            order (legacy).

    Returns:
        New list in selection-priority order (does not mutate ``candidates``).
    """
    if not candidates:
        return []

    if sentiment_target is None:
        # DB already returns sentiment DESC; re-sort so callers can pass any
        # order after relevance filtering without depending on that.
        return sorted(
            candidates,
            key=lambda a: (
                -(a.sentiment_score if a.sentiment_score is not None else 0.0),
                a.id if a.id is not None else 0,
            ),
        )

    # INTERVIEW: understand this — target ranking vs max ranking
    # |score - target| small ⇒ closer to the Glass "sweet spot"; among equal
    # distances, prefer the slightly more positive story.
    target = sentiment_target
    return sorted(
        candidates,
        key=lambda a: (
            abs((a.sentiment_score if a.sentiment_score is not None else 0.0) - target),
            -(a.sentiment_score if a.sentiment_score is not None else 0.0),
            a.id if a.id is not None else 0,
        ),
    )


def _source_key(article: ArticleRecord) -> str:
    """Normalize source for uniqueness checks (case/whitespace-insensitive).

    Args:
        article: Candidate with optional ``source``.

    Returns:
        Stable key; empty/missing sources collapse to ``unknown`` so we still
        cap how many unlabeled rows can dominate a digest.
    """
    raw = (article.source or "").strip()
    return raw.casefold() if raw else "unknown"


def pick_diverse_articles(
    candidates: list[ArticleRecord],
    *,
    digest_size: int,
    diversity_threshold: float,
) -> list[ArticleRecord]:
    """Greedily pick up to ``digest_size`` topic- and source-diverse articles.

    Caller must pass candidates already ordered by selection preference
    (max sentiment, or closeness to ``sentiment_target``).

    Pass 1 prefers a new source on every pick (plus topic diversity). Pass 2
    fills remaining slots with topic diversity only, so a thin multi-source
    pool does not leave the digest short.

    Args:
        candidates: Eligible pool, highest priority first.
        digest_size: Maximum articles to keep (starting point: 5).
        diversity_threshold: Max allowed cosine similarity to any already
            selected article (starting point: 0.70 — tunable).

    Returns:
        Selected articles in selection order.

    Raises:
        DigestSelectionError: If embedding or similarity comparison fails.
    """
    if not candidates or digest_size <= 0:
        return []

    try:
        texts = [text_for_nlp(a.title, a.full_text) for a in candidates]
        embeddings = _encode(texts)
    except Exception as exc:
        raise DigestSelectionError("Failed to encode digest candidates") from exc

    selected: list[ArticleRecord] = []
    selected_indices: list[int] = []
    selected_sources: set[str] = set()

    def _topic_ok(index: int) -> bool:
        """Return True if candidate ``index`` is topic-distinct from selected."""
        if not selected_indices:
            return True
        try:
            # INTERVIEW: understand this — diversity via max similarity to set
            # Cosine to each kept embedding; if the closest kept story is still
            # below the threshold, this one is a different enough topic.
            kept = embeddings[selected_indices]
            similarities = util.cos_sim(embeddings[index], kept)[0]
            max_similarity = float(similarities.max())
        except Exception as exc:
            raise DigestSelectionError(
                f"Topic diversity check failed for url={candidates[index].url}"
            ) from exc
        if max_similarity >= diversity_threshold:
            logger.info(
                "Skipping overlapping topic url=%s max_similarity=%.3f "
                "threshold=%.3f",
                candidates[index].url,
                max_similarity,
                diversity_threshold,
            )
            return False
        return True

    def _try_add(index: int, *, require_new_source: bool) -> bool:
        """Attempt to append candidate ``index``; return True if kept."""
        if len(selected) >= digest_size:
            return False
        if index in selected_indices:
            return False

        article = candidates[index]
        source = _source_key(article)
        if require_new_source and source in selected_sources:
            logger.info(
                "Skipping duplicate source url=%s source=%s",
                article.url,
                article.source,
            )
            return False
        if not _topic_ok(index):
            return False

        selected.append(article)
        selected_indices.append(index)
        selected_sources.add(source)
        return True

    # Pass 1: unique sources + topic diversity.
    for index in range(len(candidates)):
        if len(selected) >= digest_size:
            break
        _try_add(index, require_new_source=True)

    # Pass 2: if still short, allow source repeats but keep topic diversity.
    if len(selected) < digest_size:
        logger.info(
            "Source-diverse pass filled %d/%d; relaxing source uniqueness",
            len(selected),
            digest_size,
        )
        for index in range(len(candidates)):
            if len(selected) >= digest_size:
                break
            _try_add(index, require_new_source=False)

    logger.info(
        "Selected %d/%d digest articles from %d candidates "
        "(unique_sources=%d)",
        len(selected),
        digest_size,
        len(candidates),
        len(selected_sources),
    )
    return selected


def select_articles_for_digest(
    settings: Settings | None = None,
    *,
    digest_size: int | None = None,
) -> list[ArticleRecord]:
    """Load the positive pool and return final digest picks.

    Args:
        settings: Optional settings override for testing.
        digest_size: Cap on picks; defaults to ``settings.digest_size``.
            Summarization may pass a larger value to over-summarize backups.

    Returns:
        Up to ``digest_size`` topic-diverse articles in selection order.

    Raises:
        DigestSelectionError: If loading or selection fails.
    """
    cfg = settings or get_settings()
    size_cap = cfg.digest_size if digest_size is None else digest_size
    cutoff = fetch_cutoff(cfg)

    try:
        candidates = articles_db.get_digest_candidates(
            cutoff,
            cfg.sentiment_threshold,
        )
    except Exception as exc:
        raise DigestSelectionError("Failed to load digest candidates") from exc

    if not candidates:
        logger.info("No digest candidates in window")
        return []

    try:
        relevant = filter_good_news_articles(
            candidates,
            margin=cfg.relevance_margin,
        )
    except RelevanceFilterError as exc:
        raise DigestSelectionError("Relevance filtering failed") from exc

    if not relevant:
        logger.info(
            "No digest candidates left after relevance filter (from %d)",
            len(candidates),
        )
        return []

    ranked = rank_digest_candidates(
        relevant,
        sentiment_target=cfg.sentiment_target,
    )
    logger.info(
        "Ranked %d candidates mode=%s target=%s",
        len(ranked),
        "target" if cfg.sentiment_target is not None else "max_positive",
        cfg.sentiment_target,
    )

    try:
        return pick_diverse_articles(
            ranked,
            digest_size=size_cap,
            diversity_threshold=cfg.topic_diversity_threshold,
        )
    except DigestSelectionError:
        raise
    except Exception as exc:
        raise DigestSelectionError("Failed to select digest articles") from exc


def compile_digest(settings: Settings | None = None) -> CompiledDigest:
    """Select today's stories, upsert the digests row, and stamp articles.

    Selection ignores summary availability. Prefer running
    ``summarize_articles`` first so as many picks as possible have blurbs;
    stories without a usable summary still appear (title/source only).

    Idempotent for an unsent day: re-running replaces ``article_ids``.
    Refuses to overwrite a day that already has ``email_sent_at`` set.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Persisted digest metadata and the selected article records.

    Raises:
        DigestCompileError: If selection or persistence fails, or the day
            was already emailed.
    """
    cfg = settings or get_settings()
    digest_date = digest_calendar_date(cfg)

    try:
        existing = digests_db.get_digest_by_date(digest_date)
    except Exception as exc:
        raise DigestCompileError("Failed to load existing digest") from exc

    if existing is not None and existing.email_sent_at is not None:
        raise DigestCompileError(
            f"Digest for {digest_date} was already emailed; refusing to recompile"
        )

    try:
        selected = select_articles_for_digest(cfg)
    except DigestSelectionError as exc:
        raise DigestCompileError("Digest selection failed") from exc

    article_ids = [article.id for article in selected if article.id is not None]
    if len(article_ids) != len(selected):
        raise DigestCompileError("Selected articles are missing database ids")

    try:
        saved = digests_db.upsert_digest(digest_date, article_ids)
    except Exception as exc:
        raise DigestCompileError("Failed to persist digest row") from exc

    if saved is None:
        raise DigestCompileError(
            f"Digest for {digest_date} was already emailed; refusing to recompile"
        )

    try:
        articles_db.assign_digest_date(digest_date, article_ids)
    except Exception as exc:
        raise DigestCompileError("Failed to stamp article digest_date") from exc

    logger.info(
        "Compiled digest id=%d date=%s article_count=%d",
        saved.id,
        saved.date,
        len(selected),
    )
    return CompiledDigest(
        digest_id=saved.id,
        digest_date=saved.date,
        articles=selected,
        already_sent=False,
    )


@dataclass(frozen=True)
class DigestResetResult:
    """Outcome of unlocking today's digest for another compile/send cycle."""

    digest_id: int
    digest_date: date
    email_was_sent: bool
    articles_unstamped: int


def reset_digest(settings: Settings | None = None) -> DigestResetResult:
    """Unlock today's digest so it can be recompiled and resent.

    Clears ``email_sent_at`` and article ``digest_date`` stamps for the
    local calendar day. Gated by ``settings.allow_digest_reset`` so this
    cannot run in production unless explicitly enabled.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Metadata about what was unlocked.

    Raises:
        DigestResetError: If reset is disabled, no digest exists, or DB fails.
    """
    cfg = settings or get_settings()
    if not cfg.allow_digest_reset:
        raise DigestResetError(
            "Digest reset is disabled; set ALLOW_DIGEST_RESET=true to enable"
        )

    digest_date = digest_calendar_date(cfg)

    try:
        existing = digests_db.get_digest_by_date(digest_date)
    except Exception as exc:
        raise DigestResetError("Failed to load digest for reset") from exc

    if existing is None:
        raise DigestResetError(f"No digest found for {digest_date}")

    email_was_sent = existing.email_sent_at is not None

    try:
        cleared = digests_db.clear_email_sent(digest_date)
    except Exception as exc:
        raise DigestResetError("Failed to clear email_sent_at") from exc

    if cleared is None:
        raise DigestResetError(f"No digest found for {digest_date}")

    try:
        unstamped = articles_db.clear_digest_dates(digest_date)
    except Exception as exc:
        raise DigestResetError("Failed to clear article digest_date stamps") from exc

    logger.info(
        "Reset digest id=%d date=%s email_was_sent=%s articles_unstamped=%d",
        cleared.id,
        cleared.date,
        email_was_sent,
        unstamped,
    )
    return DigestResetResult(
        digest_id=cleared.id,
        digest_date=cleared.date,
        email_was_sent=email_was_sent,
        articles_unstamped=unstamped,
    )
