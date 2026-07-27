"""Embedding-based near-duplicate detection for articles.

Model: ``all-MiniLM-L6-v2`` (sentence-transformers).

Why this model: it maps a sentence into a 384-dimensional vector that
captures *meaning*, not just shared words. Same story from AP vs Reuters
with different headlines still lands near each other in vector space.

Math intuition (cosine similarity):
  After L2-normalization, cosine(a, b) = a · b  (dot product).
  1.0 = identical direction (same meaning), 0.0 = orthogonal (unrelated).
  Threshold 0.85 is a *starting point* — tune on real duplicate pairs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer, util

from app.config import Settings, get_settings
from app.exceptions import DuplicateDetectionError
from app.fetcher import fetch_cutoff, text_for_nlp
from db import articles as articles_db
from db.articles import ArticleRecord

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Lazy singleton — first call downloads/caches weights; later calls reuse.
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared MiniLM embedding model, loading it once.

    Returns:
        Cached ``SentenceTransformer`` instance.
    """
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model all-MiniLM-L6-v2")
        # INTERVIEW: understand this — load once, not per request
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


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


def find_duplicate_ids(
    articles: list[ArticleRecord],
    *,
    threshold: float,
) -> list[int]:
    """Identify newer articles that are semantic duplicates of older ones.

    Walks articles in ascending ``id`` order (caller must sort). The first
    time we see a story we keep it; later near-matches are flagged.

    Args:
        articles: Windowed articles, oldest id first.
        threshold: Cosine similarity cutoff (env default 0.85 — tunable).

    Returns:
        Ids that should be marked ``is_duplicate=True``.

    Raises:
        DuplicateDetectionError: If encoding or comparison fails.
    """
    if len(articles) < 2:
        return []

    try:
        texts = [text_for_nlp(a.title, a.full_text) for a in articles]
        embeddings = _encode(texts)
    except Exception as exc:
        raise DuplicateDetectionError("Failed to encode articles") from exc

    duplicate_ids: list[int] = []
    # Indices into ``articles`` / ``embeddings`` that we treat as canonical.
    kept_indices: list[int] = []

    for index, article in enumerate(articles):
        if article.id is None:
            continue

        if not kept_indices:
            kept_indices.append(index)
            continue

        try:
            # Compare this row only against already-kept (non-duplicate) rows.
            kept_embeddings = embeddings[kept_indices]
            similarities = util.cos_sim(embeddings[index], kept_embeddings)[0]
            max_similarity = float(similarities.max())
        except Exception as exc:
            raise DuplicateDetectionError(
                f"Similarity comparison failed for url={article.url}"
            ) from exc

        if max_similarity >= threshold:
            logger.info(
                "Duplicate detected url=%s max_similarity=%.3f threshold=%.3f",
                article.url,
                max_similarity,
                threshold,
            )
            duplicate_ids.append(article.id)
        else:
            kept_indices.append(index)

    return duplicate_ids


def flag_duplicates(settings: Settings | None = None) -> int:
    """Run windowed deduplication and persist ``is_duplicate`` flags.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Number of articles newly marked as duplicates.

    Raises:
        DuplicateDetectionError: If the stage fails.
    """
    cfg = settings or get_settings()
    cutoff = fetch_cutoff(cfg)

    try:
        window_articles = articles_db.get_articles_in_window(cutoff)
    except Exception as exc:
        raise DuplicateDetectionError("Failed to load articles for dedup") from exc

    # Already-flagged rows stay flagged; we only decide among the current set.
    # Re-running is safe: previously flagged ids may be flagged again.
    try:
        duplicate_ids = find_duplicate_ids(
            window_articles,
            threshold=cfg.similarity_threshold,
        )
        return articles_db.mark_duplicates(duplicate_ids)
    except DuplicateDetectionError:
        raise
    except Exception as exc:
        raise DuplicateDetectionError("Failed to flag duplicates") from exc
