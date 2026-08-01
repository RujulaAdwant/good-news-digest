"""Good-news relevance filter via embedding prototypes.

SST-2 sentiment only answers "is the tone positive?" — corporate earnings
and travel/hotel ads score high too. This stage asks: is the article
*closer in meaning* to general constructive news than to junk classes
(investor/IR prose or advertising / destination promo)?

Horoscope / astrology copy is excluded up front with a keyword gate
(high precision) and also appears in the promotional prototype set as a
embedding backstop for softer phrasings.

Model: reuse ``all-MiniLM-L6-v2`` from the deduplicator (same 384-d space).

Math intuition:
  Encode the article and three small sets of prototype sentences.
  Take max cosine similarity to each set (best-matching prototype).
  Keep the article if:
      sim(good_news) - max(sim(corporate), sim(promotional)) >= relevance_margin
  Margin 0.05 is a starting point — raise it to be stricter.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from sentence_transformers import util

from app.deduplicator import get_embedding_model
from app.exceptions import RelevanceFilterError
from app.fetcher import text_for_nlp
from db.articles import ArticleRecord

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

# Exact-ish token matches — cheaper and more reliable than embeddings alone
# for this narrow junk class (daily horoscopes score oddly "positive").
_ASTROLOGY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhoroscope\b", re.IGNORECASE),
    re.compile(r"\bastrolog", re.IGNORECASE),  # astrology, astrological, …
    re.compile(r"\bzodiac\b", re.IGNORECASE),
    re.compile(r"\bstar signs?\b", re.IGNORECASE),
    re.compile(r"\bbirth chart\b", re.IGNORECASE),
    re.compile(r"\btarot\b", re.IGNORECASE),
)

# Short, concrete exemplars — not keywords. Embeddings match *meaning*.
GOOD_NEWS_PROTOTYPES: tuple[str, ...] = (
    "A scientific breakthrough helps people and the planet",
    "A community charity project improves local lives",
    "Wildlife recovery and conservation success story",
    "Students or workers gain new opportunity through a public program",
    "Humanitarian aid and peacebuilding make measurable progress",
    "Clean energy or climate action delivers a concrete win",
    "Medical research leads to a treatment that helps patients",
)

CORPORATE_PROTOTYPES: tuple[str, ...] = (
    "Company reports strong quarterly earnings and solid H1 results",
    "Investor update on portfolio performance and shareholder returns",
    "Press release announces revenue growth and EBITDA improvement",
    "Business unit demonstrates relevance of its product portfolio",
    "Stock market and financial results beat analyst expectations",
    "Corporate merger acquisition creates shareholder value",
)

# Travel blurbs, hotel features, product plugs, SEO listicles — positive tone, not news.
PROMOTIONAL_PROTOTYPES: tuple[str, ...] = (
    "Hotel review praises impressively designed rooms for a weekend getaway",
    "Travel destination guide highlights retro charm and modern appeal for visitors",
    "Sponsored content advertises a resort stay and memorable vacation experience",
    "Product advertisement recommends buying a brand for its stylish design",
    "Lifestyle blog promotes a restaurant or boutique as must-visit for tourists",
    "Real estate listing markets a luxury property with stunning amenities",
    "Top 10 oldest boarding schools in India ranking list for parents to choose",
    "Best schools ranking advertisement promoting private boarding school admissions",
    "SEO listicle of best classical music pieces of all time clickbait",
    "Sponsored education guide listing top colleges and courses to enroll",
    "Daily horoscope predicts love money and luck for your zodiac sign",
    "Astrology forecast for Aries Taurus Gemini this week",
    "What the stars say about your relationships according to your birth chart",
)


def is_astrology_content(title: str, full_text: str | None) -> bool:
    """Return whether title/body looks like horoscope or astrology content.

    Args:
        title: Article headline.
        full_text: Optional body snippet.

    Returns:
        True if a high-confidence astrology/horoscope token match is found.
    """
    haystack = f"{title}\n{full_text or ''}"
    return any(pattern.search(haystack) for pattern in _ASTROLOGY_PATTERNS)

_prototype_embeddings: tuple[Tensor, Tensor, Tensor] | None = None


def _encode(texts: list[str]) -> Tensor:
    """Encode texts into L2-normalized embedding vectors.

    Args:
        texts: Strings to embed.

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


def _get_prototype_embeddings() -> tuple[Tensor, Tensor, Tensor]:
    """Return cached (good_news, corporate, promotional) embedding matrices."""
    global _prototype_embeddings
    if _prototype_embeddings is None:
        logger.info(
            "Encoding good-news, corporate, and promotional relevance prototypes"
        )
        good = _encode(list(GOOD_NEWS_PROTOTYPES))
        corporate = _encode(list(CORPORATE_PROTOTYPES))
        promotional = _encode(list(PROMOTIONAL_PROTOTYPES))
        _prototype_embeddings = (good, corporate, promotional)
    return _prototype_embeddings


def relevance_margin_score(
    article_embedding: Tensor,
    *,
    good: Tensor,
    corporate: Tensor,
    promotional: Tensor,
) -> float:
    """Compute sim(good) - max(sim(corporate), sim(promotional)).

    Args:
        article_embedding: Shape (384,) or (1, 384).
        good: Prototype matrix for constructive general news.
        corporate: Prototype matrix for earnings / IR copy.
        promotional: Prototype matrix for ads / travel / product plugs.

    Returns:
        Margin score; higher means more good-news-like than junk classes.
    """
    # INTERVIEW: understand this — prototype classification via max cosine
    # Junk score is the stronger of corporate vs promo so either class can veto.
    good_sim = float(util.cos_sim(article_embedding, good).max())
    corporate_sim = float(util.cos_sim(article_embedding, corporate).max())
    promotional_sim = float(util.cos_sim(article_embedding, promotional).max())
    junk_sim = max(corporate_sim, promotional_sim)
    return good_sim - junk_sim


def filter_good_news_articles(
    articles: list[ArticleRecord],
    *,
    margin: float,
) -> list[ArticleRecord]:
    """Keep articles closer to good-news than corporate or promotional copy.

    Drops horoscope/astrology matches first (keyword gate), then scores the
    remainder with prototype embeddings. Preserves input order.

    Args:
        articles: Positive non-duplicate candidates.
        margin: Minimum ``sim(good) - max(sim(corporate), sim(promo))``
            to keep (starting point: 0.05 — tunable).

    Returns:
        Filtered article list.

    Raises:
        RelevanceFilterError: If embedding or scoring fails.
    """
    if not articles:
        return []

    # Keyword gate before encode — no reason to embed obvious horoscopes.
    eligible: list[ArticleRecord] = []
    for article in articles:
        if is_astrology_content(article.title, article.full_text):
            logger.info(
                "Skipping astrology/horoscope article url=%s",
                article.url,
            )
            continue
        eligible.append(article)

    if not eligible:
        logger.info(
            "Relevance filter kept 0/%d articles (all astrology or empty)",
            len(articles),
        )
        return []

    try:
        texts = [text_for_nlp(a.title, a.full_text) for a in eligible]
        embeddings = _encode(texts)
        good, corporate, promotional = _get_prototype_embeddings()
    except Exception as exc:
        raise RelevanceFilterError("Failed to encode relevance embeddings") from exc

    kept: list[ArticleRecord] = []
    for index, article in enumerate(eligible):
        try:
            score = relevance_margin_score(
                embeddings[index],
                good=good,
                corporate=corporate,
                promotional=promotional,
            )
        except Exception as exc:
            raise RelevanceFilterError(
                f"Relevance scoring failed for url={article.url}"
            ) from exc

        if score < margin:
            logger.info(
                "Skipping non-news (corporate/promo) article url=%s "
                "relevance_margin=%.3f threshold=%.3f",
                article.url,
                score,
                margin,
            )
            continue

        kept.append(article)

    logger.info(
        "Relevance filter kept %d/%d articles (margin>=%.3f)",
        len(kept),
        len(articles),
        margin,
    )
    return kept
