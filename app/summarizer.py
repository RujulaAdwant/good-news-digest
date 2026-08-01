"""Claude-based article summarization.

Only the final digest picks are summarized — summarizing the full positive
pool wastes API cost and latency. Selection lives in ``app.digest``; this
module only calls Claude for those picks that still have ``summary IS NULL``.

Model: ``claude-sonnet-4-6``, max_tokens=150.
Summaries are cached in ``articles.summary`` and never recomputed.

Claude sometimes returns a meta-refusal ("I'm sorry, but…") on thin/caption
content. Those are not summaries — we reject them and leave ``summary`` null
so digests never show the refusal text.
"""

from __future__ import annotations

import logging
import re
import time

import anthropic

from app.config import Settings, get_settings
from app.digest import select_articles_for_digest
from app.exceptions import DigestSelectionError, SummarizationError
from app.fetcher import text_for_nlp
from db import articles as articles_db
from db.articles import ArticleRecord

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 150
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0
# Skip the API call when title+snippet is too thin to produce a real summary.
MIN_SUMMARY_CHARS = 80
INSUFFICIENT_CONTENT_TOKEN = "INSUFFICIENT_CONTENT"

SUMMARY_PROMPT = (
    "Summarize this news article in 2-3 sentences, focusing on what's "
    "hopeful or constructive. Do not add facts that are not in the text. "
    "If the text is too thin, is only a caption/headline with no article body, "
    f"or otherwise cannot support a real news summary, reply with exactly "
    f"{INSUFFICIENT_CONTENT_TOKEN} and nothing else. "
    "Never apologize, never explain why you cannot summarize, and never "
    "describe what the text is about when refusing."
)

# Phrases Claude (and similar models) use when declining to summarize.
_REFUSAL_PATTERNS = (
    re.compile(r"\binsufficient_content\b", re.IGNORECASE),
    re.compile(r"\bi'?m sorry\b", re.IGNORECASE),
    re.compile(r"\bi am sorry\b", re.IGNORECASE),
    re.compile(r"\bunable to summarize\b", re.IGNORECASE),
    re.compile(r"\bnot enough (substantive |meaningful )?content\b", re.IGNORECASE),
    re.compile(r"\bdoesn'?t contain enough\b", re.IGNORECASE),
    re.compile(r"\bdoes not contain enough\b", re.IGNORECASE),
    re.compile(
        r"\bcannot (construct|provide|create) (a |an )?(summary|2-3)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bi can'?t summarize\b", re.IGNORECASE),
    re.compile(r"\bno actual article content\b", re.IGNORECASE),
)


def is_usable_summary(summary: str | None) -> bool:
    """Return whether ``summary`` is a real digest blurb (not empty/refusal).

    Args:
        summary: Cached Claude output, if any.

    Returns:
        True only for non-empty text that does not look like a model refusal.
    """
    if summary is None:
        return False
    text = summary.strip()
    if not text:
        return False
    if text.upper() == INSUFFICIENT_CONTENT_TOKEN:
        return False
    return not any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


def _build_user_message(article: ArticleRecord) -> str:
    """Build the user message Claude will summarize.

    Args:
        article: Article with title and body snippet.

    Returns:
        Prompt body including title and text.
    """
    body = text_for_nlp(article.title, article.full_text)
    return f"Title: {article.title}\n\nArticle:\n{body}"


def _article_text_for_length_check(article: ArticleRecord) -> str:
    """Return the same text window used for summarization length gating."""
    return text_for_nlp(article.title, article.full_text).strip()


def summarize_with_claude(
    article: ArticleRecord,
    *,
    api_key: str,
) -> str:
    """Call Claude to summarize one article, with up to 2 retries.

    Args:
        article: Article to summarize.
        api_key: Anthropic API key.

    Returns:
        Summary text from Claude.

    Raises:
        SummarizationError: If all attempts fail, or Claude refuses / returns
            insufficient-content (caller must not persist those).
    """
    client = anthropic.Anthropic(api_key=api_key)
    user_content = _build_user_message(article)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            # INTERVIEW: understand this — max_tokens caps output length AND cost
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": f"{SUMMARY_PROMPT}\n\n{user_content}",
                    }
                ],
            )
            text_blocks = [
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            summary = "\n".join(text_blocks).strip()
            if not summary:
                raise SummarizationError("Claude returned an empty summary")
            if not is_usable_summary(summary):
                # Do not retry refusals — thin content will refuse again.
                raise SummarizationError(
                    f"Claude returned a non-summary refusal for url={article.url}"
                )
            return summary
        except SummarizationError:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Claude summarize attempt %d/%d failed url=%s error=%s",
                attempt + 1,
                MAX_RETRIES + 1,
                article.url,
                type(exc).__name__,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise SummarizationError(
        f"Claude summarization failed for url={article.url}"
    ) from last_error


# Over-select relative to DIGEST_SIZE so Claude refusals / thin articles
# still leave summaries available for as many final digest picks as possible.
SUMMARY_CANDIDATE_MULTIPLIER = 2


def summarize_articles(settings: Settings | None = None) -> int:
    """Summarize final digest picks and cache results in PostgreSQL.

    Eligibility: selected by ``select_articles_for_digest`` (top sentiment,
    topic-diverse) and still missing a *usable* summary. We over-select
    (``digest_size * 2``) so more of the eventual digest picks get blurbs
    even when some Claude calls refuse. Cached refusal text is cleared so
    digests never surface it.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Number of summaries written.

    Raises:
        SummarizationError: If selection or bulk persist fails.
    """
    cfg = settings or get_settings()
    if not cfg.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY is not set; skipping summarization")
        return 0

    try:
        selected = select_articles_for_digest(
            cfg,
            digest_size=cfg.digest_size * SUMMARY_CANDIDATE_MULTIPLIER,
        )
    except DigestSelectionError as exc:
        raise SummarizationError("Digest selection failed") from exc

    # Drop previously cached refusal prose so it cannot leak into /digest.
    stale_refusal_ids = [
        article.id
        for article in selected
        if article.id is not None
        and article.summary
        and not is_usable_summary(article.summary)
    ]
    if stale_refusal_ids:
        try:
            cleared = articles_db.clear_summaries(stale_refusal_ids)
            logger.info(
                "Cleared %d unusable cached summaries from digest picks",
                cleared,
            )
        except Exception as exc:
            raise SummarizationError(
                "Failed to clear unusable cached summaries"
            ) from exc

    needing_summary: list[ArticleRecord] = []
    for article in selected:
        if article.id is None:
            continue
        if is_usable_summary(article.summary):
            continue
        body = _article_text_for_length_check(article)
        if len(body) < MIN_SUMMARY_CHARS:
            logger.info(
                "Skipping thin article for Claude url=%s chars=%d min=%d",
                article.url,
                len(body),
                MIN_SUMMARY_CHARS,
            )
            continue
        needing_summary.append(article)

    if not needing_summary:
        logger.info(
            "No digest picks need summarization (selected=%d)",
            len(selected),
        )
        return 0

    logger.info(
        "Summarizing %d digest picks (selected=%d summarize_cap=%d digest_size=%d)",
        len(needing_summary),
        len(selected),
        cfg.digest_size * SUMMARY_CANDIDATE_MULTIPLIER,
        cfg.digest_size,
    )

    written: list[tuple[int, str]] = []
    for article in needing_summary:
        assert article.id is not None  # filtered above
        try:
            summary = summarize_with_claude(article, api_key=cfg.anthropic_api_key)
            written.append((article.id, summary))
            logger.info("Summarized url=%s", article.url)
        except SummarizationError:
            logger.warning(
                "Summarization skipped (no usable summary) url=%s stage=summarize",
                article.url,
            )

    try:
        return articles_db.update_summaries(written)
    except Exception as exc:
        raise SummarizationError("Failed to persist summaries") from exc
