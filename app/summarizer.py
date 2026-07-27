"""Claude-based article summarization.

Only articles that are non-duplicates and above the sentiment threshold are
summarized — summarizing discarded stories wastes API cost.

Model: ``claude-sonnet-4-6``, max_tokens=150.
Summaries are cached in ``articles.summary`` and never recomputed.
"""

from __future__ import annotations

import logging
import time

import anthropic

from app.config import Settings, get_settings
from app.exceptions import SummarizationError
from app.fetcher import fetch_cutoff, text_for_nlp
from db import articles as articles_db
from db.articles import ArticleRecord

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 150
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0

SUMMARY_PROMPT = (
    "Summarize this news article in 2-3 sentences, focusing on what's "
    "hopeful or constructive. Do not add facts that are not in the text."
)


def _build_user_message(article: ArticleRecord) -> str:
    """Build the user message Claude will summarize.

    Args:
        article: Article with title and body snippet.

    Returns:
        Prompt body including title and text.
    """
    body = text_for_nlp(article.title, article.full_text)
    return f"Title: {article.title}\n\nArticle:\n{body}"


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
        SummarizationError: If all attempts fail.
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


def summarize_articles(settings: Settings | None = None) -> int:
    """Summarize eligible articles and cache results in PostgreSQL.

    Eligibility: in fetch window, not duplicate, sentiment >= threshold,
    summary IS NULL.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Number of summaries written.

    Raises:
        SummarizationError: If candidate load or bulk persist fails.
    """
    cfg = settings or get_settings()
    if not cfg.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY is not set; skipping summarization")
        return 0

    cutoff = fetch_cutoff(cfg)

    try:
        candidates = articles_db.get_articles_needing_summary(
            cutoff,
            cfg.sentiment_threshold,
        )
    except Exception as exc:
        raise SummarizationError("Failed to load articles for summarization") from exc

    if not candidates:
        logger.info("No articles need summarization")
        return 0

    written: list[tuple[int, str]] = []
    for article in candidates:
        if article.id is None:
            continue
        try:
            summary = summarize_with_claude(article, api_key=cfg.anthropic_api_key)
            written.append((article.id, summary))
            logger.info("Summarized url=%s", article.url)
        except SummarizationError:
            logger.exception(
                "Summarization failed for url=%s stage=summarize",
                article.url,
            )

    try:
        return articles_db.update_summaries(written)
    except Exception as exc:
        raise SummarizationError("Failed to persist summaries") from exc
