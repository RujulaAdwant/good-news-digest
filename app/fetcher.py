"""News ingestion from TheNewsAPI and RSS feeds."""

import logging
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from time import struct_time
from typing import Any
from zoneinfo import ZoneInfo

import feedparser
import httpx

from app.config import Settings, get_settings
from db.articles import ArticleRecord

logger = logging.getLogger(__name__)

THENNEWSAPI_URL = "https://api.thenewsapi.com/v1/news/all"

RSS_FEEDS: dict[str, str] = {
    "AP News": "https://feeds.apnews.com/rss/topnews",
    "Good News Network": "https://www.goodnewsnetwork.org/feed/",
    "Positive News": "https://www.positive.news/feed/",
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
}

# TheNewsAPI categories — mirrors the prior Event Registry topic coverage.
THENNEWSAPI_CATEGORIES: tuple[str, ...] = (
    "politics",
    "science",
    "health",
    "tech",
    "business",
    "entertainment",
    "sports",
)


def first_paragraph(text: str | None) -> str:
    """Return the first paragraph from article body text.

    Args:
        text: Raw article body or snippet.

    Returns:
        The first paragraph, or an empty string if no text is available.
    """
    if not text:
        return ""
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    return paragraphs[0] if paragraphs else text.strip()


def text_for_nlp(title: str, full_text: str | None) -> str:
    """Build the title + first paragraph string used by later NLP stages.

    Args:
        title: Article headline.
        full_text: Stored article body/snippet.

    Returns:
        Combined text for embedding and sentiment analysis.
    """
    paragraph = first_paragraph(full_text)
    if paragraph:
        return f"{title}\n\n{paragraph}"
    return title


def _to_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 datetime string into UTC."""
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return _to_utc(datetime.fromisoformat(normalized))
    except ValueError:
        logger.warning("Could not parse ISO datetime: %s", value)
        return None


def _parse_struct_time(value: struct_time | None) -> datetime | None:
    """Parse a feedparser struct_time into UTC."""
    if not value:
        return None
    return _to_utc(datetime(*value[:6], tzinfo=UTC))


def _parse_rss_datetime(entry: feedparser.FeedParserDict) -> datetime | None:
    """Extract the best available published datetime from an RSS entry."""
    for key in ("published_parsed", "updated_parsed"):
        parsed = _parse_struct_time(entry.get(key))
        if parsed:
            return parsed

    for key in ("published", "updated"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            return _to_utc(parsedate_to_datetime(raw))
        except (TypeError, ValueError):
            logger.warning("Could not parse RSS datetime %s=%r", key, raw)
    return None


def _extract_rss_body(entry: feedparser.FeedParserDict) -> str | None:
    """Return the best available body/snippet from an RSS entry."""
    content = entry.get("content")
    if content and isinstance(content, list) and content[0].get("value"):
        return content[0]["value"]

    for key in ("summary", "description"):
        value = entry.get(key)
        if value:
            return value
    return None


def _extract_thenewsapi_body(article: dict[str, Any]) -> str | None:
    """Return the best available snippet from a TheNewsAPI article.

    Prefer ``description`` (meta description) over ``snippet`` (often truncated
    to ~60 characters).
    """
    for key in ("description", "snippet"):
        value = article.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_thenewsapi_source(article: dict[str, Any], *, category: str) -> str:
    """Return the publisher domain, or a category fallback."""
    source = article.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    return f"TheNewsAPI:{category}"


def fetch_cutoff(settings: Settings | None = None) -> datetime:
    """Return the UTC cutoff for the fetch window.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Articles published before this UTC timestamp are excluded.
    """
    cfg = settings or get_settings()
    tz = ZoneInfo(cfg.digest_timezone)
    local_now = datetime.now(tz)
    local_cutoff = local_now - timedelta(hours=cfg.fetch_window_hours)
    return local_cutoff.astimezone(UTC)


def within_fetch_window(
    published_at: datetime | None,
    *,
    settings: Settings | None = None,
) -> bool:
    """Return whether an article falls inside the configured fetch window.

    Args:
        published_at: Article publication time in UTC.
        settings: Optional settings override for testing.

    Returns:
        True if the article should be kept, False otherwise.
    """
    if published_at is None:
        return False
    return _to_utc(published_at) >= fetch_cutoff(settings)


def _get_thenewsapi(params: dict[str, Any]) -> dict[str, Any]:
    """Send a GET request to TheNewsAPI all-news endpoint.

    Args:
        params: Query string parameters including ``api_token``.

    Returns:
        Parsed JSON response body.

    Raises:
        httpx.HTTPError: If the HTTP request fails.
    """
    response = httpx.get(THENNEWSAPI_URL, params=params, timeout=30.0)
    response.raise_for_status()
    return response.json()


def _parse_thenewsapi_article(
    item: dict[str, Any],
    *,
    category: str,
) -> ArticleRecord | None:
    """Normalize a single TheNewsAPI article into an ArticleRecord."""
    url = item.get("url")
    title = item.get("title")
    if not isinstance(url, str) or not isinstance(title, str):
        return None
    if not url.strip() or not title.strip():
        return None

    return ArticleRecord(
        title=title.strip(),
        url=url.strip(),
        source=_extract_thenewsapi_source(item, category=category),
        published_at=_parse_iso_datetime(item.get("published_at")),
        full_text=_extract_thenewsapi_body(item),
    )


def fetch_from_thenewsapi(settings: Settings | None = None) -> list[ArticleRecord]:
    """Fetch recent headlines across TheNewsAPI categories.

    Uses https://www.thenewsapi.com (``THENEWSAPI_KEY`` / ``api_token``).
    One request per category so a low plan ``limit`` still covers topics.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Parsed article records from TheNewsAPI responses.
    """
    cfg = settings or get_settings()
    if not cfg.thenewsapi_key:
        logger.warning("THENEWSAPI_KEY is not set; skipping TheNewsAPI fetch")
        return []

    published_after = fetch_cutoff(cfg).strftime("%Y-%m-%dT%H:%M:%S")
    articles: list[ArticleRecord] = []
    seen_urls: set[str] = set()

    for category in THENNEWSAPI_CATEGORIES:
        params = {
            "api_token": cfg.thenewsapi_key,
            "categories": category,
            "language": "en",
            "published_after": published_after,
            "sort": "published_at",
            "limit": 50,
            "page": 1,
        }
        try:
            response = _get_thenewsapi(params)
        except httpx.HTTPError:
            logger.exception("TheNewsAPI fetch failed for category=%s", category)
            continue

        if not isinstance(response, dict):
            logger.error("TheNewsAPI unexpected response type for category=%s", category)
            continue

        results = response.get("data")
        if not isinstance(results, list):
            logger.error(
                "TheNewsAPI missing data list for category=%s response_keys=%s",
                category,
                list(response.keys()),
            )
            continue

        for item in results:
            if not isinstance(item, dict):
                continue
            record = _parse_thenewsapi_article(item, category=category)
            if not record or record.url in seen_urls:
                continue
            articles.append(record)
            seen_urls.add(record.url)

    logger.info("Fetched %d articles from TheNewsAPI", len(articles))
    return articles


def fetch_from_rss() -> list[ArticleRecord]:
    """Fetch articles from configured RSS feeds.

    Returns:
        Parsed article records from all RSS feeds.
    """
    articles: list[ArticleRecord] = []
    seen_urls: set[str] = set()

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            logger.exception("RSS fetch failed for source=%s url=%s", source_name, feed_url)
            continue

        if parsed.bozo:
            logger.warning(
                "RSS feed parse warning for source=%s url=%s error=%s",
                source_name,
                feed_url,
                parsed.bozo_exception,
            )

        for entry in parsed.entries:
            url = entry.get("link")
            title = entry.get("title")
            if not url or not title or url in seen_urls:
                continue

            articles.append(
                ArticleRecord(
                    title=title.strip(),
                    url=url.strip(),
                    source=source_name,
                    published_at=_parse_rss_datetime(entry),
                    full_text=_extract_rss_body(entry),
                )
            )
            seen_urls.add(url)

    logger.info("Fetched %d articles from RSS feeds", len(articles))
    return articles


def fetch_articles(settings: Settings | None = None) -> list[ArticleRecord]:
    """Fetch articles from all sources and apply the time window filter.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Article records published within the configured fetch window.
    """
    cfg = settings or get_settings()
    combined = fetch_from_thenewsapi(cfg) + fetch_from_rss()
    filtered = [
        article
        for article in combined
        if within_fetch_window(article.published_at, settings=cfg)
    ]
    dropped = len(combined) - len(filtered)
    if dropped:
        logger.info("Dropped %d articles outside the fetch window", dropped)
    return filtered
