"""HTML formatting and SendGrid delivery for the daily digest.

Split from compilation: ``compile_digest`` persists picks; this module loads
an existing ``digests`` row and emails it. That lets you preview/recompile
before spending a SendGrid send.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from html import escape

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from app.config import Settings, get_settings
from app.digest import digest_calendar_date
from app.exceptions import EmailDeliveryError
from app.summarizer import is_usable_summary
from db import articles as articles_db
from db import digests as digests_db
from db.articles import ArticleRecord
from db.digests import DigestRecord

logger = logging.getLogger(__name__)


def format_digest_subject(digest_date: date) -> str:
    """Build the email subject line for a digest date.

    Args:
        digest_date: Local calendar date of the digest.

    Returns:
        Subject string, e.g. ``The Glass Digest — July 28, 2026``.
    """
    return f"The Glass Digest — {digest_date.strftime('%B')} {digest_date.day}, {digest_date.year}"


def format_digest_html(
    articles: list[ArticleRecord],
    digest_date: date,
) -> str:
    """Render a glass-minimalist HTML body for the digest email.

    Cool frosted palette, generous whitespace, serif headlines with tracked
    sans-serif meta. Numbered stories (source kicker → linked title → optional
    summary). No images — banner support is postponed until a hosted URL is
    ready for multi-user delivery.

    Every selected article is rendered (title + source). A Claude summary
    paragraph is included only when ``is_usable_summary`` passes — refusals
    and missing summaries are omitted, not shown as placeholder text.

    Args:
        articles: Stories in display order (already selected/persisted).
        digest_date: Local calendar date shown in the header.

    Returns:
        HTML string suitable for SendGrid ``html_content``.
    """
    # Email clients strip <style> / webfonts, so everything is inline with
    # system stacks. Outer mist + inner frost panel = "glass" without images.
    date_label = escape(
        f"{digest_date.strftime('%B')} {digest_date.day}, {digest_date.year}"
    )
    serif = (
        "Georgia, 'Iowan Old Style', 'Palatino Linotype', Palatino, "
        "'Times New Roman', serif"
    )
    sans = (
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, "
        "sans-serif"
    )
    mist = "#e8eef2"
    frost = "#fbfcfd"
    ink = "#1c2430"
    muted = "#6b7585"
    accent = "#7a9bb0"
    hairline = "#d5dee6"

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "</head>",
        f'<body style="margin: 0; padding: 0; background-color: {mist};">',
        f'<div style="background-color: {mist}; padding: 32px 16px;">',
        f'<div style="max-width: 560px; margin: 0 auto; background-color: {frost}; '
        f"border: 1px solid {hairline}; border-radius: 2px; "
        f'padding: 40px 36px 36px; color: {ink}; line-height: 1.6;">',
        # Masthead — tracked brand mark, not a loud H1 block
        f'<p style="margin: 0 0 10px; font-family: {sans}; font-size: 11px; '
        f"font-weight: 500; letter-spacing: 0.22em; text-transform: uppercase; "
        f'color: {accent}; text-align: center;">The Glass Digest</p>',
        f'<p style="margin: 0 0 28px; font-family: {serif}; font-size: 15px; '
        f'font-style: italic; color: {muted}; text-align: center;">'
        f"{date_label}</p>",
        f'<div style="height: 1px; background-color: {hairline}; '
        f'margin: 0 auto 36px; max-width: 48px;"></div>',
    ]

    if not articles:
        parts.append(
            f'<p style="margin: 0; font-family: {serif}; font-size: 16px; '
            f'color: {muted}; text-align: center;">Check back tomorrow.</p>'
        )
    else:
        last_index = len(articles) - 1
        for index, article in enumerate(articles):
            title = escape(article.title)
            url = escape(article.url, quote=True)
            source = escape(article.source or "Unknown source")
            number = f"{index + 1:02d}"
            bottom_pad = "0" if index == last_index else "32px"
            block = [
                f'<article style="margin: 0 0 {bottom_pad};">',
                f'<p style="margin: 0 0 8px; font-family: {sans}; font-size: 11px; '
                f"letter-spacing: 0.16em; text-transform: uppercase; "
                f'color: {accent};">{number} · {source}</p>',
                f'<h2 style="margin: 0 0 12px; font-family: {serif}; '
                f'font-size: 20px; font-weight: normal; line-height: 1.35;">'
                f'<a href="{url}" style="color: {ink}; text-decoration: none;">'
                f"{title}</a></h2>",
            ]
            if is_usable_summary(article.summary):
                summary = escape(article.summary or "")
                block.append(
                    f'<p style="margin: 0; font-family: {sans}; font-size: 15px; '
                    f'line-height: 1.65; color: {muted};">{summary}</p>'
                )
            if index != last_index:
                block.append(
                    f'<div style="height: 1px; background-color: {hairline}; '
                    f'margin: 32px 0 0;"></div>'
                )
            block.append("</article>")
            parts.append("".join(block))

    parts.extend(
        [
            f'<div style="height: 1px; background-color: {hairline}; '
            f'margin: 40px auto 20px; max-width: 48px;"></div>',
            f'<p style="margin: 0; font-family: {sans}; font-size: 11px; '
            f"letter-spacing: 0.06em; color: {muted}; text-align: center;\">"
            "A clearer view of the news.</p>",
            "</div>",
            "</div>",
            "</body></html>",
        ]
    )
    return "\n".join(parts)


def _require_send_config(cfg: Settings) -> None:
    """Raise if SendGrid delivery settings are incomplete."""
    missing = [
        name
        for name, value in (
            ("SENDGRID_API_KEY", cfg.sendgrid_api_key),
            ("SENDGRID_FROM_EMAIL", cfg.sendgrid_from_email),
            ("DIGEST_RECIPIENT_EMAIL", cfg.digest_recipient_email),
        )
        if not value
    ]
    if missing:
        raise EmailDeliveryError(
            "Missing email config: " + ", ".join(missing)
        )


def _load_digest_or_raise(digest_date: date) -> DigestRecord:
    """Load a compiled digest or raise EmailDeliveryError."""
    try:
        digest = digests_db.get_digest_by_date(digest_date)
    except Exception as exc:
        raise EmailDeliveryError(
            f"Failed to load digest for date={digest_date}"
        ) from exc

    if digest is None:
        raise EmailDeliveryError(
            f"No compiled digest for {digest_date}; POST /digest first"
        )
    return digest


def send_digest(
    settings: Settings | None = None,
    *,
    digest_date: date | None = None,
) -> DigestRecord:
    """Send the compiled digest for a date via SendGrid.

    Args:
        settings: Optional settings override for testing.
        digest_date: Local digest date; defaults to today in digest timezone.

    Returns:
        Digest row after ``email_sent_at`` is stamped.

    Raises:
        EmailDeliveryError: If config, load, SendGrid, or stamp fails.
    """
    cfg = settings or get_settings()
    _require_send_config(cfg)
    target_date = digest_date or digest_calendar_date(cfg)
    digest = _load_digest_or_raise(target_date)

    if digest.email_sent_at is not None:
        raise EmailDeliveryError(
            f"Digest for {target_date} was already sent at {digest.email_sent_at}"
        )

    try:
        articles = articles_db.get_articles_by_ids(digest.article_ids)
    except Exception as exc:
        raise EmailDeliveryError("Failed to load digest articles") from exc

    subject = format_digest_subject(target_date)
    html_body = format_digest_html(articles, target_date)
    message = Mail(
        from_email=cfg.sendgrid_from_email,
        to_emails=cfg.digest_recipient_email,
        subject=subject,
        html_content=html_body,
    )

    try:
        client = SendGridAPIClient(cfg.sendgrid_api_key)
        response = client.send(message)
        status = getattr(response, "status_code", None)
        if status is not None and status >= 400:
            raise EmailDeliveryError(
                f"SendGrid rejected email status={status} date={target_date}"
            )
    except EmailDeliveryError:
        raise
    except Exception as exc:
        logger.exception(
            "SendGrid send failed date=%s recipient=%s",
            target_date,
            cfg.digest_recipient_email,
        )
        raise EmailDeliveryError(
            f"SendGrid send failed for date={target_date}"
        ) from exc

    sent_at = datetime.now(UTC)
    try:
        updated = digests_db.mark_email_sent(target_date, sent_at)
    except Exception as exc:
        raise EmailDeliveryError(
            "Email sent but failed to stamp email_sent_at"
        ) from exc

    if updated is None:
        raise EmailDeliveryError(
            f"Email sent but digest stamp raced for date={target_date}"
        )

    logger.info(
        "Sent digest id=%d date=%s articles=%d to=%s",
        updated.id,
        updated.date,
        len(articles),
        cfg.digest_recipient_email,
    )
    return updated
