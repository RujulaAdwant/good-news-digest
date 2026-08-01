"""Apply ``schema.sql`` idempotently (safe for local + Railway first boot)."""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg2

from db.connection import get_connection

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def apply_schema() -> None:
    """Create ``articles`` and ``digests`` tables if they do not exist.

    Raises:
        FileNotFoundError: If ``schema.sql`` is missing.
        psycopg2.Error: If PostgreSQL rejects the DDL.
    """
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
    except psycopg2.Error:
        logger.exception("Failed to apply schema from %s", _SCHEMA_PATH)
        raise

    logger.info("Schema applied from %s", _SCHEMA_PATH.name)


def main() -> None:
    """CLI entrypoint: ``python -m db.migrate``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    apply_schema()
    print("Schema ready.")


if __name__ == "__main__":
    main()
