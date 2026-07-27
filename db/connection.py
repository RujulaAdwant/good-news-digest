"""PostgreSQL connection helpers."""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import psycopg2
from psycopg2.extensions import connection

from app.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Generator[connection, None, None]:
    """Yield a PostgreSQL connection with automatic commit/rollback.

    Yields:
        An open psycopg2 connection. Commits on success, rolls back on error.

  Raises:
        psycopg2.Error: If the connection or transaction fails.
    """
    conn = psycopg2.connect(get_settings().database_url)
    try:
        yield conn
        conn.commit()
    except psycopg2.Error:
        conn.rollback()
        logger.exception("Database transaction failed; rolled back")
        raise
    finally:
        conn.close()
