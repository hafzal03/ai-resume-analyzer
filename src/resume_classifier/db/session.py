"""Database engine and session management.

Persistence is optional. When ``RC_DATABASE_URL`` is unset the application still
classifies perfectly well -- it simply does not record anything. That keeps
local development and the test suite free of a PostgreSQL dependency, and it
means a database outage degrades the service rather than taking it down.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from resume_classifier.config import Settings

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(RuntimeError):
    """Persistence was requested but the database cannot be reached."""


def create_db_engine(settings: Settings) -> Engine:
    """Build a SQLAlchemy engine from configuration.

    Raises:
        DatabaseUnavailableError: if no database URL is configured.
    """
    if not settings.database_url:
        msg = "RC_DATABASE_URL is not set; persistence is disabled"
        raise DatabaseUnavailableError(msg)
    return create_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_pre_ping=True,  # survive a database restart without leaking dead connections
        pool_size=5,
        max_overflow=5,
        future=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a configured session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(engine: Engine) -> bool:
    """Return whether the database answers a trivial query.

    Used by the readiness probe. Never raises: an unreachable database is a
    reported state, not a crash.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("database health check failed: %s", exc)
        return False
    return True
