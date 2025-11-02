"""
Database connection management for Bucket Brigade.

Provides connection pooling, session management, and database initialization
for PostgreSQL with SQLAlchemy.
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base

# Database configuration from environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/bucket_brigade")

# Connection pool configuration for concurrent access
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))


def get_engine():
    """
    Create and configure SQLAlchemy engine with connection pooling.

    Returns:
        SQLAlchemy Engine instance configured for PostgreSQL
    """
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT,
        pool_recycle=POOL_RECYCLE,
        pool_pre_ping=True,  # Verify connections before using them
        echo=os.getenv("DB_ECHO", "false").lower() == "true",  # SQL logging
    )
    return engine


# Global engine instance
_engine = None
_SessionLocal = None


def init_db(drop_existing: bool = False) -> None:
    """
    Initialize database schema.

    Args:
        drop_existing: If True, drops all tables before creating them (DANGEROUS!)
    """
    global _engine, _SessionLocal

    _engine = get_engine()

    if drop_existing:
        Base.metadata.drop_all(bind=_engine)

    Base.metadata.create_all(bind=_engine)

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def get_session() -> Generator[Session, None, None]:
    """
    Get a database session for dependency injection.

    Yields:
        SQLAlchemy Session instance

    Usage:
        with get_session() as session:
            # Use session here
            pass
    """
    global _SessionLocal

    if _SessionLocal is None:
        init_db()

    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
