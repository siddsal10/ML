"""SQLAlchemy engine, session factory, and declarative base.

Engine/session construction is lazy and keyed by the current DATABASE_URL
(rather than computed once at import time) so tests can point the app at an
isolated temporary SQLite file per test without needing to reload modules.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=8)
def _build_engine(database_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, future=True)


def get_engine() -> Engine:
    settings = get_settings()
    return _build_engine(settings.DATABASE_URL)


def get_session_factory() -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine(), future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a database session."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables against the currently-configured database."""
    # Import models so they are registered on Base.metadata before create_all.
    from app.models import (  # noqa: F401
        student,  # Student
        submission,  # Submission
        region,  # ExtractedRegion
        analysis,  # AnalysisResult
        mistake,  # Mistake
        quiz,  # Quiz, QuizQuestion, QuizAttempt
        mastery,  # MasteryRecord
        benchmark,  # BenchmarkRun, BenchmarkResult
        learning,  # RecognizedSymbol, PracticeAttempt
    )

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Minimal forward migration for SQLite dev databases created by older
    versions: add any nullable column that exists on a model but not in the table."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing and col.nullable:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}'))


def SessionLocal() -> Session:  # noqa: N802 - kept as a familiar name for scripts
    """Convenience factory used by CLI scripts: SessionLocal() -> Session."""
    return get_session_factory()()
