from sqlmodel import SQLModel, create_engine, Session

# Simple session factory
from contextlib import contextmanager
from sqlalchemy import text

DB_URL = "sqlite:///./cme.sqlite"
engine = create_engine(DB_URL, echo=False)


def _apply_migrations() -> None:
    """Minimal in-place migrations for SQLite."""
    with engine.begin() as conn:
        # User table: add prefer_live
        cols_u = conn.execute(text("PRAGMA table_info(user)")).fetchall()
        u_names = {row[1] for row in cols_u}
        if "prefer_live" not in u_names:
            conn.execute(text("ALTER TABLE user ADD COLUMN prefer_live BOOLEAN"))
            conn.execute(
                text("UPDATE user SET prefer_live=0 WHERE prefer_live IS NULL")
            )

        # Activity table: add metadata columns
        cols = conn.execute(text("PRAGMA table_info(activity)")).fetchall()
        names = {row[1] for row in cols}  # row[1] is column name
        if "url" not in names:
            conn.execute(text("ALTER TABLE activity ADD COLUMN url TEXT"))
        if "summary" not in names:
            conn.execute(text("ALTER TABLE activity ADD COLUMN summary TEXT"))
        if "source" not in names:
            conn.execute(text("ALTER TABLE activity ADD COLUMN source TEXT"))
        # Backfill source for existing rows
        conn.execute(
            text("UPDATE activity SET source='seed' WHERE source IS NULL OR source=''")
        )

        # AssistantMessage: add role
        cols_m = conn.execute(text("PRAGMA table_info(assistantmessage)")).fetchall()
        m_names = {row[1] for row in cols_m}
        if "role" not in m_names:
            conn.execute(text("ALTER TABLE assistantmessage ADD COLUMN role TEXT"))
            conn.execute(
                text(
                    "UPDATE assistantmessage SET role='assistant' WHERE role IS NULL OR role=''"
                )
            )


def purge_seed_activities() -> None:
    """Delete legacy seed activities if present."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM activity WHERE source='seed'"))


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)
    # Safe to run every startup
    try:
        _apply_migrations()
    except Exception:
        # If table does not exist yet, ignore (fresh DB will be created above)
        pass


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session
