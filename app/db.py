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
        if "affiliations" not in u_names:
            conn.execute(text("ALTER TABLE user ADD COLUMN affiliations TEXT"))
            conn.execute(
                text(
                    "UPDATE user SET affiliations='[]' WHERE affiliations IS NULL OR affiliations=''"
                )
            )
        if "memberships" not in u_names:
            conn.execute(text("ALTER TABLE user ADD COLUMN memberships TEXT"))
            conn.execute(
                text(
                    "UPDATE user SET memberships='[]' WHERE memberships IS NULL OR memberships=''"
                )
            )
        if "training_level" not in u_names:
            conn.execute(text("ALTER TABLE user ADD COLUMN training_level TEXT"))

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
        if "eligibility_text" not in names:
            conn.execute(text("ALTER TABLE activity ADD COLUMN eligibility_text TEXT"))
        if "eligible_institutions" not in names:
            conn.execute(
                text("ALTER TABLE activity ADD COLUMN eligible_institutions TEXT")
            )
            conn.execute(
                text(
                    "UPDATE activity SET eligible_institutions='[]' WHERE eligible_institutions IS NULL OR eligible_institutions=''"
                )
            )
        if "eligible_groups" not in names:
            conn.execute(text("ALTER TABLE activity ADD COLUMN eligible_groups TEXT"))
            conn.execute(
                text(
                    "UPDATE activity SET eligible_groups='[]' WHERE eligible_groups IS NULL OR eligible_groups=''"
                )
            )
        if "membership_required" not in names:
            conn.execute(
                text("ALTER TABLE activity ADD COLUMN membership_required TEXT")
            )
        if "open_to_public" not in names:
            conn.execute(text("ALTER TABLE activity ADD COLUMN open_to_public BOOLEAN"))
            conn.execute(
                text(
                    "UPDATE activity SET open_to_public=1 WHERE open_to_public IS NULL"
                )
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

        # RequirementsSnapshot table safety: ensure columns exist (in case of legacy)
        try:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS requirementssnapshot ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "board TEXT,"
                    "specialty TEXT,"
                    "version TEXT,"
                    "effective_date DATE,"
                    "source_urls TEXT,"
                    "rules TEXT,"
                    "content_hash TEXT,"
                    "created_at TIMESTAMP"
                    ")"
                )
            )
        except Exception:
            pass

        try:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS userpolicy ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "user_id INTEGER,"
                    "mode TEXT,"
                    "payload TEXT,"
                    "ttl_days INTEGER,"
                    "active BOOLEAN,"
                    "created_at TIMESTAMP,"
                    "expires_at TIMESTAMP"
                    ")"
                )
            )
        except Exception:
            pass

        try:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS completedactivity ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "user_id INTEGER,"
                    "activity_id INTEGER,"
                    "completed_at TIMESTAMP"
                    ")"
                )
            )
        except Exception:
            pass


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
