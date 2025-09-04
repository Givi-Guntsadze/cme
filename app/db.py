from sqlmodel import SQLModel, create_engine, Session

# Simple session factory
from contextlib import contextmanager

DB_URL = "sqlite:///./cme.sqlite"
engine = create_engine(DB_URL, echo=False)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session():
    with Session(engine) as session:
        yield session
