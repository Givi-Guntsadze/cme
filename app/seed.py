from datetime import date
from sqlmodel import Session
from .models import Activity

SEED_ACTIVITIES = [
    dict(
        title="Ethics in Psychiatry",
        provider="Provider A",
        credits=3.0,
        cost_usd=75.0,
        modality="online",
        city=None,
        start_date=None,
        end_date=None,
        days_required=0,
    ),
    dict(
        title="Patient Safety in Behavioral Health",
        provider="Provider B",
        credits=2.0,
        cost_usd=0.0,
        modality="online",
        city=None,
        start_date=None,
        end_date=None,
        days_required=0,
    ),
    dict(
        title="JAMA Psychiatry Article Bundle",
        provider="JAMA",
        credits=5.0,
        cost_usd=0.0,
        modality="online",
        city=None,
        start_date=None,
        end_date=None,
        days_required=0,
    ),
    dict(
        title="Psychopharm Update",
        provider="Provider C",
        credits=4.0,
        cost_usd=99.0,
        modality="online",
        city=None,
        start_date=None,
        end_date=None,
        days_required=0,
    ),
    dict(
        title="State Psychiatry Update (local)",
        provider="State Society",
        credits=6.0,
        cost_usd=120.0,
        modality="live",
        city="Local",
        start_date=date(2025, 11, 5),
        end_date=date(2025, 11, 5),
        days_required=1,
    ),
    dict(
        title="Psychiatric Congress (Chicago)",
        provider="Congress Org",
        credits=20.0,
        cost_usd=650.0,
        modality="live",
        city="Chicago",
        start_date=date(2025, 11, 20),
        end_date=date(2025, 11, 21),
        days_required=2,
    ),
]


def seed_activities(session: Session) -> None:
    existing = session.query(Activity).count()
    if existing:
        return
    for a in SEED_ACTIVITIES:
        session.add(Activity(**a))
    session.commit()
