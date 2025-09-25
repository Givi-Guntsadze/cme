import os
from datetime import date
from sqlmodel import Session, select, delete
from .models import Activity

SEED_ACTIVITIES = [
    dict(
        title="American Psychiatric Association Annual Meeting 2025",
        provider="APA",
        credits=24.0,
        cost_usd=1000.0,
        modality="live",
        city="New Orleans",
        start_date=date(2025, 5, 3),
        end_date=date(2025, 5, 6),
        days_required=4,
        hybrid_available=True,
        summary="Flagship APA meeting with hybrid access to plenaries and on-demand recordings.",
        pricing_options=[
            {
                "label": "APA member early bird",
                "cost_usd": 550.0,
                "deadline": "2025-02-15",
                "conditions": {"membership": ["APA"]},
            },
            {
                "label": "APA member",
                "cost_usd": 700.0,
                "deadline": "2025-04-01",
                "conditions": {"membership": ["APA"]},
            },
            {
                "label": "Early career psychiatrist",
                "cost_usd": 300.0,
                "deadline": "2025-04-01",
                "conditions": {"stage": ["early_career"]},
            },
            {
                "label": "Non-member standard",
                "cost_usd": 1000.0,
            },
        ],
    ),
    dict(
        title="Psych Congress Southwest Summit",
        provider="Psych Congress",
        credits=14.0,
        cost_usd=795.0,
        modality="live",
        city="Phoenix",
        start_date=date(2025, 10, 18),
        end_date=date(2025, 10, 20),
        days_required=3,
        hybrid_available=True,
        summary="Regional congress with hybrid attendance and focused workshops.",
        pricing_options=[
            {
                "label": "Early bird (through Jul 31)",
                "cost_usd": 595.0,
                "deadline": "2025-07-31",
            },
            {
                "label": "Member rate",
                "cost_usd": 495.0,
                "conditions": {"membership": ["Psych Congress", "APA"]},
            },
            {
                "label": "On-site",
                "cost_usd": 795.0,
            },
        ],
    ),
    dict(
        title="Ethics Case Consults for Psychiatrists (On-Demand)",
        provider="American Psychiatric Association",
        credits=6.0,
        cost_usd=249.0,
        modality="online",
        days_required=0,
        summary="Interactive ethics modules with downloadable case materials.",
        pricing_options=[
            {
                "label": "APA member",
                "cost_usd": 199.0,
                "conditions": {"membership": ["APA"]},
            },
            {
                "label": "Non-member",
                "cost_usd": 249.0,
            },
        ],
    ),
    dict(
        title="Psychopharmacology Master Class (Virtual Live)",
        provider="MGH Psychiatry Academy",
        credits=12.0,
        cost_usd=495.0,
        modality="live",
        city="Virtual",
        start_date=date(2025, 6, 12),
        end_date=date(2025, 6, 13),
        days_required=2,
        hybrid_available=False,
        summary="Live-streamed course with post-event replay window.",
        pricing_options=[
            {
                "label": "Early career (under 3 yrs)",
                "cost_usd": 395.0,
                "conditions": {
                    "stage": ["early_career"],
                    "max_years_post_residency": 4,
                },
            },
            {
                "label": "Standard",
                "cost_usd": 495.0,
            },
        ],
    ),
    dict(
        title="State Psychiatry Update Weekend",
        provider="State Psychiatric Society",
        credits=10.0,
        cost_usd=650.0,
        modality="live",
        city="Austin",
        start_date=date(2025, 9, 6),
        end_date=date(2025, 9, 7),
        days_required=2,
        hybrid_available=True,
        summary="State society annual meeting with in-person networking and live-stream option.",
        pricing_options=[
            {
                "label": "Member",
                "cost_usd": 450.0,
                "conditions": {"membership": ["Texas Psychiatric Society", "APA"]},
            },
            {
                "label": "Non-member",
                "cost_usd": 650.0,
            },
            {
                "label": "Late registration",
                "cost_usd": 800.0,
                "deadline": "2025-08-25",
            },
        ],
    ),
]


def seed_activities(session: Session) -> None:
    """Ensure demo seed activities exist without clobbering ingested rows."""

    force_env = os.getenv("FORCE_SEED")
    force = str(force_env).lower() in {"1", "true", "yes"}

    existing_seed = session.exec(
        select(Activity.id).where(Activity.source == "seed")
    ).first()

    if existing_seed and not force:
        return

    if existing_seed and force:
        session.exec(delete(Activity).where(Activity.source == "seed"))
        session.commit()

    for item in SEED_ACTIVITIES:
        session.add(Activity(**item))

    session.commit()
