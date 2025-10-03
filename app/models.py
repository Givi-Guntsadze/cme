from __future__ import annotations
from datetime import date, datetime, timezone
from typing import List, Optional
from sqlalchemy import Column
from sqlalchemy.types import Boolean
from sqlalchemy.types import JSON
from sqlmodel import SQLModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = "Doctor"
    specialty: str = "Psychiatry"
    city: Optional[str] = None
    budget_usd: float = 300.0
    days_off: int = 0
    target_credits: float = 10.0
    remaining_credits: float = 10.0
    allow_live: bool = False
    prefer_live: bool = False
    affiliations: Optional[List[str]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    memberships: Optional[List[str]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    training_level: Optional[str] = None
    professional_stage: Optional[str] = None  # e.g., early_career, standard, resident
    residency_completion_year: Optional[int] = None


class Activity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    provider: str
    credits: float
    cost_usd: float
    modality: str  # "online" | "live"
    city: Optional[str] = None
    # New metadata
    url: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = "seed"  # seed | web | ai
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    days_required: int = 0  # 0 online, 1 small live, 2 big live
    eligibility_text: Optional[str] = None
    eligible_institutions: Optional[List[str]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    eligible_groups: Optional[List[str]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    membership_required: Optional[str] = None
    open_to_public: bool = True
    hybrid_available: bool = False
    pricing_options: Optional[List[dict]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    requirement_tags: Optional[List[str]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )


class Claim(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    credits: float
    topic: Optional[str] = None
    date: date
    source_text: str


class PlanRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    mode: str = "balanced"
    generated_at: datetime = Field(default_factory=utcnow)
    status: str = "active"  # active | stale | superseded
    reason: Optional[str] = None
    total_credits: float = 0.0
    total_cost: float = 0.0
    days_used: int = 0
    remaining_credits: float = 0.0
    requirement_focus: Optional[List[str]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    context: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))


class PlanItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    activity_id: int = Field(foreign_key="activity.id")
    plan_run_id: Optional[int] = Field(default=None, foreign_key="planrun.id")
    mode: str = "balanced"
    position: int = 0
    chosen: bool = True
    pricing_snapshot: Optional[dict] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    requirement_snapshot: Optional[dict] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    eligibility_status: Optional[str] = None
    notes: Optional[str] = None
    generated_at: datetime = Field(default_factory=utcnow)
    committed: bool = Field(default=False, sa_column=Column(Boolean, default=False))


class AssistantMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    role: str = "assistant"  # 'user' | 'assistant'
    content: str
    created_at: datetime = Field(default_factory=utcnow)


class RequirementsSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    board: str
    specialty: str
    version: str
    effective_date: Optional[date] = None
    source_urls: Optional[List[str]] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    rules: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    content_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class UserPolicy(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    mode: str = "default"
    payload: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    ttl_days: int = 1
    active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: Optional[datetime] = None


class CompletedActivity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    activity_id: int = Field(foreign_key="activity.id")
    completed_at: datetime = Field(default_factory=utcnow)
