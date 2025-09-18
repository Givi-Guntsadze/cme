from __future__ import annotations
from datetime import date, datetime
from typing import List, Optional
from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import SQLModel, Field


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


class Claim(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    credits: float
    topic: Optional[str] = None
    date: date
    source_text: str


class PlanItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    activity_id: int = Field(foreign_key="activity.id")
    chosen: bool = True


class AssistantMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    role: str = "assistant"  # 'user' | 'assistant'
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserPolicy(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    mode: str = "default"
    payload: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    ttl_days: int = 1
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class CompletedActivity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    activity_id: int = Field(foreign_key="activity.id")
    completed_at: datetime = Field(default_factory=datetime.utcnow)
