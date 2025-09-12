from __future__ import annotations
from datetime import date, datetime
from typing import Optional
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
