from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PricingPreview(BaseModel):
    tier_name: Optional[str] = None
    price_amount: Optional[float] = None
    currency: Optional[str] = None
    tier_count: int = 0


class EligibilityFlag(BaseModel):
    requirement_type: Optional[str] = None
    requirement_value: Optional[str] = None


class ActivitySummary(BaseModel):
    id: str
    provider_id: str
    provider_name: str
    title: str
    slug: Optional[str] = None
    modality: Optional[str] = None
    format: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    credits: Optional[float] = None
    primary_credit_type: Optional[str] = None
    pricing_preview: PricingPreview = Field(default_factory=PricingPreview)
    eligibility_flags: List[EligibilityFlag] = Field(default_factory=list)
    last_verified_at: Optional[datetime] = None
    confidence_score: Optional[float] = None
    field_gaps: List[str] = Field(default_factory=list)


class ActivityPricingTier(BaseModel):
    tier_name: Optional[str] = None
    price_amount: Optional[float] = None
    currency: Optional[str] = None
    price_type: Optional[str] = None
    eligibility_notes: Optional[str] = None
    discount_start: Optional[datetime] = None
    discount_end: Optional[datetime] = None
    taxes_and_fees: Optional[float] = None
    refund_policy: Optional[str] = None
    last_checked_at: Optional[datetime] = None


class ActivityCreditBreakdown(BaseModel):
    credit_type: str
    credit_quantity: float
    credit_expiration: Optional[date] = None
    notes: Optional[str] = None


class ActivityEligibilityDetail(BaseModel):
    requirement_type: Optional[str] = None
    requirement_value: Optional[str] = None
    notes: Optional[str] = None
    verified_at: Optional[datetime] = None


class ActivityCommitmentDetail(BaseModel):
    seat_time_hours: Optional[float] = None
    total_time_hours: Optional[float] = None
    completion_window_days: Optional[int] = None
    cohort_size: Optional[int] = None
    pacing: Optional[str] = None
    notes: Optional[str] = None


class RequirementMappingDetail(BaseModel):
    requirement_code: str
    coverage_min: Optional[float] = None
    coverage_max: Optional[float] = None
    derived_from: Optional[str] = None
    confidence: Optional[float] = None


class ActivityDetail(ActivitySummary):
    summary: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    pricing: List[ActivityPricingTier] = Field(default_factory=list)
    credit_types: List[ActivityCreditBreakdown] = Field(default_factory=list)
    eligibility: List[ActivityEligibilityDetail] = Field(default_factory=list)
    commitment: Optional[ActivityCommitmentDetail] = None
    requirement_mappings: List[RequirementMappingDetail] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    filters: dict = Field(default_factory=dict)
    total_hits: Optional[int] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    staleness_seconds: Optional[int] = None


class ActivityListResponse(BaseModel):
    results: List[ActivitySummary]
    next_cursor: Optional[str] = None
    retrieval_context: RetrievalContext


class ProviderSummary(BaseModel):
    id: str
    slug: str
    legal_name: str
    preferred_partner: bool
    activities_count: int
    last_catalog_update: Optional[datetime] = None


class ActivityFilter(BaseModel):
    board: Optional[str] = None
    requirement_codes: Optional[List[str]] = None
    modalities: Optional[List[str]] = None
    formats: Optional[List[str]] = None
    credit_types: Optional[List[str]] = None
    min_credits: Optional[float] = None
    max_price: Optional[float] = None
    start_after: Optional[date] = None
    end_before: Optional[date] = None
    allow_hybrid: Optional[bool] = None
    only_committed: Optional[bool] = None
    topic_ids: Optional[List[str]] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    search: Optional[str] = None


class ActivitySearchRequest(BaseModel):
    query: Optional[str] = None
    filters: Optional[ActivityFilter] = None
    max_results: int = 30
