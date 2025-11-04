from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid4())


class CatalogProvider(SQLModel, table=True):
    __tablename__ = "catalog_providers"

    id: str = Field(default_factory=_uuid, primary_key=True, index=True)
    slug: str = Field(index=True)
    legal_name: str
    dba_name: Optional[str] = None
    accreditation_number: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    website_url: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    city: Optional[str] = None
    state_province: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    preferred_partner: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class CatalogProviderRelationship(SQLModel, table=True):
    __tablename__ = "catalog_provider_relationships"

    id: str = Field(default_factory=_uuid, primary_key=True)
    provider_id: str = Field(foreign_key="catalog_providers.id", index=True)
    relationship_type: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    notes: Optional[str] = None


class CatalogActivity(SQLModel, table=True):
    __tablename__ = "catalog_activities"

    id: str = Field(default_factory=_uuid, primary_key=True, index=True)
    provider_id: str = Field(foreign_key="catalog_providers.id", index=True)
    canonical_title: str
    slug: Optional[str] = Field(default=None, index=True)
    description: Optional[str] = None
    modality: Optional[str] = Field(
        default=None
    )  # expected values: online, live, hybrid
    format: Optional[str] = None
    release_date: Optional[date] = None
    expiration_date: Optional[date] = None
    status: str = Field(default="active")
    max_claimable_credits: Optional[float] = None
    accreditation_statement: Optional[str] = None
    last_verified_at: Optional[datetime] = None
    data_confidence: Optional[float] = None
    revision_of_activity_id: Optional[str] = Field(
        default=None, foreign_key="catalog_activities.id"
    )
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    city: Optional[str] = None
    state_province: Optional[str] = None
    country: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = Field(default=None, index=True)
    field_gaps: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    provenance: dict = Field(default_factory=dict, sa_column=Column(JSON))


class CatalogTopic(SQLModel, table=True):
    __tablename__ = "catalog_topics"

    id: str = Field(default_factory=_uuid, primary_key=True, index=True)
    taxonomy_term: str
    taxonomy_path: Optional[str] = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utcnow)


class CatalogActivityTopic(SQLModel, table=True):
    __tablename__ = "catalog_activity_topics"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    topic_id: str = Field(foreign_key="catalog_topics.id", index=True)
    source: Optional[str] = None


class CatalogActivityCredit(SQLModel, table=True):
    __tablename__ = "catalog_activity_credits"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    credit_type: str
    credit_quantity: float = 0.0
    credit_expiration: Optional[date] = None
    notes: Optional[str] = None


class CatalogSession(SQLModel, table=True):
    __tablename__ = "catalog_activity_sessions"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    registration_deadline: Optional[datetime] = None
    time_zone: Optional[str] = None
    venue_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state_province: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    recurrence_pattern: Optional[str] = None
    capacity: Optional[int] = None
    waitlist_status: Optional[str] = None
    cancellation_policy_url: Optional[str] = None


class CatalogDeliveryLink(SQLModel, table=True):
    __tablename__ = "catalog_activity_delivery_links"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    access_url: Optional[str] = None
    platform_name: Optional[str] = None
    instructions: Optional[str] = None
    availability_start: Optional[datetime] = None
    availability_end: Optional[datetime] = None
    self_paced: Optional[bool] = None


class CatalogPricingTier(SQLModel, table=True):
    __tablename__ = "catalog_activity_pricing"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
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


class CatalogEligibilityRequirement(SQLModel, table=True):
    __tablename__ = "catalog_activity_eligibility"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    requirement_type: Optional[str] = None
    requirement_value: Optional[str] = None
    notes: Optional[str] = None
    verified_at: Optional[datetime] = None


class CatalogCommitment(SQLModel, table=True):
    __tablename__ = "catalog_activity_commitment"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    seat_time_hours: Optional[float] = None
    total_time_hours: Optional[float] = None
    completion_window_days: Optional[int] = None
    cohort_size: Optional[int] = None
    pacing: Optional[str] = None
    notes: Optional[str] = None


class CatalogDocument(SQLModel, table=True):
    __tablename__ = "catalog_activity_documents"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    document_type: Optional[str] = None
    file_url: Optional[str] = None
    checksum: Optional[str] = None
    ingest_source_id: Optional[str] = Field(
        default=None, foreign_key="catalog_ingest_sources.id"
    )


class CatalogRequirementMapping(SQLModel, table=True):
    __tablename__ = "catalog_requirement_mappings"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    requirement_code: str
    coverage_min: Optional[float] = None
    coverage_max: Optional[float] = None
    derived_from: Optional[str] = None
    confidence: Optional[float] = None


class CatalogIngestSource(SQLModel, table=True):
    __tablename__ = "catalog_ingest_sources"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    source_url: Optional[str] = None
    crawl_job_id: Optional[str] = None
    scraped_at: Optional[datetime] = None
    checksum: Optional[str] = None
    http_status: Optional[int] = None
    robots_mode: Optional[str] = None
    parser_version: Optional[str] = None
    manual_review_status: Optional[str] = None
    reviewer_id: Optional[str] = None


class CatalogQualitySignal(SQLModel, table=True):
    __tablename__ = "catalog_quality_signals"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    metric_type: Optional[str] = None
    metric_value: Optional[float] = None
    sample_size: Optional[int] = None
    collected_at: Optional[datetime] = None


class CatalogActivityEmbedding(SQLModel, table=True):
    __tablename__ = "catalog_activity_embeddings"

    id: str = Field(default_factory=_uuid, primary_key=True)
    activity_id: str = Field(foreign_key="catalog_activities.id", index=True)
    model: str
    embedding: list[float] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    embedding_version: Optional[str] = None
