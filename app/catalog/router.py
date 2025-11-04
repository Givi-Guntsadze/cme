from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query

from sqlmodel import select

from ..db import get_session
from .models import CatalogActivity, CatalogProvider
from .schemas import (
    ActivityDetail,
    ActivityFilter,
    ActivityListResponse,
    ActivitySearchRequest,
    ProviderSummary,
)
from .service import get_activity_detail, list_activities, search_activities


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/activities", response_model=ActivityListResponse)
def list_catalog_activities(
    limit: int = Query(25, ge=1, le=100),
    cursor: str | None = Query(default=None, description="Opaque cursor (offset)"),
    modalities: list[str] | None = Query(default=None),
    formats: list[str] | None = Query(default=None),
    credit_types: list[str] | None = Query(default=None),
    requirement_codes: list[str] | None = Query(default=None),
    max_price: float | None = Query(default=None),
    start_after: str | None = None,
    end_before: str | None = None,
    city: str | None = None,
    state: str | None = None,
    country: str | None = None,
    search: str | None = None,
) -> ActivityListResponse:
    offset = 0
    if cursor:
        try:
            offset = int(cursor)
        except ValueError:
            offset = 0
    filters = ActivityFilter(
        modalities=modalities or None,
        formats=formats or None,
        credit_types=credit_types or None,
        requirement_codes=requirement_codes or None,
        max_price=max_price,
        start_after=_parse_date(start_after),
        end_before=_parse_date(end_before),
        city=city,
        state=state,
        country=country,
        search=search,
    )
    with get_session() as session:
        response = list_activities(session, filters, limit=limit, offset=offset)
    return response


@router.get("/activities/{activity_id}", response_model=ActivityDetail)
def get_catalog_activity(activity_id: str) -> ActivityDetail:
    with get_session() as session:
        detail = get_activity_detail(session, activity_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Activity not found")
    return detail


@router.post("/search", response_model=ActivityListResponse)
def search_catalog(request: ActivitySearchRequest) -> ActivityListResponse:
    with get_session() as session:
        return search_activities(session, request)


@router.get("/providers/{provider_id}", response_model=ProviderSummary)
def get_provider(provider_id: str) -> ProviderSummary:
    with get_session() as session:
        provider = session.get(CatalogProvider, provider_id)
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        activity_count = session.exec(
            select(CatalogActivity).where(CatalogActivity.provider_id == provider.id)
        ).all()
    return ProviderSummary(
        id=provider.id,
        slug=provider.slug,
        legal_name=provider.legal_name,
        preferred_partner=provider.preferred_partner,
        activities_count=len(activity_count),
        last_catalog_update=provider.updated_at,
    )
