from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from sqlmodel import Session, delete, select

from .embeddings import upsert_activity_embedding
from .models import (
    CatalogActivity,
    CatalogActivityCredit,
    CatalogActivityTopic,
    CatalogCommitment,
    CatalogEligibilityRequirement,
    CatalogIngestSource,
    CatalogPricingTier,
    CatalogProvider,
    CatalogRequirementMapping,
    CatalogTopic,
)
from .schemas import (
    ActivityDetail,
    ActivityEligibilityDetail,
    ActivityFilter,
    ActivityListResponse,
    ActivityPricingTier,
    ActivitySearchRequest,
    ActivitySummary,
    ActivityCommitmentDetail,
    ActivityCreditBreakdown,
    EligibilityFlag,
    PricingPreview,
    RequirementMappingDetail,
    RetrievalContext,
)

# simple slugify
SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = SLUG_RE.sub("-", text)
    return text.strip("-")


def _parse_date(raw: object) -> Optional[date]:
    if raw is None:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "")).date()
        except ValueError:
            return None
    return None


def ensure_provider(
    session: Session,
    slug: str,
    *,
    legal_name: Optional[str] = None,
    website_url: Optional[str] = None,
) -> CatalogProvider:
    legal = legal_name or slug.replace("-", " ").title()
    row = session.exec(
        select(CatalogProvider).where(CatalogProvider.slug == slug).limit(1)
    ).first()
    now = datetime.now(timezone.utc)
    if row:
        changed = False
        if legal and row.legal_name != legal:
            row.legal_name = legal
            changed = True
        if website_url and row.website_url != website_url:
            row.website_url = website_url
            changed = True
        if changed:
            row.updated_at = now
            session.add(row)
        return row
    row = CatalogProvider(slug=slug, legal_name=legal, website_url=website_url or None)
    row.created_at = now
    row.updated_at = now
    session.add(row)
    return row


def _float(value: object | None) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _activity_unique_key(activity: dict) -> Tuple[str, Optional[str]]:
    slug = activity.get("slug") or slugify(activity.get("title") or "")
    url = activity.get("url")
    return slug, url


def upsert_activity_bundle(session: Session, bundle: dict) -> CatalogActivity:
    activity_data = dict(bundle.get("activity") or {})
    provider_slug = activity_data.get("provider_slug") or slugify(
        activity_data.get("provider_name") or "unknown-provider"
    )
    provider = ensure_provider(
        session,
        provider_slug,
        legal_name=activity_data.get("provider_name"),
        website_url=activity_data.get("provider_url"),
    )

    slug, url = _activity_unique_key(activity_data)

    existing = None
    if url:
        existing = session.exec(
            select(CatalogActivity).where(CatalogActivity.url == url).limit(1)
        ).first()
    if not existing and slug:
        existing = session.exec(
            select(CatalogActivity)
            .where(
                CatalogActivity.provider_id == provider.id,
                CatalogActivity.slug == slug,
            )
            .limit(1)
        ).first()

    now = datetime.now(timezone.utc)
    summary = activity_data.get("summary") or activity_data.get("description")
    start_date = _parse_date(activity_data.get("start_date"))
    end_date = _parse_date(activity_data.get("end_date"))

    if existing:
        row = existing
        row.canonical_title = activity_data.get("title") or row.canonical_title
        row.slug = slug or row.slug
        row.description = activity_data.get("description") or row.description
        row.summary = summary or row.summary
        row.modality = activity_data.get("modality") or row.modality
        row.format = activity_data.get("format") or row.format
        row.release_date = start_date or row.release_date
        row.expiration_date = end_date or row.expiration_date
        row.url = url or row.url
        row.city = activity_data.get("city") or row.city
        row.state_province = activity_data.get("state") or row.state_province
        row.country = activity_data.get("country") or row.country
    else:
        row = CatalogActivity(
            provider_id=provider.id,
            canonical_title=activity_data.get("title") or "Untitled activity",
            slug=slug,
            description=activity_data.get("description"),
            summary=summary,
            modality=activity_data.get("modality"),
            format=activity_data.get("format"),
            release_date=start_date,
            expiration_date=end_date,
            url=url,
            city=activity_data.get("city"),
            state_province=activity_data.get("state"),
            country=activity_data.get("country"),
        )
    row.last_verified_at = now
    field_gaps = list(bundle.get("field_gaps") or [])
    row.field_gaps = field_gaps
    provenance = dict(bundle.get("provenance") or {})
    row.provenance = provenance
    row.max_claimable_credits = _float(activity_data.get("max_claimable_credits"))
    row.data_confidence = max(0.2, 1.0 - 0.1 * len(field_gaps))
    row.updated_at = now
    session.add(row)

    # Clear existing related rows
    session.exec(
        delete(CatalogActivityCredit).where(CatalogActivityCredit.activity_id == row.id)
    )
    session.exec(
        delete(CatalogPricingTier).where(CatalogPricingTier.activity_id == row.id)
    )
    session.exec(
        delete(CatalogEligibilityRequirement).where(
            CatalogEligibilityRequirement.activity_id == row.id
        )
    )
    session.exec(
        delete(CatalogRequirementMapping).where(
            CatalogRequirementMapping.activity_id == row.id
        )
    )
    session.exec(
        delete(CatalogCommitment).where(CatalogCommitment.activity_id == row.id)
    )
    session.exec(
        delete(CatalogActivityTopic).where(CatalogActivityTopic.activity_id == row.id)
    )

    # Credits
    credit_rows = []
    for credit in bundle.get("credit_types") or []:
        qty = _float(credit.get("credit_quantity")) or 0.0
        if qty <= 0:
            continue
        credit_rows.append(
            CatalogActivityCredit(
                activity_id=row.id,
                credit_type=credit.get("credit_type") or "unknown",
                credit_quantity=qty,
                credit_expiration=_parse_date(credit.get("credit_expiration")),
                notes=credit.get("notes"),
            )
        )
    if credit_rows:
        session.add_all(credit_rows)
        row.max_claimable_credits = max(
            row.max_claimable_credits or 0.0,
            max(c.credit_quantity for c in credit_rows),
        )

    # Pricing
    pricing_rows = []
    for tier in bundle.get("pricing_tiers") or []:
        pricing_rows.append(
            CatalogPricingTier(
                activity_id=row.id,
                tier_name=tier.get("tier_name"),
                price_amount=_float(tier.get("price_amount")),
                currency=tier.get("currency") or "USD",
                price_type=tier.get("price_type"),
                eligibility_notes=tier.get("eligibility_notes"),
                discount_start=_parse_datetime(tier.get("discount_start")),
                discount_end=_parse_datetime(tier.get("discount_end")),
                taxes_and_fees=_float(tier.get("taxes_and_fees")),
                refund_policy=tier.get("refund_policy"),
                last_checked_at=_parse_datetime(tier.get("last_checked_at")),
            )
        )
    if pricing_rows:
        session.add_all(pricing_rows)

    # Eligibility
    eligibility_rows = []
    for item in bundle.get("eligibility") or []:
        eligibility_rows.append(
            CatalogEligibilityRequirement(
                activity_id=row.id,
                requirement_type=item.get("requirement_type"),
                requirement_value=item.get("requirement_value"),
                notes=item.get("notes"),
                verified_at=_parse_datetime(item.get("verified_at")),
            )
        )
    if eligibility_rows:
        session.add_all(eligibility_rows)

    # Requirement mappings
    mapping_rows = []
    for requirement in bundle.get("requirement_mappings") or []:
        if isinstance(requirement, str):
            mapping_rows.append(
                CatalogRequirementMapping(
                    activity_id=row.id,
                    requirement_code=requirement,
                )
            )
        elif isinstance(requirement, dict):
            mapping_rows.append(
                CatalogRequirementMapping(
                    activity_id=row.id,
                    requirement_code=requirement.get("requirement_code")
                    or requirement.get("code")
                    or "unknown",
                    coverage_min=_float(requirement.get("coverage_min")),
                    coverage_max=_float(requirement.get("coverage_max")),
                    derived_from=requirement.get("derived_from"),
                    confidence=_float(requirement.get("confidence")),
                )
            )
    if mapping_rows:
        session.add_all(mapping_rows)

    # Commitment
    commitment_payload = bundle.get("commitment") or {}
    if commitment_payload:
        commitment_row = CatalogCommitment(
            activity_id=row.id,
            seat_time_hours=_float(commitment_payload.get("seat_time_hours")),
            total_time_hours=_float(commitment_payload.get("total_time_hours")),
            completion_window_days=commitment_payload.get("completion_window_days"),
            cohort_size=commitment_payload.get("cohort_size"),
            pacing=commitment_payload.get("pacing"),
            notes=commitment_payload.get("notes"),
        )
        session.add(commitment_row)

    # Topics if present
    for topic in bundle.get("topics") or []:
        term = topic.get("taxonomy_term") if isinstance(topic, dict) else str(topic)
        if not term:
            continue
        topic_slug = slugify(term)
        topic_row = session.exec(
            select(CatalogTopic).where(CatalogTopic.taxonomy_term == term).limit(1)
        ).first()
        if not topic_row:
            topic_row = CatalogTopic(taxonomy_term=term, taxonomy_path=topic_slug)
            session.add(topic_row)
            session.flush()
        session.add(
            CatalogActivityTopic(
                activity_id=row.id,
                topic_id=topic_row.id,
                source=topic.get("source") if isinstance(topic, dict) else None,
            )
        )

    # Provenance ingestion record
    if provenance:
        ingest_row = CatalogIngestSource(
            activity_id=row.id,
            source_url=provenance.get("source_hint") or provenance.get("source_url"),
            crawl_job_id=provenance.get("crawl_job_id"),
            scraped_at=_parse_datetime(provenance.get("extracted_at")),
            checksum=provenance.get("checksum"),
            http_status=provenance.get("http_status"),
            robots_mode=provenance.get("robots_mode"),
            parser_version=provenance.get("parser_version"),
            manual_review_status=provenance.get("review_status"),
            reviewer_id=provenance.get("reviewer_id"),
        )
        session.add(ingest_row)

    # Embedding
    content_chunks = [
        row.canonical_title,
        provider.legal_name,
        row.summary,
    ]
    cheapest = _cheapest_price_from_rows(pricing_rows)
    if cheapest is not None:
        content_chunks.append(f"Cheapest price: {cheapest}")
    for cred in credit_rows[:3]:
        content_chunks.append(f"{cred.credit_type}: {cred.credit_quantity} credits")
    for elig in eligibility_rows[:3]:
        if elig.notes:
            content_chunks.append(elig.notes)
        elif elig.requirement_value:
            content_chunks.append(elig.requirement_value)
    upsert_activity_embedding(session, row.id, content_chunks)

    return row


def _parse_datetime(raw: object) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _cheapest_price_from_rows(
    pricing_rows: Iterable[CatalogPricingTier],
) -> Optional[float]:
    prices = [
        tier.price_amount
        for tier in pricing_rows
        if isinstance(tier.price_amount, (int, float))
    ]
    return min(prices) if prices else None


def _collect_related(session: Session, activity_ids: List[str]) -> Tuple[
    Dict[str, List[CatalogActivityCredit]],
    Dict[str, List[CatalogPricingTier]],
    Dict[str, List[CatalogEligibilityRequirement]],
    Dict[str, CatalogCommitment],
    Dict[str, List[CatalogRequirementMapping]],
]:
    if not activity_ids:
        return {}, {}, {}, {}, {}
    credits: Dict[str, List[CatalogActivityCredit]] = {}
    pricing: Dict[str, List[CatalogPricingTier]] = {}
    eligibility: Dict[str, List[CatalogEligibilityRequirement]] = {}
    commitment: Dict[str, CatalogCommitment] = {}
    mappings: Dict[str, List[CatalogRequirementMapping]] = {}

    for row in session.exec(
        select(CatalogActivityCredit).where(
            CatalogActivityCredit.activity_id.in_(activity_ids)
        )
    ):
        credits.setdefault(row.activity_id, []).append(row)
    for row in session.exec(
        select(CatalogPricingTier).where(
            CatalogPricingTier.activity_id.in_(activity_ids)
        )
    ):
        pricing.setdefault(row.activity_id, []).append(row)
    for row in session.exec(
        select(CatalogEligibilityRequirement).where(
            CatalogEligibilityRequirement.activity_id.in_(activity_ids)
        )
    ):
        eligibility.setdefault(row.activity_id, []).append(row)
    for row in session.exec(
        select(CatalogCommitment).where(CatalogCommitment.activity_id.in_(activity_ids))
    ):
        commitment[row.activity_id] = row
    for row in session.exec(
        select(CatalogRequirementMapping).where(
            CatalogRequirementMapping.activity_id.in_(activity_ids)
        )
    ):
        mappings.setdefault(row.activity_id, []).append(row)
    return credits, pricing, eligibility, commitment, mappings


def _provider_map(
    session: Session, provider_ids: Iterable[str]
) -> Dict[str, CatalogProvider]:
    ids = list({pid for pid in provider_ids if pid})
    if not ids:
        return {}
    rows = session.exec(
        select(CatalogProvider).where(CatalogProvider.id.in_(ids))
    ).all()
    return {row.id: row for row in rows}


def _calc_total_credits(rows: List[CatalogActivityCredit]) -> Optional[float]:
    if not rows:
        return None
    return sum(credit.credit_quantity for credit in rows if credit.credit_quantity)


def _select_primary_credit(rows: List[CatalogActivityCredit]) -> Optional[str]:
    if not rows:
        return None
    sorted_rows = sorted(rows, key=lambda r: r.credit_quantity or 0, reverse=True)
    return sorted_rows[0].credit_type


def _pricing_preview(rows: List[CatalogPricingTier]) -> PricingPreview:
    if not rows:
        return PricingPreview()
    cheapest = min(
        (
            row.price_amount
            for row in rows
            if isinstance(row.price_amount, (int, float))
        ),
        default=None,
    )
    representative = rows[0]
    return PricingPreview(
        tier_name=representative.tier_name,
        price_amount=cheapest if cheapest is not None else representative.price_amount,
        currency=representative.currency,
        tier_count=len(rows),
    )


def _eligibility_flags(
    rows: List[CatalogEligibilityRequirement],
) -> List[EligibilityFlag]:
    flags: List[EligibilityFlag] = []
    for row in rows[:3]:
        flags.append(
            EligibilityFlag(
                requirement_type=row.requirement_type,
                requirement_value=row.requirement_value,
            )
        )
    return flags


def list_activities(
    session: Session,
    filters: Optional[ActivityFilter] = None,
    *,
    limit: int = 25,
    offset: int = 0,
) -> ActivityListResponse:
    filters = filters or ActivityFilter()
    rows = session.exec(
        select(CatalogActivity).where(CatalogActivity.status == "active")
    ).all()

    def passes_filters(row: CatalogActivity) -> bool:
        if filters.modalities and row.modality not in filters.modalities:
            if not (filters.allow_hybrid and row.modality == "hybrid"):
                return False
        if filters.formats and row.format not in filters.formats:
            return False
        if filters.city and (row.city or "").lower() != filters.city.lower():
            return False
        if (
            filters.state
            and (row.state_province or "").lower() != filters.state.lower()
        ):
            return False
        if filters.country and (row.country or "").lower() != filters.country.lower():
            return False
        if (
            filters.start_after
            and row.release_date
            and row.release_date < filters.start_after
        ):
            return False
        if (
            filters.end_before
            and row.expiration_date
            and row.expiration_date > filters.end_before
        ):
            return False
        if filters.search:
            needle = filters.search.lower()
            haystacks = [
                row.canonical_title or "",
                row.summary or "",
                row.description or "",
            ]
            if not any(needle in (h or "").lower() for h in haystacks):
                return False
        return True

    filtered = [row for row in rows if passes_filters(row)]

    # Additional filters requiring joins
    if (
        filters.credit_types
        or filters.requirement_codes
        or filters.max_price is not None
    ):
        activity_ids = [row.id for row in filtered]
        credits_map, pricing_map, _, _, requirement_map = _collect_related(
            session, activity_ids
        )
        if filters.credit_types:
            filtered = [
                row
                for row in filtered
                if any(
                    credit.credit_type in filters.credit_types
                    for credit in credits_map.get(row.id, [])
                )
            ]
        if filters.max_price is not None:
            filtered = [
                row
                for row in filtered
                if _cheapest_price_from_rows(pricing_map.get(row.id, [])) is not None
                and _cheapest_price_from_rows(pricing_map.get(row.id, []))
                <= filters.max_price
            ]
        if filters.requirement_codes:
            codes = {code.lower() for code in filters.requirement_codes}
            filtered = [
                row
                for row in filtered
                if any(
                    (mapping.requirement_code or "").lower() in codes
                    for mapping in requirement_map.get(row.id, [])
                )
            ]

    total = len(filtered)
    filtered.sort(
        key=lambda row: row.last_verified_at
        or row.updated_at
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    paged = filtered[offset : offset + limit]
    activity_ids = [row.id for row in paged]
    credits_map, pricing_map, eligibility_map, commitment_map, requirement_map = (
        _collect_related(session, activity_ids)
    )
    provider_map = _provider_map(session, [row.provider_id for row in paged])

    summaries: List[ActivitySummary] = []
    for row in paged:
        provider = provider_map.get(row.provider_id)
        credits = credits_map.get(row.id, [])
        pricing = pricing_map.get(row.id, [])
        eligibility = eligibility_map.get(row.id, [])
        summary = ActivitySummary(
            id=row.id,
            provider_id=row.provider_id,
            provider_name=provider.legal_name if provider else "Unknown provider",
            title=row.canonical_title,
            slug=row.slug,
            modality=row.modality,
            format=row.format,
            start_date=row.release_date,
            end_date=row.expiration_date,
            city=row.city,
            state=row.state_province,
            country=row.country,
            credits=_calc_total_credits(credits),
            primary_credit_type=_select_primary_credit(credits),
            pricing_preview=_pricing_preview(pricing),
            eligibility_flags=_eligibility_flags(eligibility),
            last_verified_at=row.last_verified_at,
            confidence_score=row.data_confidence,
            field_gaps=row.field_gaps or [],
        )
        summaries.append(summary)

    next_cursor = None
    if offset + limit < total:
        next_cursor = str(offset + limit)

    retrieval_context = RetrievalContext(
        filters=filters.dict(exclude_none=True),
        total_hits=total,
        retrieved_at=datetime.utcnow(),
        staleness_seconds=None,
    )
    return ActivityListResponse(
        results=summaries,
        next_cursor=next_cursor,
        retrieval_context=retrieval_context,
    )


def get_activity_detail(session: Session, activity_id: str) -> Optional[ActivityDetail]:
    row = session.get(CatalogActivity, activity_id)
    if not row:
        return None
    provider = session.get(CatalogProvider, row.provider_id)
    credits_map, pricing_map, eligibility_map, commitment_map, requirement_map = (
        _collect_related(session, [row.id])
    )
    credits = [
        ActivityCreditBreakdown(
            credit_type=item.credit_type,
            credit_quantity=item.credit_quantity,
            credit_expiration=item.credit_expiration,
            notes=item.notes,
        )
        for item in credits_map.get(row.id, [])
    ]
    pricing = [
        ActivityPricingTier(
            tier_name=item.tier_name,
            price_amount=item.price_amount,
            currency=item.currency,
            price_type=item.price_type,
            eligibility_notes=item.eligibility_notes,
            discount_start=item.discount_start,
            discount_end=item.discount_end,
            taxes_and_fees=item.taxes_and_fees,
            refund_policy=item.refund_policy,
            last_checked_at=item.last_checked_at,
        )
        for item in pricing_map.get(row.id, [])
    ]
    eligibility = [
        ActivityEligibilityDetail(
            requirement_type=item.requirement_type,
            requirement_value=item.requirement_value,
            notes=item.notes,
            verified_at=item.verified_at,
        )
        for item in eligibility_map.get(row.id, [])
    ]
    commitment_row = commitment_map.get(row.id)
    commitment = (
        ActivityCommitmentDetail(
            seat_time_hours=commitment_row.seat_time_hours,
            total_time_hours=commitment_row.total_time_hours,
            completion_window_days=commitment_row.completion_window_days,
            cohort_size=commitment_row.cohort_size,
            pacing=commitment_row.pacing,
            notes=commitment_row.notes,
        )
        if commitment_row
        else None
    )
    requirement_mappings = [
        RequirementMappingDetail(
            requirement_code=item.requirement_code,
            coverage_min=item.coverage_min,
            coverage_max=item.coverage_max,
            derived_from=item.derived_from,
            confidence=item.confidence,
        )
        for item in requirement_map.get(row.id, [])
    ]

    summary = ActivitySummary(
        id=row.id,
        provider_id=row.provider_id,
        provider_name=provider.legal_name if provider else "Unknown provider",
        title=row.canonical_title,
        slug=row.slug,
        modality=row.modality,
        format=row.format,
        start_date=row.release_date,
        end_date=row.expiration_date,
        city=row.city,
        state=row.state_province,
        country=row.country,
        credits=_calc_total_credits(credits_map.get(row.id, [])),
        primary_credit_type=_select_primary_credit(credits_map.get(row.id, [])),
        pricing_preview=_pricing_preview(pricing_map.get(row.id, [])),
        eligibility_flags=_eligibility_flags(eligibility_map.get(row.id, [])),
        last_verified_at=row.last_verified_at,
        confidence_score=row.data_confidence,
        field_gaps=row.field_gaps or [],
    )

    detail = ActivityDetail(**summary.dict())
    detail.summary = row.summary
    detail.description = row.description
    detail.url = row.url
    detail.pricing = pricing
    detail.credit_types = credits
    detail.eligibility = eligibility
    detail.commitment = commitment
    detail.requirement_mappings = requirement_mappings
    detail.provenance = row.provenance or {}
    return detail


def search_activities(
    session: Session, request: ActivitySearchRequest
) -> ActivityListResponse:
    filters = request.filters or ActivityFilter()
    if request.query:
        filters = filters.copy(update={"search": request.query})
    return list_activities(session, filters, limit=request.max_results, offset=0)
