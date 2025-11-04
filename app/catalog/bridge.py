from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

from sqlmodel import Session, select

from .models import (
    CatalogActivity,
    CatalogActivityCredit,
    CatalogCommitment,
    CatalogEligibilityRequirement,
    CatalogPricingTier,
    CatalogProvider,
    CatalogRequirementMapping,
)
from ..models import Activity


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
    requirement_map: Dict[str, List[CatalogRequirementMapping]] = {}

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
        requirement_map.setdefault(row.activity_id, []).append(row)
    return credits, pricing, eligibility, commitment, requirement_map


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


def _cheapest_price(tiers: List[CatalogPricingTier]) -> Optional[float]:
    prices = [
        tier.price_amount
        for tier in tiers
        if isinstance(tier.price_amount, (int, float))
    ]
    return min(prices) if prices else None


def _commitment_days(
    commitment: Optional[CatalogCommitment], start: Optional[date], end: Optional[date]
) -> int:
    if commitment and commitment.seat_time_hours:
        return max(1, int(round(commitment.seat_time_hours / 8.0)))
    if start and end:
        delta = (end - start).days + 1
        return max(delta, 1)
    return 0


def _extract_eligibility(
    rows: List[CatalogEligibilityRequirement],
) -> Tuple[str | None, List[str], List[str], str]:
    membership_required: Optional[str] = None
    eligible_institutions: List[str] = []
    eligible_groups: List[str] = []
    notes: List[str] = []

    for row in rows:
        note_text = row.notes or row.requirement_value or ""
        if note_text:
            notes.append(note_text)
        rtype = (row.requirement_type or "").lower()
        if "membership" in rtype:
            membership_required = row.requirement_value or row.notes or "membership"
        elif "institution" in rtype or "hospital" in rtype:
            if row.requirement_value:
                eligible_institutions.append(row.requirement_value)
        elif row.requirement_value:
            eligible_groups.append(row.requirement_value)

    eligibility_text = "; ".join(dict.fromkeys(notes)) if notes else ""
    return membership_required, eligible_institutions, eligible_groups, eligibility_text


def _requirement_tags(rows: List[CatalogRequirementMapping]) -> List[str]:
    tags: List[str] = []
    for row in rows:
        code = (row.requirement_code or "").strip()
        if not code:
            continue
        tag = code.split(".")[-1].lower()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _pricing_payload(tiers: List[CatalogPricingTier]) -> List[dict]:
    payload: List[dict] = []
    for tier in tiers:
        payload.append(
            {
                "label": tier.tier_name,
                "cost_usd": tier.price_amount,
                "currency": tier.currency or "USD",
                "price_type": tier.price_type,
                "notes": tier.eligibility_notes,
                "discount_start": (
                    tier.discount_start.isoformat() if tier.discount_start else None
                ),
                "discount_end": (
                    tier.discount_end.isoformat() if tier.discount_end else None
                ),
                "refund_policy": tier.refund_policy,
            }
        )
    return payload


def sync_catalog_to_activity_table(session: Session, limit: int = 200) -> int:
    rows = session.exec(
        select(CatalogActivity)
        .where(CatalogActivity.status == "active")
        .order_by(CatalogActivity.updated_at.desc())
        .limit(limit)
    ).all()
    if not rows:
        return 0

    provider_map = _provider_map(session, [row.provider_id for row in rows])
    ids = [row.id for row in rows]
    credits_map, pricing_map, eligibility_map, commitment_map, requirement_map = (
        _collect_related(session, ids)
    )

    updated = 0
    for row in rows:
        provider = provider_map.get(row.provider_id)
        credits = credits_map.get(row.id, [])
        pricing = pricing_map.get(row.id, [])
        eligibility_rows = eligibility_map.get(row.id, [])
        commitment_row = commitment_map.get(row.id)
        requirement_rows = requirement_map.get(row.id, [])

        cheapest_price = _cheapest_price(pricing) or 0.0
        total_credits = sum(
            credit.credit_quantity for credit in credits if credit.credit_quantity
        )
        if total_credits <= 0:
            total_credits = row.max_claimable_credits or 0.0
        (
            membership_required,
            eligible_institutions,
            eligible_groups,
            eligibility_text,
        ) = _extract_eligibility(eligibility_rows)
        requirement_tags = _requirement_tags(requirement_rows)
        pricing_payload = _pricing_payload(pricing)
        open_to_public = not (membership_required or eligible_institutions)
        modality = (row.modality or "online").lower()
        if modality not in {"online", "live", "hybrid"}:
            modality = "online"
        days_required = _commitment_days(
            commitment_row, row.release_date, row.expiration_date
        )

        existing = None
        if row.url:
            existing = session.exec(
                select(Activity).where(Activity.url == row.url).limit(1)
            ).first()
        if not existing:
            existing = session.exec(
                select(Activity)
                .where(
                    Activity.title == row.canonical_title,
                    Activity.provider
                    == (provider.legal_name if provider else row.provider_id),
                )
                .limit(1)
            ).first()
        if existing:
            activity = existing
        else:
            activity = Activity(
                title=row.canonical_title,
                provider=provider.legal_name if provider else "Unknown provider",
                credits=total_credits or 0.0,
                cost_usd=cheapest_price or 0.0,
                modality=modality,
            )

        activity.title = row.canonical_title
        activity.provider = provider.legal_name if provider else activity.provider
        activity.credits = total_credits or 0.0
        activity.cost_usd = float(cheapest_price or 0.0)
        activity.modality = modality
        activity.city = row.city
        activity.url = row.url
        activity.summary = row.summary
        activity.source = "catalog"
        activity.start_date = row.release_date
        activity.end_date = row.expiration_date
        activity.days_required = days_required
        activity.eligibility_text = eligibility_text or None
        activity.eligible_institutions = eligible_institutions
        activity.eligible_groups = eligible_groups
        activity.membership_required = membership_required
        activity.open_to_public = open_to_public
        activity.pricing_options = pricing_payload
        activity.requirement_tags = requirement_tags
        activity.hybrid_available = modality == "hybrid"

        session.add(activity)
        updated += 1

    if updated:
        session.commit()
    return updated
