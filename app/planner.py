from __future__ import annotations
import logging
from typing import List, Tuple, Dict, Any
from sqlmodel import select
from .models import User, Activity, CompletedActivity


MIN_VARIETY_ITEMS = 3
LOGGER = logging.getLogger(__name__)


CHEAPEST = {
    "diversity_w": 0.0,  # no diversity pressure
    "single_cap_ratio": 1.0,  # allow 100% from one item
    "subscription_penalty": 0.0,  # no penalty
}

VARIETY = {
    "diversity_w": 0.35,  # penalize repeating same provider/modality
    "single_cap_ratio": 0.6,  # any single item ≤ 60% of remaining
    "subscription_penalty": 0.4,  # soft-penalize “subscription” items
}


def _is_subscription(a: Activity) -> bool:
    t = (a.title or "").lower()
    return any(k in t for k in ("subscription", "unlimited", "all-access"))


def _normalize_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, str):
        v = value.strip()
        return [v.lower()] if v else []
    return []


def is_eligible(user: User, activity: Activity) -> bool:
    if getattr(activity, "open_to_public", True):
        return True

    user_affiliations = _normalize_list(getattr(user, "affiliations", []) or [])
    user_memberships = _normalize_list(getattr(user, "memberships", []) or [])
    user_level = (getattr(user, "training_level", "") or "").strip().lower()

    eligible_institutions = _normalize_list(
        getattr(activity, "eligible_institutions", []) or []
    )
    if eligible_institutions:
        if not user_affiliations:
            return True
        if not any(inst in user_affiliations for inst in eligible_institutions):
            return False

    eligible_groups = _normalize_list(getattr(activity, "eligible_groups", []) or [])
    if eligible_groups:
        if not user_level:
            return True
        if user_level not in eligible_groups:
            return False

    membership_required = (
        (getattr(activity, "membership_required", None) or "").strip().lower()
    )
    if membership_required:
        if not user_memberships:
            return True
        if membership_required not in user_memberships:
            return False

    return True


def _base_score(activity: Activity, prefer_live: bool) -> float:
    cost_per_credit = (
        (activity.cost_usd / activity.credits) if activity.credits else 9999
    )
    live_penalty = 2.0 if activity.modality == "live" else 0.0
    if prefer_live:
        live_penalty = 0.8 if activity.modality == "live" else 0.0
    return cost_per_credit + live_penalty


def _policy_adjustments(
    activity: Activity,
    policy: Dict[str, Any],
    provider_counts: Dict[str, int],
    remaining: float,
) -> float:
    score_adj = 0.0
    title = (activity.title or "").lower()
    summary = (activity.summary or "").lower()
    provider = (activity.provider or "").lower()

    # Avoid terms penalty
    for t in policy.get("avoid_terms") or []:
        if t and (t.lower() in title or t.lower() in summary):
            score_adj += 3.0
            break

    # Prefer topics boost
    for t in policy.get("prefer_topics") or []:
        if t and (t.lower() in title or t.lower() in summary):
            score_adj -= 0.5
            break

    # Diversity penalty by provider
    div_w = float(policy.get("diversity_weight") or 0.0)
    if div_w > 0:
        score_adj += div_w * provider_counts.get(provider, 0)

    # Cap a single activity size if configured
    frac = policy.get("max_per_activity_fraction")
    try:
        frac = float(frac) if frac is not None else None
    except Exception:
        frac = None
    if frac and remaining > 0 and activity.credits > frac * remaining:
        score_adj += 3.0

    # Prefer live override
    if policy.get("prefer_live_override") is True and activity.modality == "live":
        score_adj -= 0.4
    if policy.get("prefer_live_override") is False and activity.modality == "live":
        score_adj += 0.8

    return score_adj


def _apply_policy_filters(
    activities: List[Activity], policy: Dict[str, Any]
) -> List[Activity]:
    if not policy:
        return activities

    original = list(activities)
    filtered = list(activities)

    remove_titles = [
        t.lower() for t in (policy.get("remove_titles") or []) if isinstance(t, str)
    ]
    if remove_titles:
        filtered = [
            a
            for a in filtered
            if all(rt not in (a.title or "").lower() for rt in remove_titles)
        ]

    min_allowed = max(5, int(0.3 * len(original))) if original else 0
    if original and len(filtered) < min_allowed:
        LOGGER.warning(
            "Policy filters removed too many activities (%s of %s); ignoring filters.",
            len(filtered),
            len(original),
        )
        return original

    return filtered


# Returns (selected_activities, total_credits, total_cost, days_used)
def build_plan(
    user: User, session, mode: str = "variety"
) -> Tuple[List[Activity], float, float, int]:
    remaining = max(user.remaining_credits, 0.0)
    if remaining <= 0:
        return ([], 0.0, 0.0, 0)

    q = select(Activity)
    activities = list(session.exec(q))

    completed_ids = set(
        session.exec(
            select(CompletedActivity.activity_id).where(
                CompletedActivity.user_id == user.id
            )
        )
    )
    if completed_ids:
        activities = [a for a in activities if a.id not in completed_ids]

    # Filter by prefs
    if not user.allow_live:
        activities = [a for a in activities if a.modality == "online"]
    activities = [a for a in activities if is_eligible(user, a)]

    chosen: List[Activity] = []
    total_credits = 0.0
    total_cost = 0.0
    days_used = 0

    mode_key = (mode or "").lower()
    config = VARIETY if mode_key == "variety" else CHEAPEST

    provider_counts: Dict[str, int] = {}
    modality_counts: Dict[str, int] = {}

    def needs_more_variety() -> bool:
        return mode_key == "variety" and len(chosen) < MIN_VARIETY_ITEMS

    while (total_credits < remaining or needs_more_variety()) and activities:
        best = None
        best_score = float("inf")
        remaining_needed = max(remaining - total_credits, 0.0)

        for a in activities:
            new_days = days_used + (a.days_required if a.modality == "live" else 0)
            if new_days > user.days_off:
                continue

            new_cost = total_cost + a.cost_usd
            if new_cost > user.budget_usd:
                continue

            if mode_key == "variety" and remaining_needed > 0:
                limit = config.get("single_cap_ratio", 1.0) * remaining_needed
                if a.credits > limit:
                    continue

            if not a.credits:
                continue

            base_score = a.cost_usd / a.credits
            score = base_score

            if mode_key == "variety":
                provider = (a.provider or "").lower()
                modality = (a.modality or "").lower()
                div_w = float(config.get("diversity_w") or 0.0)
                score += div_w * provider_counts.get(provider, 0)
                score += 0.5 * div_w * modality_counts.get(modality, 0)
                if config.get("subscription_penalty") and _is_subscription(a):
                    score += float(config.get("subscription_penalty") or 0.0)
                if (a.cost_usd or 0.0) == 0.0:
                    score += 0.6

            if score < best_score:
                best_score = score
                best = a

        if not best:
            break

        chosen.append(best)
        total_credits += best.credits
        total_cost += best.cost_usd
        if best.modality == "live":
            days_used += best.days_required

        if mode_key == "variety":
            provider = (best.provider or "").lower()
            modality = (best.modality or "").lower()
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        activities.remove(best)

    return (chosen, total_credits, total_cost, days_used)


def build_plan_with_policy(
    user: User, session, policy: Dict[str, Any], mode: str = "variety"
) -> Tuple[List[Activity], float, float, int]:
    remaining = max(user.remaining_credits, 0.0)
    if remaining <= 0:
        return ([], 0.0, 0.0, 0)

    q = select(Activity)
    activities = list(session.exec(q))

    completed_ids = set(
        session.exec(
            select(CompletedActivity.activity_id).where(
                CompletedActivity.user_id == user.id
            )
        )
    )
    if completed_ids:
        activities = [a for a in activities if a.id not in completed_ids]
    activities = _apply_policy_filters(activities, policy)

    # Respect allow_live
    if not user.allow_live:
        activities = [a for a in activities if a.modality == "online"]
    activities = [a for a in activities if is_eligible(user, a)]

    chosen: List[Activity] = []
    total_credits = 0.0
    total_cost = 0.0
    days_used = 0
    provider_counts: Dict[str, int] = {}
    mode_key = (mode or "").lower()

    # Greedy selection with policy-aware scoring
    while total_credits < remaining:
        best = None
        best_score = float("inf")
        for a in activities:
            # Skip if would violate hard constraints
            new_days = days_used + (a.days_required if a.modality == "live" else 0)
            new_cost = total_cost + a.cost_usd
            if new_days > user.days_off:
                continue
            if new_cost > user.budget_usd:
                # allow small over if policy budget_tolerance
                tol = float(policy.get("budget_tolerance") or 0)
                if tol <= 0 or new_cost > user.budget_usd * (1 + tol):
                    continue
            base = _base_score(a, bool(user.prefer_live))
            adj = _policy_adjustments(
                a, policy, provider_counts, remaining - total_credits
            )
            score = max(0.0, base + adj)
            if mode_key == "variety" and (a.cost_usd or 0.0) == 0.0:
                score += 0.6
            if score < best_score:
                best_score = score
                best = a
        if not best:
            break
        chosen.append(best)
        total_credits += best.credits
        total_cost += best.cost_usd
        if best.modality == "live":
            days_used += best.days_required
        key = (best.provider or "").lower()
        provider_counts[key] = provider_counts.get(key, 0) + 1
        # Remove picked to avoid duplicates
        activities.remove(best)

    return (chosen, total_credits, total_cost, days_used)
