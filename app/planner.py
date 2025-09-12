from __future__ import annotations
from typing import List, Tuple, Dict, Any
from sqlmodel import select
from .models import User, Activity


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


# Returns (selected_activities, total_credits, total_cost, days_used)
def build_plan(user: User, session) -> Tuple[List[Activity], float, float, int]:
    remaining = max(user.remaining_credits, 0.0)
    if remaining <= 0:
        return ([], 0.0, 0.0, 0)

    q = select(Activity)
    activities = list(session.exec(q))

    # Filter by prefs
    if not user.allow_live:
        activities = [a for a in activities if a.modality == "online"]

    activities.sort(key=lambda a: _base_score(a, user.prefer_live))

    chosen: List[Activity] = []
    total_credits = 0.0
    total_cost = 0.0
    days_used = 0

    for a in activities:
        if total_credits >= remaining:
            break
        # Check constraints if adding this activity
        new_days = days_used + (a.days_required if a.modality == "live" else 0)
        new_cost = total_cost + a.cost_usd
        if new_days > user.days_off:
            continue
        if new_cost > user.budget_usd:
            continue
        chosen.append(a)
        total_credits += a.credits
        total_cost = new_cost
        days_used = new_days

    return (chosen, total_credits, total_cost, days_used)


def build_plan_with_policy(
    user: User, session, policy: Dict[str, Any]
) -> Tuple[List[Activity], float, float, int]:
    remaining = max(user.remaining_credits, 0.0)
    if remaining <= 0:
        return ([], 0.0, 0.0, 0)

    activities = list(session.exec(select(Activity)))

    # Hard-remove titles if requested
    remove_titles = [
        t.lower() for t in (policy.get("remove_titles") or []) if isinstance(t, str)
    ]
    if remove_titles:
        activities = [
            a
            for a in activities
            if all(rt not in (a.title or "").lower() for rt in remove_titles)
        ]

    # Respect allow_live
    if not user.allow_live:
        activities = [a for a in activities if a.modality == "online"]

    chosen: List[Activity] = []
    total_credits = 0.0
    total_cost = 0.0
    days_used = 0
    provider_counts: Dict[str, int] = {}

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
