from __future__ import annotations
from typing import List, Tuple
from sqlmodel import select
from .models import User, Activity


# Simple scoring: lower is better
def _score(activity: Activity) -> float:
    cost_per_credit = (
        (activity.cost_usd / activity.credits) if activity.credits else 9999
    )
    travel_penalty = 2.0 if activity.modality == "live" else 0.0
    return cost_per_credit + travel_penalty


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

    activities.sort(key=_score)

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
