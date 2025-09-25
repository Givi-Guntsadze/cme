from __future__ import annotations
import logging
from datetime import date
from typing import List, Tuple, Dict, Any, Optional
from sqlmodel import select
from .models import User, Activity, CompletedActivity


MIN_VARIETY_ITEMS = 4
LOGGER = logging.getLogger(__name__)

MEMBERSHIP_ALIASES: Dict[str, str] = {
    "american psychiatric association": "apa",
    "apa": "apa",
    "psych congress": "psych congress",
    "psychiatric congress": "psych congress",
    "texas psychiatric society": "texas psychiatric society",
    "american academy of child and adolescent psychiatry": "aacap",
    "aacap": "aacap",
}

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "addiction": [
        "addiction",
        "substance",
        "opioid",
        "alcohol",
        "use disorder",
        "dependence",
    ],
    "child_adolescent": ["child", "adolescent", "youth", "pediatric", "teen"],
    "geriatric": ["geriatric", "older", "aging", "senior"],
    "ethics": ["ethic", "risk", "legal", "compliance"],
    "practice": ["practice", "leadership", "management", "billing", "operations"],
    "telehealth": ["telehealth", "telepsychiatry", "virtual care", "remote"],
    "psychopharm": ["pharm", "pharmacology", "medication", "prescrib", "drug"],
    "culture": ["culture", "equity", "diversity", "inclusion", "bias"],
}


CHEAPEST = {
    "diversity_w": 0.0,  # no diversity pressure
    "single_cap_ratio": 1.0,  # allow 100% from one item
    "subscription_penalty": 0.0,  # no penalty
}

VARIETY = {
    "diversity_w": 0.55,  # stronger penalty for repeating same provider/modality
    "topic_diversity_w": 0.6,  # discourage repeating the same focus area
    "single_cap_ratio": 0.55,  # any single item ≤ 55% of remaining
    "subscription_penalty": 0.6,  # favor mixing over subscriptions
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


def _user_memberships(user: User) -> set[str]:
    memberships = getattr(user, "memberships", []) or []
    normalized: set[str] = set()
    for item in memberships:
        text = str(item).strip().lower()
        if not text:
            continue
        normalized.add(text)
        alias = MEMBERSHIP_ALIASES.get(text)
        if alias:
            normalized.add(alias)
    return normalized


def _years_since_residency(user: User) -> Optional[int]:
    year = getattr(user, "residency_completion_year", None)
    if year is None:
        return None
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return None
    current_year = date.today().year
    return max(current_year - year_int, 0)


def _normalize_stage_value(value: object | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    aliases = {
        "early": "early_career",
        "early career": "early_career",
        "early-career": "early_career",
        "early_career": "early_career",
        "standard": "standard",
        "established": "standard",
        "attending": "standard",
        "resident": "resident",
        "fellow": "resident",
        "trainee": "resident",
    }
    return aliases.get(text, text.replace(" ", "_"))


def pricing_context_for_user(user: User, activity: Activity) -> Dict[str, Any]:
    options = getattr(activity, "pricing_options", []) or []
    base_cost = float(activity.cost_usd or 0.0)
    today = date.today()
    user_memberships = _user_memberships(user)
    stage = _normalize_stage_value(getattr(user, "professional_stage", None))
    years_since = _years_since_residency(user)

    selected_cost = base_cost if base_cost > 0 else None
    selected_label: Optional[str] = None
    selected_deadline: Optional[date] = None
    discount_notes: List[str] = []
    missing_membership_info = bool(options) and not user_memberships
    missing_stage_info = bool(options) and stage is None and years_since is None

    def _coerce_deadline(raw_deadline: Any) -> Optional[date]:
        if not raw_deadline:
            return None
        try:
            return date.fromisoformat(str(raw_deadline))
        except Exception:
            return None

    def _cost_value(raw_cost: Any) -> Optional[float]:
        try:
            val = float(raw_cost)
            return val if val >= 0 else None
        except (TypeError, ValueError):
            return None

    for opt in options:
        if not isinstance(opt, dict):
            continue
        label = str(opt.get("label") or "").strip() or "Alternate pricing"
        cost_val = _cost_value(opt.get("cost_usd"))
        if cost_val is None:
            continue
        conditions = opt.get("conditions") or {}
        if not isinstance(conditions, dict):
            conditions = {}
        membership_requirements = conditions.get("membership") or conditions.get(
            "memberships"
        )
        if membership_requirements and not isinstance(membership_requirements, list):
            membership_requirements = [membership_requirements]
        membership_requirements = [
            str(m).strip().lower()
            for m in membership_requirements or []
            if str(m).strip()
        ]
        stage_requirements = conditions.get("stage") or conditions.get("stages")
        if stage_requirements and not isinstance(stage_requirements, list):
            stage_requirements = [stage_requirements]
        stage_requirements = [
            _normalize_stage_value(s) for s in stage_requirements or []
        ]
        stage_requirements = [s for s in stage_requirements if s]
        max_years = conditions.get("max_years_post_residency")
        try:
            max_years_val = float(max_years) if max_years is not None else None
        except (TypeError, ValueError):
            max_years_val = None

        deadline_obj = _coerce_deadline(opt.get("deadline"))

        unmet: List[str] = []
        if membership_requirements:
            if not user_memberships:
                unmet.append("membership_data")
            elif not any(m in user_memberships for m in membership_requirements):
                unmet.append("membership")
        if stage_requirements:
            if not stage:
                unmet.append("stage_data")
            elif stage not in stage_requirements:
                unmet.append("stage")
        if max_years_val is not None:
            if years_since is None:
                unmet.append("stage_data")
            elif years_since > max_years_val:
                unmet.append("stage")
        if deadline_obj and deadline_obj < today:
            unmet.append("deadline")

        qualifies = not unmet

        if qualifies:
            if selected_cost is None or cost_val <= selected_cost:
                selected_cost = cost_val
                selected_label = label
                selected_deadline = deadline_obj
            continue

        if "deadline" in unmet:
            continue

        # Capture potential savings if unmet is due to missing data/membership
        reasons = []
        if "membership_data" in unmet:
            reasons.append("Provide memberships for member pricing")
        elif "membership" in unmet:
            reasons.append("Only for listed members")
        if "stage_data" in unmet:
            reasons.append("Update professional stage to confirm eligibility")
        elif "stage" in unmet:
            reasons.append("Restricted to a different stage")
        if reasons and (selected_cost is None or cost_val <= selected_cost):
            discount_notes.append(f"{label}: ${cost_val:,.0f} ({'; '.join(reasons)})")

    # Fallback to base cost when no price selected
    if selected_cost is None:
        selected_cost = base_cost
        selected_label = None

    context: Dict[str, Any] = {
        "cost": selected_cost,
        "label": selected_label,
        "base_cost": base_cost if base_cost else selected_cost,
        "deadline": selected_deadline,
        "notes": "; ".join(discount_notes) if discount_notes else "",
        "missing_membership_data": missing_membership_info,
        "missing_stage_data": missing_stage_info,
        "hybrid_available": bool(getattr(activity, "hybrid_available", False)),
    }

    if selected_deadline:
        delta_days = (selected_deadline - today).days
        context["deadline_days"] = delta_days

    return context


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
    topic_counts: Dict[str, int] = {}
    pricing_cache: Dict[int, Dict[str, Any]] = {}

    def needs_more_variety() -> bool:
        return mode_key == "variety" and len(chosen) < MIN_VARIETY_ITEMS

    while (total_credits < remaining or needs_more_variety()) and activities:
        remaining_needed = max(remaining - total_credits, 0.0)

        def _pick_candidate(
            ignore_cap: bool = False,
            *,
            current_remaining: float = remaining_needed,
            current_days_used: int = days_used,
            current_total_cost: float = total_cost,
        ) -> tuple[
            Optional[Activity], float, Optional[Dict[str, Any]], Optional[str], bool
        ]:
            best_local = None
            best_score_local = float("inf")
            best_context_local: Optional[Dict[str, Any]] = None
            best_topic_local: Optional[str] = None
            skipped_for_cap = False

            for a in activities:
                new_days = current_days_used + (
                    a.days_required if a.modality == "live" else 0
                )
                if new_days > user.days_off:
                    continue

                cache_key = a.id or id(a)
                pricing = pricing_cache.get(cache_key)
                if not pricing:
                    pricing = pricing_context_for_user(user, a)
                    pricing_cache[cache_key] = pricing
                effective_cost = float(pricing.get("cost") or a.cost_usd or 0.0)

                new_cost = current_total_cost + effective_cost
                if new_cost > user.budget_usd:
                    continue

                if mode_key == "variety" and current_remaining > 0:
                    limit = config.get("single_cap_ratio", 1.0) * current_remaining
                    if limit > 0 and a.credits > limit and not ignore_cap:
                        skipped_for_cap = True
                        continue
                    if ignore_cap:
                        relaxed_limit = max(
                            current_remaining,
                            min(
                                max(remaining, current_remaining + 6.0),
                                remaining * 1.1,
                            ),
                        )
                        if a.credits > relaxed_limit:
                            continue

                if not a.credits:
                    continue

                base_score = effective_cost / a.credits
                score = base_score
                topic = getattr(a, "_topic_tag", None)
                if topic is None:
                    topic = _infer_topic(a)
                    a._topic_tag = topic

                if mode_key == "variety":
                    provider = (a.provider or "").lower()
                    modality = (a.modality or "").lower()
                    div_w = float(config.get("diversity_w") or 0.0)
                    score += div_w * provider_counts.get(provider, 0)
                    score += 0.5 * div_w * modality_counts.get(modality, 0)
                    topic_w = float(config.get("topic_diversity_w") or 0.0)
                    if topic_w:
                        score += topic_w * topic_counts.get(topic, 0)
                    if config.get("subscription_penalty") and _is_subscription(a):
                        score += float(config.get("subscription_penalty") or 0.0)
                    if effective_cost == 0.0:
                        score += 0.6

                if score < best_score_local:
                    best_score_local = score
                    best_local = a
                    best_context_local = pricing
                    best_topic_local = topic

            return (
                best_local,
                best_score_local,
                best_context_local,
                best_topic_local,
                skipped_for_cap,
            )

        best, best_score, best_context, best_topic, skipped_cap = _pick_candidate(False)
        if not best and skipped_cap and mode_key == "variety":
            best, best_score, best_context, best_topic, _ = _pick_candidate(True)

        if not best:
            break

        chosen.append(best)
        total_credits += best.credits
        if best_context is None:
            cache_key = best.id or id(best)
            best_context = pricing_cache.get(cache_key) or pricing_context_for_user(
                user, best
            )
        total_cost += float(best_context.get("cost") or best.cost_usd or 0.0)
        best._pricing_context = best_context
        if best.modality == "live":
            days_used += best.days_required

        if mode_key == "variety":
            provider = (best.provider or "").lower()
            modality = (best.modality or "").lower()
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            topic_key = (
                best_topic or getattr(best, "_topic_tag", None) or _infer_topic(best)
            )
            topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1

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
    modality_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    pricing_cache: Dict[int, Dict[str, Any]] = {}
    mode_key = (mode or "").lower()
    config = VARIETY if mode_key == "variety" else CHEAPEST

    # Greedy selection with policy-aware scoring
    while total_credits < remaining:
        best = None
        best_score = float("inf")
        best_context: Optional[Dict[str, Any]] = None
        best_topic: Optional[str] = None
        for a in activities:
            # Skip if would violate hard constraints
            new_days = days_used + (a.days_required if a.modality == "live" else 0)
            cache_key = a.id or id(a)
            pricing = pricing_cache.get(cache_key)
            if not pricing:
                pricing = pricing_context_for_user(user, a)
                pricing_cache[cache_key] = pricing
            effective_cost = float(pricing.get("cost") or a.cost_usd or 0.0)
            new_cost = total_cost + effective_cost
            if new_days > user.days_off:
                continue
            if new_cost > user.budget_usd:
                # allow small over if policy budget_tolerance
                tol = float(policy.get("budget_tolerance") or 0)
                if tol <= 0 or new_cost > user.budget_usd * (1 + tol):
                    continue
            base = (
                effective_cost / a.credits
                if a.credits
                else _base_score(a, bool(user.prefer_live))
            )
            adj = _policy_adjustments(
                a, policy, provider_counts, remaining - total_credits
            )
            score = max(0.0, base + adj)
            topic = getattr(a, "_topic_tag", None)
            if topic is None:
                topic = _infer_topic(a)
                a._topic_tag = topic

            if mode_key == "variety":
                if effective_cost == 0.0:
                    score += 0.6
                provider = (a.provider or "").lower()
                modality = (a.modality or "").lower()
                div_w = float(config.get("diversity_w") or 0.0)
                score += div_w * provider_counts.get(provider, 0)
                score += 0.5 * div_w * modality_counts.get(modality, 0)
                topic_w = float(config.get("topic_diversity_w") or 0.0)
                if topic_w:
                    score += topic_w * topic_counts.get(topic, 0)
            if score < best_score:
                best_score = score
                best = a
                best_context = pricing
                best_topic = topic
        if not best:
            break
        chosen.append(best)
        total_credits += best.credits
        if best_context is None:
            cache_key = best.id or id(best)
            best_context = pricing_cache.get(cache_key) or pricing_context_for_user(
                user, best
            )
        total_cost += float(best_context.get("cost") or best.cost_usd or 0.0)
        best._pricing_context = best_context
        if best.modality == "live":
            days_used += best.days_required
        key = (best.provider or "").lower()
        provider_counts[key] = provider_counts.get(key, 0) + 1
        if mode_key == "variety":
            modality = (best.modality or "").lower()
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            topic_key = (
                best_topic or getattr(best, "_topic_tag", None) or _infer_topic(best)
            )
            topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
        # Remove picked to avoid duplicates
        activities.remove(best)

    return (chosen, total_credits, total_cost, days_used)


def _infer_topic(activity: Activity) -> str:
    text_bits: List[str] = []
    if activity.title:
        text_bits.append(activity.title.lower())
    if activity.summary:
        text_bits.append(activity.summary.lower())
    text = " ".join(text_bits)
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return topic
    return "general"
