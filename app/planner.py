from __future__ import annotations
import logging
from datetime import date
from typing import List, Tuple, Dict, Any, Optional
from sqlmodel import select
from .models import User, Activity, CompletedActivity, Claim
from .requirements import (
    SA_CME_KEYWORDS,
    PIP_KEYWORDS,
    PATIENT_SAFETY_KEYWORDS,
    classify_topic,
    load_abpn_psychiatry_requirements,
    pip_activity_count,
    requirements_rules,
    sa_cme_credit_sum,
)


MIN_BALANCED_ITEMS = 4
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
    "patient_safety": [
        "patient safety",
        "risk management",
        "error prevention",
        "quality improvement",
    ],
    "practice": ["practice", "leadership", "management", "billing", "operations"],
    "telehealth": ["telehealth", "telepsychiatry", "virtual care", "remote"],
    "psychopharm": ["pharm", "pharmacology", "medication", "prescrib", "drug"],
    "culture": ["culture", "equity", "diversity", "inclusion", "bias"],
}

CHEAPEST = {
    "diversity_w": 0.0,
    "single_cap_ratio": 1.0,
    "subscription_penalty": 0.1,
    "topic_diversity_w": 0.0,
}

BALANCED = {
    "diversity_w": 0.45,
    "topic_diversity_w": 0.45,
    "single_cap_ratio": 0.5,
    "subscription_penalty": 1.0,
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


def _activity_text(activity: Activity) -> str:
    text_bits: List[str] = []
    if activity.title:
        text_bits.append(activity.title.lower())
    if activity.summary:
        text_bits.append(activity.summary.lower())
    return " ".join(text_bits)


def _is_patient_safety_activity(activity: Activity) -> bool:
    tags = set(getattr(activity, "requirement_tags", []) or [])
    if "patient_safety" in tags:
        return True
    text = _activity_text(activity)
    if not text:
        return False
    if "patient" in text and "safety" in text:
        return True
    for kw in PATIENT_SAFETY_KEYWORDS:
        if kw in text:
            return True
    return False


def _is_sa_cme_activity(activity: Activity) -> bool:
    tags = set(getattr(activity, "requirement_tags", []) or [])
    if "sa_cme" in tags:
        return True
    text = _activity_text(activity)
    if not text:
        return False
    return any(keyword in text for keyword in SA_CME_KEYWORDS)


def _is_pip_activity(activity: Activity) -> bool:
    tags = set(getattr(activity, "requirement_tags", []) or [])
    if "pip" in tags:
        return True
    text = _activity_text(activity)
    if not text:
        return False
    return any(keyword in text for keyword in PIP_KEYWORDS)


def _requirements_planning_context(session, user: User) -> dict[str, object]:
    requirements = load_abpn_psychiatry_requirements()
    rules = requirements_rules(requirements)
    safety_block = rules.get("patient_safety_activity")
    safety_required = False
    if isinstance(safety_block, dict):
        safety_required = bool(safety_block.get("required"))
    elif isinstance(safety_block, bool):
        safety_required = bool(safety_block)
    stmt = select(Claim).where(Claim.user_id == user.id)
    claims = list(session.exec(stmt))
    safety_completed = any(classify_topic(claim) == "safety" for claim in claims)
    sa_min = float(rules.get("sa_cme_min_per_cycle") or 0.0)
    sa_total = sa_cme_credit_sum(claims)
    sa_needed = max(sa_min - sa_total, 0.0)
    pip_required = int(rules.get("pip_required_per_cycle") or 0)
    pip_completed = pip_activity_count(claims)
    pip_needed = max(pip_required - pip_completed, 0)
    return {
        "requirements": requirements,
        "needs_patient_safety": bool(safety_required and not safety_completed),
        "has_patient_safety_credit": safety_completed,
        "sa_credits_needed": sa_needed,
        "pip_needed": pip_needed,
    }


def requirements_gap_summary(session, user: User) -> dict[str, object]:
    ctx = _requirements_planning_context(session, user)
    return {
        "needs_patient_safety": bool(ctx.get("needs_patient_safety")),
        "sa_credits_needed": float(ctx.get("sa_credits_needed") or 0.0),
        "pip_needed": int(ctx.get("pip_needed") or 0),
        "requirements": ctx.get("requirements"),
    }


def pricing_context_for_user(user: User, activity: Activity) -> Dict[str, Any]:
    options = getattr(activity, "pricing_options", []) or []
    try:
        base_cost = float(activity.cost_usd)
    except (TypeError, ValueError):
        base_cost = None
    today = date.today()
    user_memberships = _user_memberships(user)
    stage = _normalize_stage_value(getattr(user, "professional_stage", None))
    years_since = _years_since_residency(user)

    selected_cost = base_cost if (base_cost is not None and base_cost > 0) else None
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
        "base_cost": base_cost if base_cost is not None else selected_cost,
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


def build_plan(
    user: User,
    session,
    mode: str = "balanced",
    *,
    remaining_override: Optional[float] = None,
    budget_override: Optional[float] = None,
    days_override: Optional[int] = None,
    exclude_ids: Optional[set[int]] = None,
) -> Tuple[List[Activity], float, float, int]:
    remaining_target = (
        remaining_override
        if remaining_override is not None
        else max(float(user.remaining_credits or 0.0), 0.0)
    )

    budget_cap = (
        float(budget_override)
        if budget_override is not None
        else float(user.budget_usd or 0.0)
    )
    day_cap = (
        int(days_override)
        if days_override is not None
        else int(getattr(user, "days_off", 0) or 0)
    )

    excluded = set(exclude_ids or set())

    q = select(Activity)
    activities = [a for a in session.exec(q) if (a.id not in excluded)]

    completed_ids = set(
        session.exec(
            select(CompletedActivity.activity_id).where(
                CompletedActivity.user_id == user.id
            )
        )
    )
    if completed_ids:
        activities = [a for a in activities if a.id not in completed_ids]

    if not user.allow_live:
        activities = [a for a in activities if a.modality == "online"]
    activities = [a for a in activities if is_eligible(user, a)]

    req_ctx = _requirements_planning_context(session, user)
    patient_safety_pending = bool(req_ctx.get("needs_patient_safety"))
    if patient_safety_pending and not any(
        _is_patient_safety_activity(a) for a in activities
    ):
        patient_safety_pending = False

    sa_needed = float(req_ctx.get("sa_credits_needed") or 0.0)
    if sa_needed > 0 and not any(_is_sa_cme_activity(a) for a in activities):
        sa_needed = 0.0

    pip_needed = int(req_ctx.get("pip_needed") or 0)
    if pip_needed > 0 and not any(_is_pip_activity(a) for a in activities):
        pip_needed = 0

    requirement_floor = remaining_target
    if sa_needed > 0:
        requirement_floor = max(requirement_floor, sa_needed)
    if patient_safety_pending:
        requirement_floor = max(requirement_floor, 1.0)
    if pip_needed > 0:
        requirement_floor = max(requirement_floor, 5.0 * pip_needed)

    remaining_target = requirement_floor
    if remaining_target <= 0:
        return ([], 0.0, 0.0, 0)

    chosen: List[Activity] = []
    total_credits = 0.0
    total_cost = 0.0
    days_used = 0

    mode_key = (mode or "").lower()
    config = BALANCED if mode_key == "balanced" else CHEAPEST

    provider_counts: Dict[str, int] = {}
    modality_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    pricing_cache: Dict[int, Dict[str, Any]] = {}

    def needs_more_balanced() -> bool:
        return mode_key == "balanced" and len(chosen) < MIN_BALANCED_ITEMS

    while (total_credits < remaining_target or needs_more_balanced()) and activities:
        remaining_needed = max(remaining_target - total_credits, 0.0)

        def _pick_candidate(
            ignore_cap: bool = False,
            *,
            current_remaining: float = remaining_needed,
            current_days_used: int = days_used,
            current_total_cost: float = total_cost,
            patient_safety_required: bool = patient_safety_pending,
            sa_credits_required: float = sa_needed,
            pip_required: int = pip_needed,
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
                if new_days > day_cap:
                    continue

                cache_key = a.id or id(a)
                pricing = pricing_cache.get(cache_key)
                if not pricing:
                    pricing = pricing_context_for_user(user, a)
                    pricing_cache[cache_key] = pricing
                effective_cost = float(pricing.get("cost") or a.cost_usd or 0.0)

                new_cost = current_total_cost + effective_cost
                if new_cost > budget_cap:
                    continue

                if mode_key == "balanced" and current_remaining > 0:
                    limit = config.get("single_cap_ratio", 1.0) * current_remaining
                    if limit > 0 and a.credits > limit and not ignore_cap:
                        skipped_for_cap = True
                        continue
                    if ignore_cap:
                        relaxed_limit = max(
                            current_remaining,
                            min(
                                max(remaining_target, current_remaining + 6.0),
                                remaining_target * 1.1,
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

                if mode_key == "balanced":
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
                else:
                    if config.get("subscription_penalty") and _is_subscription(a):
                        score += float(config.get("subscription_penalty") or 0.0)

                safety_activity = _is_patient_safety_activity(a)
                sa_activity = _is_sa_cme_activity(a)
                pip_activity = _is_pip_activity(a)
                if patient_safety_required:
                    if safety_activity:
                        score -= 200.0
                    else:
                        score += 5.0
                if sa_credits_required > 0:
                    if sa_activity:
                        score -= 120.0 * min(a.credits or 0.0, sa_credits_required)
                    else:
                        score += 3.0
                if pip_required > 0:
                    if pip_activity:
                        score -= 500.0
                    else:
                        score += 200.0

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
        if not best and skipped_cap and mode_key == "balanced":
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
        requirement_tags = set(getattr(best, "requirement_tags", []) or [])
        if _is_patient_safety_activity(best):
            requirement_tags.add("patient_safety")
        if _is_sa_cme_activity(best):
            requirement_tags.add("sa_cme")
        if _is_pip_activity(best):
            requirement_tags.add("pip")
        best._requirement_tags = requirement_tags
        if best.modality == "live":
            days_used += best.days_required

        if mode_key == "balanced":
            provider = (best.provider or "").lower()
            modality = (best.modality or "").lower()
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            topic_key = (
                best_topic or getattr(best, "_topic_tag", None) or _infer_topic(best)
            )
            topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1
        if patient_safety_pending and "patient_safety" in requirement_tags:
            patient_safety_pending = False
        if sa_needed > 0 and "sa_cme" in requirement_tags:
            sa_needed = max(sa_needed - float(best.credits or 0.0), 0.0)
        if pip_needed > 0 and "pip" in requirement_tags:
            pip_needed = max(pip_needed - 1, 0)

        activities.remove(best)

    return (chosen, total_credits, total_cost, days_used)


def build_plan_with_policy(
    user: User,
    session,
    policy: Dict[str, Any],
    mode: str = "balanced",
    *,
    remaining_override: Optional[float] = None,
    budget_override: Optional[float] = None,
    days_override: Optional[int] = None,
    exclude_ids: Optional[set[int]] = None,
) -> Tuple[List[Activity], float, float, int]:
    remaining_target = (
        remaining_override
        if remaining_override is not None
        else max(float(user.remaining_credits or 0.0), 0.0)
    )
    budget_cap = (
        float(budget_override)
        if budget_override is not None
        else float(user.budget_usd or 0.0)
    )
    day_cap = (
        int(days_override)
        if days_override is not None
        else int(getattr(user, "days_off", 0) or 0)
    )

    excluded = set(exclude_ids or set())

    q = select(Activity)
    activities = [a for a in session.exec(q) if (a.id not in excluded)]

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

    if not user.allow_live:
        activities = [a for a in activities if a.modality == "online"]
    activities = [a for a in activities if is_eligible(user, a)]

    req_ctx = _requirements_planning_context(session, user)
    patient_safety_pending = bool(req_ctx.get("needs_patient_safety"))
    if patient_safety_pending and not any(
        _is_patient_safety_activity(a) for a in activities
    ):
        patient_safety_pending = False

    sa_needed = float(req_ctx.get("sa_credits_needed") or 0.0)
    if sa_needed > 0 and not any(_is_sa_cme_activity(a) for a in activities):
        sa_needed = 0.0

    pip_needed = int(req_ctx.get("pip_needed") or 0)
    if pip_needed > 0 and not any(_is_pip_activity(a) for a in activities):
        pip_needed = 0

    requirement_floor = remaining_target
    if sa_needed > 0:
        requirement_floor = max(requirement_floor, sa_needed)
    if patient_safety_pending:
        requirement_floor = max(requirement_floor, 1.0)
    if pip_needed > 0:
        requirement_floor = max(requirement_floor, 5.0 * pip_needed)

    remaining_target = requirement_floor
    if remaining_target <= 0:
        return ([], 0.0, 0.0, 0)

    chosen: List[Activity] = []
    total_credits = 0.0
    total_cost = 0.0
    days_used = 0
    provider_counts: Dict[str, int] = {}
    modality_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    pricing_cache: Dict[int, Dict[str, Any]] = {}
    mode_key = (mode or "").lower()
    config = BALANCED if mode_key == "balanced" else CHEAPEST

    def needs_more_balanced() -> bool:
        return mode_key == "balanced" and len(chosen) < MIN_BALANCED_ITEMS

    while (total_credits < remaining_target or needs_more_balanced()) and activities:
        remaining_needed = max(remaining_target - total_credits, 0.0)
        best = None
        best_score = float("inf")
        best_context: Optional[Dict[str, Any]] = None
        best_topic: Optional[str] = None
        for a in activities:
            new_days = days_used + (a.days_required if a.modality == "live" else 0)
            if new_days > day_cap:
                continue
            cache_key = a.id or id(a)
            pricing = pricing_cache.get(cache_key)
            if not pricing:
                pricing = pricing_context_for_user(user, a)
                pricing_cache[cache_key] = pricing
            effective_cost = float(pricing.get("cost") or a.cost_usd or 0.0)
            new_cost = total_cost + effective_cost
            tol = float(policy.get("budget_tolerance") or 0)
            over_budget_allowed = budget_cap * (1 + max(tol, 0.0))
            if new_cost > over_budget_allowed:
                continue

            base = (
                effective_cost / a.credits
                if a.credits
                else _base_score(a, bool(user.prefer_live))
            )
            adj = _policy_adjustments(a, policy, provider_counts, remaining_needed)
            score = max(0.0, base + adj)
            topic = getattr(a, "_topic_tag", None)
            if topic is None:
                topic = _infer_topic(a)
                a._topic_tag = topic

            if mode_key == "balanced":
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
                if config.get("subscription_penalty") and _is_subscription(a):
                    score += float(config.get("subscription_penalty") or 0.0)
            else:
                if config.get("subscription_penalty") and _is_subscription(a):
                    score += float(config.get("subscription_penalty") or 0.0)

            safety_activity = _is_patient_safety_activity(a)
            sa_activity = _is_sa_cme_activity(a)
            pip_activity = _is_pip_activity(a)
            if patient_safety_pending:
                score += -200.0 if safety_activity else 5.0
            if sa_needed > 0:
                score += (
                    -120.0 * min(a.credits or 0.0, sa_needed) if sa_activity else 3.0
                )
            if pip_needed > 0:
                score += -500.0 if pip_activity else 200.0

            if score < best_score:
                best_score = score
                best = a
                best_context = pricing
                best_topic = topic

        if not best:
            break

        chosen.append(best)
        total_credits += best.credits or 0.0
        if best_context is None:
            cache_key = best.id or id(best)
            best_context = pricing_cache.get(cache_key) or pricing_context_for_user(
                user, best
            )
        total_cost += float(best_context.get("cost") or best.cost_usd or 0.0)
        requirement_tags = set(getattr(best, "requirement_tags", []) or [])
        if _is_patient_safety_activity(best):
            requirement_tags.add("patient_safety")
        if _is_sa_cme_activity(best):
            requirement_tags.add("sa_cme")
        if _is_pip_activity(best):
            requirement_tags.add("pip")
        best._requirement_tags = requirement_tags
        if best.modality == "live":
            days_used += best.days_required

        provider = (best.provider or "").lower()
        modality = (best.modality or "").lower()
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
        topic_key = (
            best_topic or getattr(best, "_topic_tag", None) or _infer_topic(best)
        )
        topic_counts[topic_key] = topic_counts.get(topic_key, 0) + 1

        if patient_safety_pending and "patient_safety" in requirement_tags:
            patient_safety_pending = False
        if sa_needed > 0 and "sa_cme" in requirement_tags:
            sa_needed = max(sa_needed - float(best.credits or 0.0), 0.0)
        if pip_needed > 0 and "pip" in requirement_tags:
            pip_needed = max(pip_needed - 1, 0)

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
