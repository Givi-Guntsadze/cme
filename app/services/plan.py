from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlmodel import select

from ..models import (
    Activity,
    AssistantMessage,
    CompletedActivity,
    PlanItem,
    PlanRun,
    User,
    UserPolicy,
)
from ..planner import (
    build_plan,
    build_plan_with_policy,
    _is_patient_safety_activity,
    _is_sa_cme_activity,
    _is_pip_activity,
    is_eligible,
    pricing_context_for_user,
    requirements_gap_summary,
)
from ..requirements import REQUIREMENT_LABELS
from ..ingest import ingest_psychiatry_online_ai
from ..env import get_secret


DISCOVERY_IN_FLIGHT: set[int] = set()

LOGGER = logging.getLogger(__name__)


@dataclass
class PlanContext:
    pending_flags: Dict[str, bool]
    pending_values: Dict[str, Optional[float]]
    focus_messages: List[str]


def _compute_plan_context(session, user: User) -> PlanContext:
    overview = requirements_gap_summary(session, user)
    sa_needed = max(float(overview.get("sa_credits_needed") or 0.0), 0.0)
    pip_needed = max(int(overview.get("pip_needed") or 0), 0)
    safety_needed = bool(overview.get("needs_patient_safety"))

    pending_flags = {
        "patient_safety": safety_needed,
        "sa_cme": sa_needed > 0.0,
        "pip": pip_needed > 0,
    }
    pending_values: Dict[str, Optional[float]] = {
        "patient_safety": None,
        "sa_cme": sa_needed,
        "pip": float(pip_needed),
    }

    focus_messages: List[str] = []
    if pending_flags["patient_safety"]:
        focus_messages.append("Patient safety activity still required")
    if sa_needed > 0.0:
        focus_messages.append(f"{sa_needed:.1f} SA-CME credits outstanding")
    if pip_needed > 0:
        noun = "project" if pip_needed == 1 else "projects"
        focus_messages.append(f"{pip_needed} PIP {noun} remaining")

    return PlanContext(pending_flags, pending_values, focus_messages)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _source_label(source: Optional[str]) -> str:
    if not source:
        return "Curated"
    text = str(source).strip().lower()
    mapping = {
        "seed": "Curated",
        "web": "Web",
        "ai": "AI",
    }
    return mapping.get(text, text.replace("_", " ").title() or "Curated")


def load_policy_bundle(session, user: User) -> Dict[str, Any]:
    now = _utc_now()
    rows = list(
        session.exec(
            select(UserPolicy)
            .where(UserPolicy.user_id == user.id, UserPolicy.active.is_(True))
            .order_by(UserPolicy.created_at.desc())
        )
    )
    result: Dict[str, Any] = {"by_mode": {}, "default": None}
    changed = False

    for row in rows:
        if row.expires_at and row.expires_at <= now:
            row.active = False
            session.add(row)
            changed = True
            continue
        payload = row.payload or {}
        if row.mode == "default":
            if not result.get("default"):
                result["default"] = payload
        else:
            if row.mode not in result["by_mode"]:
                result["by_mode"][row.mode] = payload

    if changed:
        session.commit()

    cleaned: Dict[str, Any] = {}
    if result.get("by_mode"):
        cleaned["by_mode"] = result["by_mode"]
    if result.get("default"):
        cleaned["default"] = result["default"]
    return cleaned


def policy_for_mode(
    bundle: Optional[Dict[str, Any]], mode: str
) -> Optional[Dict[str, Any]]:
    if not bundle:
        return None
    result: Optional[Dict[str, Any]] = None
    by_mode = bundle.get("by_mode") if isinstance(bundle, dict) else None
    if isinstance(by_mode, dict):
        selected = by_mode.get(mode)
        if isinstance(selected, dict):
            result = selected
    if not result:
        default_policy = bundle.get("default") if isinstance(bundle, dict) else None
        if isinstance(default_policy, dict):
            result = default_policy
    if not result and isinstance(bundle, dict):
        fallback = {k: v for k, v in bundle.items() if k not in {"by_mode", "default"}}
        if fallback:
            result = fallback
    return result


def apply_policy_payloads(
    payloads: Iterable[str],
    user: User,
    session,
    invalidate: bool = True,
    record_message: bool = False,
) -> None:
    payload_list = [p for p in payloads if p]
    if not payload_list:
        return

    ttl = timedelta(days=1)
    now = _utc_now()
    entries: List[Tuple[str, Dict[str, Any]]] = []

    for payload in payload_list:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse POLICY payload: %s", payload)
            continue
        if not isinstance(data, dict):
            continue

        if "by_mode" in data and isinstance(data["by_mode"], dict):
            for mode, rules in data["by_mode"].items():
                if isinstance(rules, dict):
                    entries.append((mode.lower(), rules))
            continue

        if "plan_mode" in data and isinstance(data["plan_mode"], str):
            mode_key = data["plan_mode"].lower()
            rules = {k: v for k, v in data.items() if k not in {"plan_mode", "by_mode"}}
            entries.append((mode_key, rules))
            continue

        entries.append(("default", data))

    if not entries:
        return

    active_policies = list(
        session.exec(
            select(UserPolicy).where(
                UserPolicy.user_id == user.id, UserPolicy.active.is_(True)
            )
        )
    )
    for row in active_policies:
        row.active = False
        session.add(row)

    for mode, payload in entries:
        session.add(
            UserPolicy(
                user_id=user.id,
                mode=mode or "default",
                payload=payload or {},
                ttl_days=1,
                active=True,
                created_at=now,
                expires_at=now + ttl,
            )
        )

    summary = ", ".join(sorted({mode or "default" for mode, _ in entries}))
    if record_message:
        session.add(
            AssistantMessage(
                user_id=user.id,
                role="assistant",
                content=f"Policy updated for {summary}. Expires in 24h.",
            )
        )

    if invalidate:
        PlanManager.invalidate_user_plans(session, user.id, reason="policy_update")


def active_policy_status(session, user: User) -> List[Dict[str, Any]]:
    now = _utc_now()
    rows = list(
        session.exec(
            select(UserPolicy).where(
                UserPolicy.user_id == user.id, UserPolicy.active.is_(True)
            )
        )
    )
    status: List[Dict[str, Any]] = []
    changed = False
    for row in rows:
        if row.expires_at and row.expires_at <= now:
            row.active = False
            session.add(row)
            changed = True
            continue
        status.append(
            {
                "mode": row.mode or "default",
                "expires_at": row.expires_at,
                "payload": row.payload or {},
            }
        )
    if changed:
        session.commit()
    return status


def clear_policies(session, user: User) -> bool:
    rows = list(
        session.exec(
            select(UserPolicy).where(
                UserPolicy.user_id == user.id, UserPolicy.active.is_(True)
            )
        )
    )
    if not rows:
        return False
    for row in rows:
        row.active = False
        session.add(row)
    session.commit()
    PlanManager.invalidate_user_plans(session, user.id, reason="policy_clear")
    return True


class PlanManager:
    def __init__(self, session):
        self.session = session

    @staticmethod
    def invalidate_user_plans(session, user_id: int, reason: str = "update") -> None:
        runs = list(
            session.exec(
                select(PlanRun).where(
                    PlanRun.user_id == user_id, PlanRun.status == "active"
                )
            )
        )
        if not runs:
            return
        for run in runs:
            run.status = "stale"
            run.reason = reason
            session.add(run)

    def latest_run(self, user_id: int, mode: str) -> Optional[PlanRun]:
        stmt = (
            select(PlanRun)
            .where(PlanRun.user_id == user_id, PlanRun.mode == mode)
            .order_by(PlanRun.generated_at.desc())
        )
        return self.session.exec(stmt).first()

    def ensure_plan(
        self,
        user: User,
        mode: str,
        policy_bundle: Optional[Dict[str, Any]],
        force_refresh: bool = False,
        reason: Optional[str] = None,
    ) -> PlanRun:
        run = self.latest_run(user.id, mode)
        if run and not force_refresh and self._is_run_current(run, user):
            return run

        run = self._rebuild_run(
            user,
            mode,
            policy_bundle,
            reason,
            existing_run=run,
        )
        return run

    def _is_run_current(self, run: PlanRun, user: User) -> bool:
        if run.status != "active":
            return False
        if abs((run.remaining_credits or 0.0) - (user.remaining_credits or 0.0)) > 0.2:
            return False
        context = run.context or {}
        plan_context = _compute_plan_context(self.session, user)
        stored_flags = context.get("pending_flags") if isinstance(context, dict) else {}
        if stored_flags != plan_context.pending_flags:
            return False
        stored_values = (
            context.get("pending_values") if isinstance(context, dict) else {}
        )
        if stored_values != plan_context.pending_values:
            return False

        completed_ids = set(
            self.session.exec(
                select(CompletedActivity.activity_id).where(
                    CompletedActivity.user_id == user.id
                )
            )
        )
        if not completed_ids:
            return True
        item_stmt = select(PlanItem).where(PlanItem.plan_run_id == run.id)
        for item in self.session.exec(item_stmt):
            if item.activity_id in completed_ids:
                return False
        return True

    def _rebuild_run(
        self,
        user: User,
        mode: str,
        policy_bundle: Optional[Dict[str, Any]],
        reason: Optional[str],
        existing_run: Optional[PlanRun] = None,
    ) -> PlanRun:
        plan_context = _compute_plan_context(self.session, user)
        policy = policy_for_mode(policy_bundle, mode)

        committed_source_items: List[PlanItem] = []
        if existing_run:
            committed_source_items = list(
                self.session.exec(
                    select(PlanItem)
                    .where(
                        PlanItem.plan_run_id == existing_run.id,
                        PlanItem.committed.is_(True),
                    )
                    .order_by(PlanItem.position.asc())
                )
            )
        if not committed_source_items and mode != "balanced":
            anchor_run = self.latest_run(user.id, "balanced")
            if anchor_run:
                committed_source_items = list(
                    self.session.exec(
                        select(PlanItem)
                        .where(
                            PlanItem.plan_run_id == anchor_run.id,
                            PlanItem.committed.is_(True),
                        )
                        .order_by(PlanItem.position.asc())
                    )
                )

        committed_entries: List[Dict[str, Any]] = []
        committed_ids: set[int] = set()
        committed_credits = 0.0
        committed_cost = 0.0
        committed_days = 0

        for item in committed_source_items:
            activity = self.session.get(Activity, item.activity_id)
            if not activity or activity.id in committed_ids:
                continue
            committed_ids.add(activity.id)
            pricing = pricing_context_for_user(user, activity)
            cost_value = float(pricing.get("cost") or activity.cost_usd or 0.0)
            base_cost = pricing.get("base_cost")
            if base_cost is None:
                base_cost = activity.cost_usd
            deadline_obj = pricing.get("deadline")
            deadline_text = None
            deadline_urgency = None
            if deadline_obj:
                try:
                    days_left = pricing.get("deadline_days")
                    if isinstance(days_left, int):
                        if days_left == 0:
                            deadline_urgency = "today"
                        elif days_left == 1:
                            deadline_urgency = "tomorrow"
                        elif days_left <= 14:
                            deadline_urgency = f"in {days_left} days"
                    deadline_text = deadline_obj.strftime("%b %d, %Y")
                except Exception:
                    deadline_text = None

            requirement_tags = set(getattr(activity, "requirement_tags", []) or [])
            if _is_patient_safety_activity(activity):
                requirement_tags.add("patient_safety")
            if _is_sa_cme_activity(activity):
                requirement_tags.add("sa_cme")
            if _is_pip_activity(activity):
                requirement_tags.add("pip")

            eligible = is_eligible(user, activity)
            missing_profile_data = False
            if activity.eligible_institutions and not (user.affiliations or []):
                missing_profile_data = True
            if activity.eligible_groups and not getattr(user, "training_level", None):
                missing_profile_data = True
            if activity.membership_required and not (user.memberships or []):
                missing_profile_data = True
            if not eligible:
                eligibility_status = "ineligible"
            elif missing_profile_data or (
                activity.eligibility_text
                and not (
                    activity.eligible_institutions
                    or activity.eligible_groups
                    or activity.membership_required
                    or not getattr(activity, "open_to_public", True)
                )
            ):
                eligibility_status = "uncertain"
            else:
                eligibility_status = "eligible"

            committed_entries.append(
                {
                    "activity": activity,
                    "position": item.position,
                    "pricing": {
                        "cost": cost_value,
                        "base_cost": base_cost,
                        "deadline_text": deadline_text,
                        "deadline_urgency": deadline_urgency,
                        "price_label": pricing.get("label"),
                        "notes": pricing.get("notes"),
                        "hybrid_available": pricing.get("hybrid_available"),
                    },
                    "requirement_tags": requirement_tags,
                    "eligibility": eligibility_status,
                    "notes": pricing.get("notes"),
                }
            )

            committed_credits += float(activity.credits or 0.0)
            committed_cost += cost_value
            if activity.modality == "live":
                committed_days += activity.days_required

        committed_entries.sort(key=lambda entry: entry["position"])

        remaining_override = max(user.remaining_credits - committed_credits, 0.0)
        budget_override = max(float(user.budget_usd or 0.0) - committed_cost, 0.0)
        days_override = max(int(getattr(user, "days_off", 0) or 0) - committed_days, 0)

        if policy:
            recommended, rec_credits, rec_cost, rec_days = build_plan_with_policy(
                user,
                self.session,
                policy,
                mode=mode,
                remaining_override=remaining_override,
                budget_override=budget_override,
                days_override=days_override,
                exclude_ids=committed_ids,
            )
        else:
            recommended, rec_credits, rec_cost, rec_days = build_plan(
                user,
                self.session,
                mode=mode,
                remaining_override=remaining_override,
                budget_override=budget_override,
                days_override=days_override,
                exclude_ids=committed_ids,
            )

        total_credits = committed_credits + rec_credits
        total_cost = committed_cost + rec_cost
        days_used = committed_days + rec_days

        existing_runs = list(
            self.session.exec(
                select(PlanRun).where(
                    PlanRun.user_id == user.id,
                    PlanRun.mode == mode,
                    PlanRun.status == "active",
                )
            )
        )
        for prev in existing_runs:
            prev.status = "superseded"
            prev.reason = reason or "refresh"
            self.session.add(prev)

        run = PlanRun(
            user_id=user.id,
            mode=mode,
            generated_at=_utc_now(),
            status="active",
            reason=reason,
            total_credits=total_credits,
            total_cost=total_cost,
            days_used=days_used,
            remaining_credits=user.remaining_credits,
            requirement_focus=list(plan_context.focus_messages),
            context={
                "pending_flags": plan_context.pending_flags,
                "pending_values": plan_context.pending_values,
            },
        )
        self.session.add(run)
        self.session.flush()

        plan_items: List[PlanItem] = []
        position_counter = 0

        def _tag_payload_from_set(raw_tags: set[str]) -> List[Dict[str, Any]]:
            payload: List[Dict[str, Any]] = []
            for key, label in REQUIREMENT_LABELS.items():
                if key in raw_tags:
                    payload.append(
                        {
                            "key": key,
                            "label": label,
                            "pending": bool(plan_context.pending_flags.get(key)),
                            "value": plan_context.pending_values.get(key),
                        }
                    )
            return payload

        def _build_plan_item(
            *,
            activity: Activity,
            position: int,
            pricing: Dict[str, Any],
            requirement_tags: set[str],
            eligibility_status: str,
            notes: Optional[str],
            committed: bool,
        ) -> PlanItem:
            return PlanItem(
                user_id=user.id,
                activity_id=activity.id,
                plan_run_id=run.id,
                mode=mode,
                position=position,
                chosen=True,
                pricing_snapshot={
                    "cost": pricing.get("cost"),
                    "base_cost": pricing.get("base_cost"),
                    "deadline_text": pricing.get("deadline_text"),
                    "deadline_urgency": pricing.get("deadline_urgency"),
                    "price_label": pricing.get("price_label"),
                    "notes": notes,
                    "hybrid_available": pricing.get("hybrid_available"),
                },
                requirement_snapshot={
                    "tags": _tag_payload_from_set(requirement_tags),
                    "requirement_priority": any(
                        plan_context.pending_flags.get(tag) for tag in requirement_tags
                    ),
                },
                eligibility_status=eligibility_status,
                notes=notes,
                committed=committed,
            )

        for entry in committed_entries:
            activity = entry["activity"]
            pricing = entry["pricing"]
            plan_items.append(
                _build_plan_item(
                    activity=activity,
                    position=position_counter,
                    pricing=pricing,
                    requirement_tags=set(entry["requirement_tags"]),
                    eligibility_status=entry["eligibility"],
                    notes=entry.get("notes"),
                    committed=True,
                )
            )
            position_counter += 1

        for activity in recommended:
            pricing = getattr(activity, "_pricing_context", None)
            if not pricing:
                pricing = pricing_context_for_user(user, activity)
            cost_value = pricing.get("cost", activity.cost_usd)
            base_cost = pricing.get("base_cost")
            if base_cost is None:
                base_cost = activity.cost_usd
            deadline_obj = pricing.get("deadline")
            deadline_text = None
            deadline_urgency = None
            if deadline_obj:
                try:
                    days_left = pricing.get("deadline_days")
                    if isinstance(days_left, int):
                        if days_left == 0:
                            deadline_urgency = "today"
                        elif days_left == 1:
                            deadline_urgency = "tomorrow"
                        elif days_left <= 14:
                            deadline_urgency = f"in {days_left} days"
                    deadline_text = deadline_obj.strftime("%b %d, %Y")
                except Exception:
                    deadline_text = None

            pricing_payload = dict(pricing)
            pricing_payload.update(
                {
                    "cost": cost_value,
                    "base_cost": base_cost,
                    "deadline_text": deadline_text,
                    "deadline_urgency": deadline_urgency,
                    "price_label": pricing.get("label"),
                    "hybrid_available": pricing.get("hybrid_available"),
                }
            )

            requirement_tags = set(
                getattr(activity, "_requirement_tags", [])
                or getattr(activity, "requirement_tags", [])
            )
            if _is_patient_safety_activity(activity):
                requirement_tags.add("patient_safety")
            if _is_sa_cme_activity(activity):
                requirement_tags.add("sa_cme")
            if _is_pip_activity(activity):
                requirement_tags.add("pip")

            eligible = is_eligible(user, activity)
            missing_profile_data = False
            if activity.eligible_institutions and not (user.affiliations or []):
                missing_profile_data = True
            if activity.eligible_groups and not getattr(user, "training_level", None):
                missing_profile_data = True
            if activity.membership_required and not (user.memberships or []):
                missing_profile_data = True
            if not eligible:
                status = "ineligible"
            elif missing_profile_data or (
                activity.eligibility_text
                and not (
                    activity.eligible_institutions
                    or activity.eligible_groups
                    or activity.membership_required
                    or not getattr(activity, "open_to_public", True)
                )
            ):
                status = "uncertain"
            else:
                status = "eligible"

            plan_items.append(
                _build_plan_item(
                    activity=activity,
                    position=position_counter,
                    pricing=pricing_payload,
                    requirement_tags=requirement_tags,
                    eligibility_status=status,
                    notes=pricing.get("notes"),
                    committed=False,
                )
            )
            position_counter += 1

        if plan_items:
            self.session.add_all(plan_items)

        self.session.commit()
        return run

    def serialize_plan(
        self, run: PlanRun, user: User
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
        plan: List[Dict[str, Any]] = []
        stmt = (
            select(PlanItem)
            .where(PlanItem.plan_run_id == run.id)
            .order_by(PlanItem.position.asc())
        )
        items = list(self.session.exec(stmt))
        if not items:
            return (
                [],
                None,
                {
                    "pending": run.context.get("pending_flags") if run.context else {},
                    "values": run.context.get("pending_values") if run.context else {},
                    "messages": list(run.requirement_focus or []),
                },
            )

        activity_ids = [item.activity_id for item in items]
        activity_lookup = {
            activity.id: activity
            for activity in self.session.exec(
                select(Activity).where(Activity.id.in_(activity_ids))
            )
        }

        for item in items:
            activity = activity_lookup.get(item.activity_id)
            if not activity:
                LOGGER.warning(
                    "Plan item %s missing activity %s", item.id, item.activity_id
                )
                continue
            pricing = pricing_context_for_user(user, activity)
            if not pricing.get("cost") and item.pricing_snapshot:
                pricing.update(
                    {k: v for k, v in item.pricing_snapshot.items() if k not in pricing}
                )

            cost_value = pricing.get("cost")
            if cost_value is None and item.pricing_snapshot:
                cost_value = item.pricing_snapshot.get("cost")
            if cost_value is None:
                cost_value = activity.cost_usd
            if cost_value is None:
                cost_display = "TBD"
            else:
                cost_display = f"${cost_value:,.0f}"

            base_cost = pricing.get("base_cost")
            if base_cost is None and item.pricing_snapshot:
                base_cost = item.pricing_snapshot.get("base_cost")
            if base_cost is None:
                base_cost = activity.cost_usd

            deadline_obj = pricing.get("deadline")
            deadline_text = None
            deadline_urgency = None
            if isinstance(deadline_obj, date):
                days_left = pricing.get("deadline_days")
                if isinstance(days_left, int):
                    if days_left == 0:
                        deadline_urgency = "today"
                    elif days_left == 1:
                        deadline_urgency = "tomorrow"
                    elif days_left <= 14:
                        deadline_urgency = f"in {days_left} days"
                deadline_text = deadline_obj.strftime("%b %d, %Y")
            else:
                if item.pricing_snapshot:
                    deadline_text = item.pricing_snapshot.get("deadline_text")
                    deadline_urgency = item.pricing_snapshot.get("deadline_urgency")

            tag_info = []
            if item.requirement_snapshot and isinstance(
                item.requirement_snapshot, dict
            ):
                tag_info = item.requirement_snapshot.get("tags") or []

            plan.append(
                {
                    "activity_id": item.activity_id,
                    "title": activity.title,
                    "provider": activity.provider,
                    "credits": activity.credits,
                    "cost": cost_value,
                    "cost_display": cost_display,
                    "price_label": pricing.get("label")
                    or (item.pricing_snapshot or {}).get("price_label"),
                    "base_cost": base_cost,
                    "deadline_text": deadline_text,
                    "deadline_urgency": deadline_urgency,
                    "pricing_notes": pricing.get("notes")
                    or (item.pricing_snapshot or {}).get("notes"),
                    "modality": activity.modality,
                    "city": activity.city,
                    "url": activity.url,
                    "summary": activity.summary,
                    "source": activity.source,
                    "source_label": _source_label(getattr(activity, "source", None)),
                    "hybrid_available": pricing.get("hybrid_available")
                    or (item.pricing_snapshot or {}).get("hybrid_available")
                    or getattr(activity, "hybrid_available", False),
                    "topic": getattr(activity, "_topic_tag", None),
                    "eligibility_status": item.eligibility_status,
                    "eligibility_text": activity.eligibility_text,
                    "requirement_tags": tag_info,
                    "requirement_priority": bool(
                        (item.requirement_snapshot or {}).get("requirement_priority")
                    ),
                    "committed": bool(item.committed),
                }
            )

        plan_summary = {
            "item_count": len(plan),
            "total_credits": run.total_credits,
            "total_cost": run.total_cost,
            "cost_display": (
                f"${run.total_cost:,.2f}" if run.total_cost is not None else "TBD"
            ),
            "days_used": run.days_used,
            "requirement_focus": list(run.requirement_focus or []),
            "committed_count": sum(1 for item in items if item.committed),
        }

        plan_requirements = {
            "pending": (run.context or {}).get("pending_flags", {}),
            "values": (run.context or {}).get("pending_values", {}),
            "messages": list(run.requirement_focus or []),
        }

        return plan, plan_summary, plan_requirements

    def summarize_modes(
        self,
        user: User,
        policy_bundle: Optional[Dict[str, Any]],
        force_refresh: bool = False,
    ) -> Tuple[str, int, int]:
        summaries = []
        counts = {}
        for mode in ("balanced", "cheapest"):
            run = self.ensure_plan(
                user,
                mode,
                policy_bundle,
                force_refresh=force_refresh,
                reason="summary_refresh" if force_refresh else None,
            )
            plan_items, _, _ = self.serialize_plan(run, user)
            counts[mode] = len(plan_items)
            if not plan_items:
                summaries.append(f"{mode.title()}: no activities available")
                continue
            top_details = []
            for entry in plan_items[:3]:
                title = entry.get("title", "")
                credits = entry.get("credits", 0.0)
                item = f"{title[:60]} ({credits:.1f} cr)"
                elig_bits = []
                status = entry.get("eligibility_status")
                if status == "ineligible":
                    elig_bits.append("restricted access")
                elif entry.get("eligibility_text"):
                    snippet = entry["eligibility_text"][:60]
                    if len(entry["eligibility_text"]) > 60:
                        snippet += "…"
                    elig_bits.append(snippet)
                if elig_bits:
                    item += f" [elig: {'; '.join(elig_bits)}]"
                top_details.append(item)
            total = sum(entry.get("credits", 0.0) for entry in plan_items)
            summaries.append(
                f"{mode.title()}: {total:.1f} cr — {', '.join(top_details)}"
            )
        summary_text = "\n".join(["Plan refresh:", *summaries])
        return summary_text, counts.get("balanced", 0), counts.get("cheapest", 0)

    def _auto_discover_for_user(self, user: User) -> bool:
        specialty = (user.specialty or "").strip().lower()
        if specialty and "psychi" not in specialty:
            LOGGER.info(
                "Auto discovery skipped: no ingester for specialty '%s'", specialty
            )
            return False

        if not (get_secret("GOOGLE_API_KEY") and get_secret("GOOGLE_CSE_ID")):
            LOGGER.info("Auto discovery skipped: Google CSE not configured")
            return False

        if not user.id:
            LOGGER.info("Auto discovery skipped: user has no id yet")
            return False

        if user.id in DISCOVERY_IN_FLIGHT:
            LOGGER.debug("Auto discovery already running for user %s", user.id)
            return False

        DISCOVERY_IN_FLIGHT.add(user.id)

        def _run_ingest(user_id: int) -> None:
            try:
                LOGGER.info(
                    "Auto discovery triggered for user %s (%s)",
                    user_id,
                    specialty or "psychiatry",
                )
                asyncio.run(ingest_psychiatry_online_ai(count=20))
            except Exception:
                LOGGER.exception("Auto discovery ingest failed for user %s", user_id)
            finally:
                DISCOVERY_IN_FLIGHT.discard(user_id)

        threading.Thread(target=_run_ingest, args=(user.id,), daemon=True).start()
        return True
