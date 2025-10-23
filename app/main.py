from __future__ import annotations
import os
import json
import logging
import re
import asyncio
import time
from difflib import SequenceMatcher
from textwrap import dedent
from types import SimpleNamespace
from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlmodel import select

from .db import create_db_and_tables, get_session
from .models import (
    User,
    Claim,
    AssistantMessage,
    Activity,
    PlanItem,
    CompletedActivity,
)
from .requirements import (
    load_abpn_psychiatry_requirements,
    validate_against_requirements,
    validate_full_cc,
)
from .parser import parse_message
from .ingest import (
    ingest_psychiatry_online_ai,
    safe_json_loads,
    discover_activity_by_title,
)
from openai import OpenAI
from .services.plan import (
    PlanManager,
    active_policy_status,
    apply_policy_payloads,
    clear_policies,
    load_policy_bundle,
    policy_for_mode,
)
from .prompt import build_system_prompt
from .planner import pricing_context_for_user, is_eligible
from .openai_helpers import call_responses, response_text
from .env import get_secret


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_db_and_tables()
    yield
    # Shutdown: nothing for now


app = FastAPI(title="CME/MOC POC", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
DEFAULT_ASSISTANT_MODEL = get_secret("OPENAI_ASSISTANT_MODEL") or "gpt-4o-mini"

# Matches a line like:  PATCH: {"budget_usd": 200, "allow_live": false}
PATCH_RE = re.compile(r"PATCH:\s*(\{.*\})", re.DOTALL)


def _safe_json(obj_text: str):
    """Parse compact JSON; tolerate accidental backticks/code fences."""
    if not obj_text:
        return None
    t = obj_text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`")
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def _normalize_stage(value: object | None) -> str | None:
    if value is None:
        return None
    stage = str(value).strip().lower()
    if not stage:
        return None
    aliases = {
        "early career": "early_career",
        "early-career": "early_career",
        "early_career": "early_career",
        "early": "early_career",
        "established": "standard",
        "attending": "standard",
        "standard": "standard",
        "full": "standard",
        "resident": "resident",
        "fellow": "resident",
        "trainee": "resident",
    }
    return aliases.get(stage, stage.replace(" ", "_"))


def _parse_membership_input(raw: object | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = []
        for item in raw:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(raw)
    chunks = re.split(r"[,;/\n]+", text)
    members: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if cleaned:
            members.append(cleaned)
    return members


AFFIRMATIVE_REPLIES = {
    "yes",
    "y",
    "yep",
    "yeah",
    "sure",
    "ok",
    "okay",
    "please do",
    "do it",
    "sounds good",
    "lets do it",
    "let's do it",
    "absolutely",
    "definitely",
    "sure thing",
    "yes please",
    "please",
}

NEGATIVE_REPLIES = {
    "no",
    "nah",
    "nope",
    "not now",
    "not today",
    "dont",
    "don't",
    "stop",
    "cancel",
    "no thanks",
    "no thank you",
}

REMOVAL_ACTION_KEYWORDS = {
    "delete",
    "remove",
    "clear",
    "erase",
    "undo",
    "drop",
    "reset",
    "wipe",
    "strip",
    "subtract",
    "take off",
    "take out",
    "roll back",
    "rollback",
    "correct",
    "fix",
}

REMOVE_BLOCK_RE = re.compile(r"\bremove\b\s+([^.;!?]+)", re.I)
REMOVE_SPLIT_RE = re.compile(
    r"\s*(?:,|and|&|plus|as well as|along with|/|\band\b)\s*", re.I
)
GENERIC_TAIL_RE = re.compile(
    r"\b(activity|event|course|conference|session|module|plan|details?)\b$", re.I
)
LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.I)
DETAIL_TAIL_RE = re.compile(r"\b(details?|info|information)\b$", re.I)
TITLE_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
TITLE_STOPWORDS = {
    "the",
    "a",
    "an",
    "with",
    "for",
    "and",
    "event",
    "conference",
    "meeting",
    "cme",
    "annual",
    "update",
    "institute",
    "activity",
    "course",
    "plan",
    "session",
    "workshop",
    "hybrid",
    "primary",
}
ADD_PATTERNS = [
    re.compile(
        (
            r"\b(?:instead\s+)?(?:i\s+)?"
            r"(?:want|would like|plan|planning|going)\s+to\s+"
            r"(?:add(?:ing)?|include|attend(?:ing)?|go(?:ing)?\s+to|register(?:ing)?\s+for)\s+"
            r"(?P<title>[^.;!?]+)"
        ),
        re.I,
    ),
    re.compile(
        (
            r"\b(?:add(?:ing)?|include|slot\s+in|schedule|book)\s+(?:in\s+)?"
            r"(?P<title>[^.;!?]+)"
        ),
        re.I,
    ),
    re.compile(
        (
            r"\b(?:attend(?:ing)?|register(?:ing)?\s+for|go(?:ing)?\s+to|join(?:ing)?)\s+"
            r"(?P<title>[^.;!?]+)"
        ),
        re.I,
    ),
]


def _clean_title_fragment(fragment: str) -> str:
    cleaned = fragment.strip(" \t\r\n.,;:!\"'()[]{}")
    cleaned = LEADING_ARTICLE_RE.sub("", cleaned)
    trimmed = cleaned
    # remove trailing generic descriptors
    for _ in range(3):
        new_trim = GENERIC_TAIL_RE.sub("", trimmed).strip(" \t\r\n.,;:!\"'")
        if new_trim == trimmed:
            break
        trimmed = new_trim
    trimmed = DETAIL_TAIL_RE.sub("", trimmed).strip(" \t\r\n.,;:!\"'")
    return trimmed


def _extract_remove_targets(text: str) -> list[str]:
    if not text:
        return []
    results: list[str] = []
    for match in REMOVE_BLOCK_RE.finditer(text):
        block = match.group(1)
        if not block:
            continue
        block = re.split(r"\b(?:but|instead|however|so|then)\b", block, maxsplit=1)[0]
        parts = REMOVE_SPLIT_RE.split(block)
        for part in parts:
            cleaned = _clean_title_fragment(part)
            if not cleaned or cleaned.lower() in {"it", "them", "those"}:
                continue
            results.append(cleaned)
    # Deduplicate while preserving order
    deduped: list[str] = []
    seen = set()
    for item in results:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_add_targets(text: str) -> list[str]:
    if not text:
        return []
    results: list[str] = []
    for pattern in ADD_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group("title")
            if not raw:
                continue
            candidate = re.split(
                r"\b(?:but|instead|however|so|then|and then)\b",
                raw,
                maxsplit=1,
            )[0]
            cleaned = _clean_title_fragment(candidate)
            if not cleaned or cleaned.lower() in {"it", "them", "those"}:
                continue
            normalized = cleaned.lower()
            if not any(
                keyword in normalized
                for keyword in (
                    "conference",
                    "meeting",
                    "cme",
                    "summit",
                    "symposium",
                    "course",
                    "event",
                    "institute",
                    "update",
                )
            ):
                if len(cleaned.split()) < 2:
                    continue
            results.append(cleaned)
    deduped: list[str] = []
    seen = set()
    for item in results:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _title_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in TITLE_TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower()
        if not token:
            continue
        if token in TITLE_STOPWORDS and len(token) < 5:
            continue
        if token.isdigit() and len(token) < 4:
            continue
        tokens.add(token)
    return tokens


def _is_candidate_match(candidate_title: str, query_tokens: set[str]) -> bool:
    if not candidate_title or not query_tokens:
        return False
    candidate_tokens = _title_tokens(candidate_title)
    if not candidate_tokens:
        return False
    overlap = query_tokens & candidate_tokens
    if not overlap:
        return False
    if len(query_tokens) <= 3:
        return overlap == query_tokens
    overlap_threshold = max(2, len(query_tokens) // 2)
    return len(overlap) >= overlap_threshold


def _normalize_response_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _is_affirmative(text: str) -> bool:
    if not text:
        return False
    cleaned = _normalize_response_text(text)
    if not cleaned:
        return False
    if cleaned in AFFIRMATIVE_REPLIES:
        return True
    first_word = cleaned.split(" ")[0]
    return first_word in AFFIRMATIVE_REPLIES


def _is_negative(text: str) -> bool:
    if not text:
        return False
    cleaned = _normalize_response_text(text)
    if not cleaned:
        return False
    if cleaned in NEGATIVE_REPLIES:
        return True
    first_word = cleaned.split(" ")[0]
    return first_word in NEGATIVE_REPLIES


def _match_score(search: str, title: str) -> float:
    title_lower = (title or "").lower()
    if not title_lower:
        return 0.0
    ratio = SequenceMatcher(None, search, title_lower).ratio()
    if search in title_lower:
        ratio += 0.75
    if title_lower.startswith(search):
        ratio += 0.25
    tokens = [t for t in search.split(" ") if t]
    if tokens:
        hits = sum(1 for token in tokens if token in title_lower)
        ratio += 0.1 * hits
    return ratio


MIN_MATCH_SCORE = 0.25


def _activity_cost_and_days(
    user: User, activity: Activity, snapshot: dict | None = None
) -> tuple[float, int, dict]:
    pricing_context = pricing_context_for_user(user, activity)
    combined = dict(snapshot or {})
    combined.update({k: v for k, v in pricing_context.items() if v is not None})
    cost_value = combined.get("cost")
    if cost_value is None:
        cost_value = combined.get("base_cost")
    if cost_value is None:
        cost_value = activity.cost_usd
    cost = float(cost_value or 0.0)
    days_needed = (
        activity.days_required if (activity.modality or "").lower() == "live" else 0
    )
    return cost, days_needed, combined


def find_activity_by_title(session, user: User, text: str) -> Activity | None:
    search = (text or "").strip().lower()
    if not search:
        return None

    plan_manager = PlanManager(session)
    policy_bundle = load_policy_bundle(session, user)
    run = plan_manager.ensure_plan(user, "balanced", policy_bundle)

    candidates: list[tuple[int, float, int, Activity]] = []
    seen_ids: set[int] = set()

    if run:
        plan_items = list(
            session.exec(
                select(PlanItem)
                .where(PlanItem.plan_run_id == run.id)
                .order_by(PlanItem.position.asc())
            )
        )
        for item in plan_items:
            activity = session.get(Activity, item.activity_id)
            if not activity or not activity.title:
                continue
            score = _match_score(search, activity.title)
            if score < MIN_MATCH_SCORE:
                continue
            priority = 0 if item.committed else 1
            candidates.append(
                (
                    priority,
                    -score,
                    item.position,
                    activity.id or 0,
                    activity,
                )
            )
            seen_ids.add(activity.id)

    catalog_stmt = select(Activity)
    for activity in session.exec(catalog_stmt):
        if activity.id in seen_ids:
            continue
        if not activity.title:
            continue
        score = _match_score(search, activity.title)
        if score < MIN_MATCH_SCORE:
            continue
        candidates.append(
            (
                2,
                -score,
                len(activity.title or ""),
                activity.id or 0,
                activity,
            )
        )

    if not candidates:
        return None

    candidates.sort()
    best_activity = candidates[0][4]
    best_score = -candidates[0][1]
    if best_score < MIN_MATCH_SCORE:
        return None
    return best_activity


def propose_substitute(
    session, user: User, removed: Activity | None
) -> Activity | None:
    plan_manager = PlanManager(session)
    policy_bundle = load_policy_bundle(session, user)
    run = plan_manager.ensure_plan(
        user,
        "balanced",
        policy_bundle,
        force_refresh=True,
        reason="substitute_search",
    )

    plan_items = list(
        session.exec(
            select(PlanItem)
            .where(PlanItem.plan_run_id == run.id)
            .order_by(PlanItem.position.asc())
        )
    )

    committed_items = [item for item in plan_items if item.committed]
    committed_ids = {item.activity_id for item in committed_items}
    if removed and removed.id:
        committed_ids.add(removed.id)

    used_budget = 0.0
    used_days = 0
    for item in committed_items:
        activity = session.get(Activity, item.activity_id)
        if not activity:
            continue
        cost, days_needed, _ = _activity_cost_and_days(
            user, activity, item.pricing_snapshot
        )
        used_budget += cost
        used_days += days_needed

    available_budget = max(float(user.budget_usd or 0.0) - used_budget, 0.0)
    available_days = max(int(getattr(user, "days_off", 0) or 0) - used_days, 0)

    best_choice: tuple[tuple[float, float, int], Activity] | None = None

    for item in plan_items:
        if item.committed:
            continue
        if removed and item.activity_id == removed.id:
            continue
        activity = session.get(Activity, item.activity_id)
        if not activity:
            continue
        if not user.allow_live and (activity.modality or "").lower() == "live":
            continue
        cost, days_needed, pricing = _activity_cost_and_days(
            user, activity, item.pricing_snapshot
        )
        if cost > available_budget + 1e-6:
            continue
        if days_needed > available_days:
            continue
        if not is_eligible(user, activity):
            continue
        removed_credits = float(removed.credits or 0.0) if removed else 0.0
        credits = float(activity.credits or 0.0)
        delta = abs(credits - removed_credits) if removed else credits
        cost_per_credit = cost / credits if credits else float("inf")
        key = (delta, cost_per_credit, item.position)
        if best_choice is None or key < best_choice[0]:
            best_choice = (key, activity)

    if best_choice:
        return best_choice[1]

    return None


def _get_latest_internal_message(
    session, user_id: int, prefix: str
) -> AssistantMessage | None:
    stmt = (
        select(AssistantMessage)
        .where(AssistantMessage.user_id == user_id)
        .order_by(AssistantMessage.created_at.desc())
    )
    for message in session.exec(stmt):
        content = (message.content or "").strip()
        if content.startswith(prefix):
            return message
    return None


def _pop_internal_message(
    session, user_id: int, prefix: str
) -> AssistantMessage | None:
    message = _get_latest_internal_message(session, user_id, prefix)
    if message:
        session.delete(message)
        session.flush()
    return message


def _apply_patch_to_user(patch: dict, session, user: User) -> list[str]:
    """
    Apply allowed fields to User.
    """

    changed: list[str] = []
    if not isinstance(patch, dict):
        return changed

    allowed = {
        "budget_usd": float,
        "days_off": int,
        "allow_live": bool,
        "city": str,
        "specialty": str,
        "target_credits": float,
        "professional_stage": str,
        "residency_completion_year": int,
    }

    for key, caster in allowed.items():
        if key not in patch:
            continue
        try:
            if caster in (bool, str):
                value = patch[key]
                if caster is str and value is not None:
                    value = str(value)
            else:
                value = caster(patch[key])
        except Exception:
            continue

        if getattr(user, key, None) != value:
            if key == "professional_stage":
                normalized = _normalize_stage(value)
                if normalized != getattr(user, key, None):
                    setattr(user, key, normalized)
                    changed.append("professional_stage")
                continue
            setattr(user, key, value)
            changed.append(key)

    if "memberships" in patch:
        parsed_memberships = _parse_membership_input(patch.get("memberships"))
        if parsed_memberships != (user.memberships or []):
            user.memberships = parsed_memberships
            changed.append("memberships")

    if changed:
        session.add(user)
        session.flush()
        PlanManager.invalidate_user_plans(session, user.id, reason="preferences_update")

    return changed


def _maybe_set_remaining_from_text(user: User, user_text: str, session) -> str | None:
    """Handle natural language adjustments like 'set remaining to X'."""

    if not user_text:
        return None

    match = re.search(r"(?:remaining|left)\s+(\d+(?:\.\d+)?)", user_text.lower())
    if not match:
        return None

    try:
        desired_remaining = float(match.group(1))
    except Exception:
        return None

    total_claimed = 0.0
    for claim in session.exec(select(Claim).where(Claim.user_id == user.id)):
        total_claimed += float(claim.credits or 0.0)

    current_remaining = max(float(user.target_credits or 0.0) - total_claimed, 0.0)
    delta = current_remaining - desired_remaining
    if delta <= 0:
        return "No change needed (already at or below that remaining)."

    adjustment = Claim(
        user_id=user.id,
        credits=delta,
        topic="adjustment",
        date=date.today(),
        source_text=f"Assistant adjustment to set remaining to {desired_remaining:.1f}",
    )
    session.add(adjustment)
    session.flush()
    _recalculate_remaining(user, session)
    PlanManager.invalidate_user_plans(session, user.id, reason="remaining_adjusted")
    session.commit()
    return (
        f"Logged an adjustment of {delta:.1f} credits to set remaining ~"
        f"{desired_remaining:.1f}."
    )


def _recalculate_remaining(user: User, session) -> None:
    total_claimed = 0.0
    for claim in session.exec(select(Claim).where(Claim.user_id == user.id)):
        total_claimed += float(claim.credits or 0.0)
    user.remaining_credits = max(float(user.target_credits or 0.0) - total_claimed, 0.0)
    session.add(user)
    session.flush()


def _maybe_remove_claim_from_text(
    user: User, user_text: str, session
) -> tuple[str | None, bool]:
    text = (user_text or "").strip()
    if not text:
        return (None, False)
    lowered = text.lower()
    if not any(keyword in lowered for keyword in REMOVAL_ACTION_KEYWORDS):
        return (None, False)

    claims = list(session.exec(select(Claim).where(Claim.user_id == user.id)))
    if not claims:
        if "credit" in lowered or "log" in lowered:
            return ("No logged credits to remove.", False)
        return (None, False)

    amount_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:credit|credits|cr|hrs|hours)", lowered
    )
    removed: list[Claim] = []
    remaining_pool = sorted(
        claims,
        key=lambda c: (
            c.date or date.today(),
            c.id or 0,
        ),
        reverse=True,
    )

    if amount_matches:
        for amt_text in amount_matches:
            try:
                target = float(amt_text)
            except ValueError:
                continue
            candidate = None
            for claim in remaining_pool:
                if abs(float(claim.credits or 0.0) - target) < 0.01:
                    candidate = claim
                    break
            if candidate:
                session.delete(candidate)
                removed.append(candidate)
                remaining_pool.remove(candidate)
        if removed:
            session.flush()
            _recalculate_remaining(user, session)
            PlanManager.invalidate_user_plans(session, user.id, reason="claim_removed")
            session.commit()
            total_removed = sum(float(c.credits or 0.0) for c in removed)
            entry_label = "entries" if len(removed) != 1 else "entry"
            return (
                f"Removed {len(removed)} logged credit {entry_label} ({total_removed:.1f} credits).",
                True,
            )
        return ("I couldn't find a logged credit entry with that amount.", False)

    if "activity log" in lowered or "all credits" in lowered or "entire log" in lowered:
        for claim in claims:
            session.delete(claim)
        session.flush()
        _recalculate_remaining(user, session)
        PlanManager.invalidate_user_plans(session, user.id, reason="claim_removed")
        session.commit()
        count = len(claims)
        entry_label = "entries" if count != 1 else "entry"
        return (f"Cleared {count} logged credit {entry_label}.", True)

    return (None, False)


def _maybe_log_claim_from_text(user: User, user_text: str, session) -> str | None:
    credits, topic, claim_date = parse_message(user_text)
    if credits <= 0:
        return None
    if not topic:
        topic = "general"
    if not claim_date:
        claim_date = date.today()
    claim = Claim(
        user_id=user.id,
        credits=float(credits),
        topic=topic,
        date=claim_date,
        source_text=user_text,
    )
    session.add(claim)
    session.flush()
    _recalculate_remaining(user, session)
    PlanManager.invalidate_user_plans(session, user.id, reason="claim_logged")
    session.commit()
    return f"Logged {credits:.1f} credits for {topic}."


def _maybe_complete_activity_from_text(
    user: User, user_text: str, session
) -> str | None:
    text = (user_text or "").lower().strip()
    if not text:
        return None
    keywords = ["done", "completed", "finished"]
    if not any(k in text for k in keywords):
        return None
    tail = text
    for k in keywords:
        if k in tail:
            tail = tail.split(k, 1)[1]
    tail = tail.replace("with", " ").strip()
    if not tail:
        return None
    activities = list(session.exec(select(Activity)))
    match = None
    for activity in activities:
        title = (activity.title or "").lower()
        if not title:
            continue
        if tail in title or title in tail:
            match = activity
            break
    if not match:
        return None

    claim = Claim(
        user_id=user.id,
        credits=float(match.credits or 0.0),
        topic=(match.title or "activity"),
        date=date.today(),
        source_text=user_text,
    )
    session.add(claim)
    existing = session.exec(
        select(CompletedActivity).where(
            CompletedActivity.user_id == user.id,
            CompletedActivity.activity_id == match.id,
        )
    ).first()
    if not existing:
        completed_entry = CompletedActivity(user_id=user.id, activity_id=match.id)
        session.add(completed_entry)
    session.flush()
    _recalculate_remaining(user, session)
    PlanManager.invalidate_user_plans(session, user.id, reason="activity_completed")
    session.commit()
    return f"Marked '{match.title}' completed (+{match.credits:.1f} credits)."


def _state_snapshot(session, user: User) -> dict[str, object]:
    total_claimed = sum(
        c.credits for c in session.exec(select(Claim).where(Claim.user_id == user.id))
    )
    remaining = max(user.target_credits - total_claimed, 0.0)
    user.remaining_credits = remaining
    session.add(user)

    plan_manager = PlanManager(session)
    policy_bundle = load_policy_bundle(session, user)
    run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
    plan_entries, plan_summary, _ = plan_manager.serialize_plan(run, user)
    top = []
    for entry in plan_entries[:6]:
        top.append(
            {
                "title": entry.get("title"),
                "provider": entry.get("provider"),
                "credits": entry.get("credits"),
                "cost": entry.get("cost"),
                "base_cost": entry.get("base_cost"),
                "price_label": entry.get("price_label"),
                "deadline": entry.get("deadline_text"),
                "modality": entry.get("modality"),
                "city": entry.get("city"),
                "hybrid_available": entry.get("hybrid_available"),
                "committed": entry.get("committed", False),
            }
        )

    snapshot_user = {
        "budget_usd": user.budget_usd,
        "days_off": user.days_off,
        "allow_live": user.allow_live,
        "city": user.city,
        "specialty": user.specialty,
        "target_credits": user.target_credits,
        "affiliations": list(getattr(user, "affiliations", []) or []),
        "memberships": list(getattr(user, "memberships", []) or []),
        "training_level": getattr(user, "training_level", None),
        "professional_stage": getattr(user, "professional_stage", None),
        "residency_completion_year": getattr(user, "residency_completion_year", None),
    }
    req_data = load_abpn_psychiatry_requirements()
    claims = list(session.exec(select(Claim).where(Claim.user_id == user.id)))
    validation = validate_against_requirements(user, claims, req_data)
    summary = validation.get("summary", {})
    summary_line = (
        f"{summary.get('earned_total', 0.0):.1f} earned / "
        f"{summary.get('target_total', 0)} target (remaining {summary.get('remaining_total', 0.0):.1f})"
    )
    return {
        "remaining": remaining,
        "user": snapshot_user,
        "top_plan": top,
        "plan_summary": plan_summary or {},
        "requirements": {
            "summary": validation.get("summary", {}),
            "checks": validation.get("checks", []),
            "summary_line": summary_line,
        },
    }


def _apply_patch_if_present(
    payloads: list[str], user: User, session
) -> list[AssistantMessage]:
    if not payloads:
        return []
    applied: list[str] = []
    patch_data: dict[str, object] = {}
    for payload in payloads:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logging.warning("Failed to parse PATCH payload: %s", payload)
            continue
        if isinstance(data, dict):
            patch_data.update(data)

    if not patch_data:
        return []

    def _maybe_float(value: object) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _maybe_int(value: object) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    if "budget_usd" in patch_data:
        new_val = _maybe_float(patch_data.get("budget_usd"))
        if new_val is not None and new_val != user.budget_usd:
            user.budget_usd = new_val
            applied.append(f"budget ${new_val:,.0f}")

    if "days_off" in patch_data:
        new_val = _maybe_int(patch_data.get("days_off"))
        if new_val is not None and new_val != user.days_off:
            user.days_off = new_val
            applied.append(f"days_off {new_val}")

    if "allow_live" in patch_data:
        new_val = patch_data.get("allow_live")
        if isinstance(new_val, str):
            new_val = new_val.lower() in {"1", "true", "yes", "y"}
        if isinstance(new_val, bool) and new_val != user.allow_live:
            user.allow_live = new_val
            applied.append(f"allow_live {new_val}")

    if "city" in patch_data:
        new_val = patch_data.get("city") or None
        if new_val != user.city:
            user.city = new_val
            applied.append(f"city {new_val or 'cleared'}")

    if "specialty" in patch_data:
        new_val = patch_data.get("specialty") or user.specialty
        if new_val != user.specialty:
            user.specialty = new_val
            applied.append(f"specialty {new_val}")

    if "target_credits" in patch_data:
        new_val = _maybe_float(patch_data.get("target_credits"))
        if new_val is not None and new_val != user.target_credits:
            user.target_credits = new_val
            applied.append(f"target_credits {new_val}")

    if "professional_stage" in patch_data:
        normalized = _normalize_stage(patch_data.get("professional_stage"))
        if normalized != getattr(user, "professional_stage", None):
            user.professional_stage = normalized
            applied.append(f"professional_stage {normalized or 'cleared'}")

    if "residency_completion_year" in patch_data:
        new_year = _maybe_int(patch_data.get("residency_completion_year"))
        if new_year != getattr(user, "residency_completion_year", None):
            user.residency_completion_year = new_year
            applied.append(
                "residency_completion_year "
                + (str(new_year) if new_year else "cleared")
            )

    if "memberships" in patch_data:
        parsed = _parse_membership_input(patch_data.get("memberships"))
        if parsed != (getattr(user, "memberships", []) or []):
            user.memberships = parsed
            applied.append(
                "memberships " + (", ".join(parsed) if parsed else "cleared")
            )

    if not applied:
        return []

    total_claimed = 0.0
    if user.id is not None:
        total_claimed = sum(
            c.credits
            for c in session.exec(select(Claim).where(Claim.user_id == user.id))
        )
    user.remaining_credits = max(user.target_credits - total_claimed, 0.0)

    confirmation = AssistantMessage(
        user_id=user.id,
        role="assistant",
        content="Preferences updated: " + ", ".join(applied),
    )
    session.add(user)
    session.add(confirmation)
    session.flush()
    PlanManager.invalidate_user_plans(session, user.id, reason="preferences_update")
    return [confirmation]


def _derive_controls(
    user_text: str, snapshot: dict[str, object]
) -> tuple[list[str], list[str]]:
    if not openai_client:
        return ([], [])

    control_system = dedent(
        """
        You generate structured updates for a CME planning assistant.
        Respond with STRICT JSON using this schema:
        {
          "patch": null | {
            "budget_usd"?: number,
            "days_off"?: integer,
            "allow_live"?: boolean,
            "city"?: string | null,
            "specialty"?: string,
            "target_credits"?: number,
            "professional_stage"?: string | null,
            "residency_completion_year"?: integer | null,
            "memberships"?: array<string> | string | null
          },
          "policy": null | {
            "default"?: OBJECT,
            "by_mode"?: {
              "cheapest"?: OBJECT,
              "balanced"?: OBJECT
            }
          }
        }
        Each OBJECT for policies may include: remove_titles (array of exact titles), avoid_terms (array of strings),
        prefer_topics (array of strings), diversity_weight (number), max_per_activity_fraction (number 0-1),
        prefer_live_override (boolean|null), budget_tolerance (number 0-0.3).
        Use the provided plan snapshot titles when removing or preferring activities.
        Return only JSON, no prose.
        """
    ).strip()

    user_payload = {
        "request": user_text,
        "snapshot": snapshot,
    }

    try:
        messages = [
            {"role": "system", "content": control_system},
            {"role": "user", "content": json.dumps(user_payload, default=str)},
        ]
        reasoning_effort = (
            "medium" if DEFAULT_ASSISTANT_MODEL.startswith("gpt-5") else None
        )
        response = call_responses(
            openai_client,
            model=DEFAULT_ASSISTANT_MODEL,
            messages=messages,
            temperature=0,
            max_output_tokens=800,
            reasoning_effort=reasoning_effort,
        )
    except Exception:
        logging.exception("control derivation failed")
        return ([], [])

    text = response_text(response)
    if not text:
        return ([], [])

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logging.warning("control derivation returned non-JSON: %s", text[:200])
        return ([], [])

    patch_payloads: list[str] = []
    policy_payloads: list[str] = []

    patch = data.get("patch") if isinstance(data, dict) else None
    if isinstance(patch, dict) and patch:
        patch_payloads.append(json.dumps(patch))

    policy = data.get("policy") if isinstance(data, dict) else None
    if isinstance(policy, dict) and policy:
        policy_payloads.append(json.dumps(policy))

    return (patch_payloads, policy_payloads)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    plan_mode = "balanced"

    policy_status_entries: list[dict[str, object]] = []

    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "user": None,
                    "plan": [],
                    "claims": [],
                    "assistant_messages": [],
                    "plan_mode": plan_mode,
                    "active_policy": None,
                    "policy_status": [],
                    "validation": None,
                    "is_dev": bool(
                        os.getenv("ENV") in ("dev", "development")
                        or str(os.getenv("DEBUG")) in ("1", "true", "True")
                    ),
                },
            )

        claims = list(session.exec(select(Claim).where(Claim.user_id == user.id)))
        total_claimed = sum(float(c.credits or 0.0) for c in claims)
        user.remaining_credits = max(user.target_credits - total_claimed, 0.0)
        session.add(user)
        session.commit()

        msgs = list(
            session.exec(
                select(AssistantMessage).where(AssistantMessage.user_id == user.id)
            )
        )

        if not msgs:
            welcome = AssistantMessage(
                user_id=user.id,
                role="assistant",
                content=(
                    "Welcome! Let me know your CME priorities and I’ll start building a plan."
                ),
            )
            session.add(welcome)
            session.commit()
            msgs = [welcome]

        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        active_policy = policy_for_mode(policy_bundle, plan_mode)
        policy_status_entries = []
        for entry in active_policy_status(session, user):
            expires_at = entry.get("expires_at")
            hours_left = None
            if isinstance(expires_at, datetime):
                diff = expires_at - datetime.utcnow()
                hours_left = max(int(diff.total_seconds() // 3600), 0)
            policy_status_entries.append(
                {
                    "mode": entry.get("mode", "default"),
                    "expires_at": (
                        expires_at.isoformat()
                        if isinstance(expires_at, datetime)
                        else None
                    ),
                    "hours_left": hours_left,
                }
            )

        force_refresh = request.query_params.get("refresh", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        plan_run = plan_manager.ensure_plan(
            user,
            plan_mode,
            policy_bundle,
            force_refresh=force_refresh,
            reason="home_refresh" if force_refresh else None,
        )
        plan, plan_summary, plan_requirements = plan_manager.serialize_plan(
            plan_run, user
        )
        requirements_data = load_abpn_psychiatry_requirements()
        validation = validate_against_requirements(user, claims, requirements_data)

        pip_status = {"complete": False, "reason": None}
        for claim in claims:
            topic = (claim.topic or "").lower()
            if topic in {"pip_complete", "pip completion"}:
                pip_status = {
                    "complete": True,
                    "reason": "Documented pip_complete claim",
                }
                break

        validation_full = validate_full_cc(user, claims, requirements_data, pip_status)

        claims = list(session.exec(select(Claim).where(Claim.user_id == user.id)))
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "plan": plan,
                "plan_summary": plan_summary,
                "plan_requirements": plan_requirements,
                "claims": claims,
                "assistant_messages": msgs,
                "validation": validation,
                "validation_full": validation_full,
                "active_policy": active_policy,
                "policy_status": policy_status_entries,
                "plan_mode": plan_mode,
                "is_dev": bool(
                    os.getenv("ENV") in ("dev", "development")
                    or str(os.getenv("DEBUG")) in ("1", "true", "True")
                ),
            },
        )


@app.get("/fragment/plan", response_class=HTMLResponse)
def plan_fragment(request: Request):
    requested_mode = request.query_params.get("mode", "balanced").lower()
    plan_mode = "cheapest" if requested_mode == "cheapest" else "balanced"
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return HTMLResponse("", status_code=204)
        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        force_refresh = request.query_params.get("refresh", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        plan_run = plan_manager.ensure_plan(
            user,
            plan_mode,
            policy_bundle,
            force_refresh=force_refresh,
            reason="fragment_refresh" if force_refresh else None,
        )
        plan, plan_summary, plan_requirements = plan_manager.serialize_plan(
            plan_run, user
        )
        return templates.TemplateResponse(
            "_plan.html",
            {
                "request": request,
                "user": user,
                "plan": plan,
                "plan_summary": plan_summary,
                "plan_requirements": plan_requirements,
                "plan_mode": plan_mode,
            },
        )


def _refresh_plan_fragment_response(
    request: Request,
    session,
    user: User,
    plan_mode: str = "balanced",
    force_refresh: bool = True,
    reason: str | None = None,
):
    plan_manager = PlanManager(session)
    policy_bundle = load_policy_bundle(session, user)
    plan_run = plan_manager.ensure_plan(
        user,
        plan_mode,
        policy_bundle,
        force_refresh=force_refresh,
        reason=reason,
    )
    plan, plan_summary, plan_requirements = plan_manager.serialize_plan(plan_run, user)
    return templates.TemplateResponse(
        "_plan.html",
        {
            "request": request,
            "user": user,
            "plan": plan,
            "plan_summary": plan_summary,
            "plan_requirements": plan_requirements,
            "plan_mode": plan_mode,
        },
    )


@app.post("/plan/commit/{activity_id}", response_class=HTMLResponse)
def plan_commit(request: Request, activity_id: int):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return HTMLResponse("", status_code=204)
        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
        item = session.exec(
            select(PlanItem)
            .where(
                PlanItem.plan_run_id == run.id,
                PlanItem.activity_id == activity_id,
            )
            .limit(1)
        ).first()
        if not item:
            return _refresh_plan_fragment_response(
                request, session, user, plan_mode="balanced", force_refresh=True
            )
        if not item.committed:
            item.committed = True
            session.add(item)
        PlanManager.invalidate_user_plans(session, user.id, reason="commit_toggle")
        session.commit()
    with get_session() as session:
        user = session.exec(select(User)).first()
        return _refresh_plan_fragment_response(
            request,
            session,
            user,
            plan_mode="balanced",
            force_refresh=True,
            reason="commit_toggle",
        )


@app.post("/plan/uncommit/{activity_id}", response_class=HTMLResponse)
def plan_uncommit(request: Request, activity_id: int):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return HTMLResponse("", status_code=204)
        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
        item = session.exec(
            select(PlanItem)
            .where(
                PlanItem.plan_run_id == run.id,
                PlanItem.activity_id == activity_id,
            )
            .limit(1)
        ).first()
        if item and item.committed:
            item.committed = False
            session.add(item)
        PlanManager.invalidate_user_plans(session, user.id, reason="commit_toggle")
        session.commit()
    with get_session() as session:
        user = session.exec(select(User)).first()
        return _refresh_plan_fragment_response(
            request,
            session,
            user,
            plan_mode="balanced",
            force_refresh=True,
            reason="commit_toggle",
        )


@app.post("/setup")
def setup(
    name: str = Form("Doctor"),
    specialty: str = Form("Psychiatry"),
    city: str = Form(""),
    budget_usd: float = Form(...),
    days_off: int = Form(...),
    target_credits: float = Form(...),
    allow_live: bool = Form(False),
    prefer_live: bool = Form(False),
    residency_completion_year: str = Form(""),
    professional_stage: str = Form(""),
    memberships: str = Form(""),
):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            user = User()
        user.name = name or "Doctor"
        user.specialty = specialty or "Psychiatry"
        user.city = city or None
        user.budget_usd = budget_usd
        user.days_off = days_off
        user.target_credits = target_credits
        user.remaining_credits = target_credits
        user.allow_live = allow_live
        user.prefer_live = prefer_live
        try:
            user.residency_completion_year = (
                int(residency_completion_year) if residency_completion_year else None
            )
        except Exception:
            user.residency_completion_year = None
        user.professional_stage = _normalize_stage(professional_stage)
        user.memberships = _parse_membership_input(memberships)
        session.add(user)
        session.commit()
    return RedirectResponse("/", status_code=303)


@app.post("/message")
def message(text: str = Form(...)):
    credits, topic, d = parse_message(text)
    if credits <= 0:
        return RedirectResponse("/", status_code=303)
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return RedirectResponse("/", status_code=303)
        claim = Claim(
            user_id=user.id, credits=credits, topic=topic, date=d, source_text=text
        )
        session.add(claim)
        session.commit()
    return RedirectResponse("/", status_code=303)


@app.get("/log", response_class=HTMLResponse)
def log(request: Request):
    policy_status_entries: list[dict[str, object]] = []
    with get_session() as session:
        user = session.exec(select(User)).first()
        claims = (
            list(session.exec(select(Claim).where(Claim.user_id == user.id)))
            if user
            else []
        )
        msgs = (
            list(
                session.exec(
                    select(AssistantMessage).where(AssistantMessage.user_id == user.id)
                )
            )
            if user
            else []
        )
        requirements_data = load_abpn_psychiatry_requirements() if user else {}
        validation = (
            validate_against_requirements(user, claims, requirements_data)
            if user
            else None
        )
        if user:
            for entry in active_policy_status(session, user):
                expires_at = entry.get("expires_at")
                hours_left = None
                if isinstance(expires_at, datetime):
                    diff = expires_at - datetime.utcnow()
                    hours_left = max(int(diff.total_seconds() // 3600), 0)
                policy_status_entries.append(
                    {
                        "mode": entry.get("mode", "default"),
                        "expires_at": (
                            expires_at.isoformat()
                            if isinstance(expires_at, datetime)
                            else None
                        ),
                        "hours_left": hours_left,
                    }
                )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "plan": [],
            "claims": claims,
            "assistant_messages": msgs,
            "plan_mode": request.query_params.get("mode", "balanced"),
            "validation": validation,
            "policy_status": policy_status_entries,
            "is_dev": bool(
                os.getenv("ENV") in ("dev", "development")
                or str(os.getenv("DEBUG")) in ("1", "true", "True")
            ),
        },
    )


@app.api_route("/ingest", methods=["GET", "POST"])
async def ingest(request: Request = None):
    logger = logging.getLogger(__name__)
    has_google_key = bool(get_secret("GOOGLE_API_KEY"))
    has_google_cx = bool(get_secret("GOOGLE_CSE_ID"))
    logger.info(
        "ingest called. env GOOGLE_API_KEY=%s GOOGLE_CSE_ID=%s",
        has_google_key,
        has_google_cx,
    )
    try:
        # threshold from env (optional)
        min_results = int(get_secret("INGEST_MIN_RESULTS") or "12")
        debug = False
        if request is not None:
            qp = dict(request.query_params)
            debug = str(qp.get("debug", "0")) in ("1", "true", "True")
        result = await asyncio.to_thread(
            ingest_psychiatry_online_ai, count=min_results, debug=debug
        )
        if isinstance(result, dict):
            # debug mode returns counters dict
            return JSONResponse(result)
        # Support tuple sizes of 2 or 3
        google_inserted = None
        ai_inserted = None
        if isinstance(result, tuple):
            if len(result) >= 4:
                ingested = int(result[0])
                used_fallback = bool(result[1])
                google_inserted = int(result[2])
                ai_inserted = int(result[3])
            elif len(result) == 3:
                ingested = int(result[0])
                used_fallback = bool(result[1])
                google_inserted = int(result[2])
                ai_inserted = max(0, ingested - google_inserted)
            else:
                ingested = int(result[0])
                used_fallback = bool(result[1])
        else:
            ingested = int(result)
            used_fallback = False
        payload = {"ingested": ingested, "used_fallback": used_fallback}
        if google_inserted is not None:
            payload["google_inserted"] = google_inserted
            payload["ai_inserted"] = ai_inserted or 0
        return JSONResponse(payload)
    except Exception as e:
        logger.exception("ingest failed")
        return JSONResponse(
            {"error": "ingest_failed", "detail": str(e)}, status_code=500
        )


@app.post("/assist")
def assist():
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "missing OPENAI_API_KEY"}, status_code=400)
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return JSONResponse({"error": "no_user"}, status_code=400)
        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
        plan_entries, _, _ = plan_manager.serialize_plan(run, user)
        plan = [
            {
                "title": entry.get("title"),
                "provider": entry.get("provider"),
                "credits": entry.get("credits"),
                "cost": entry.get("cost"),
                "base_cost": entry.get("base_cost"),
                "price_label": entry.get("price_label"),
                "deadline": entry.get("deadline_text"),
                "modality": entry.get("modality"),
                "city": entry.get("city"),
                "hybrid_available": entry.get("hybrid_available"),
            }
            for entry in plan_entries
        ]
        claims = list(session.exec(select(Claim).where(Claim.user_id == user.id)))

    # Compose system/user prompt
    sys = (
        "You are a concise CME planner assistant. Explain what changed in the user's plan and why, "
        "based on remaining credits, budget, and days off. If remaining credits decreased due to new claims, "
        "explain which items were removed or swapped and why. Then ask 1-2 targeted preference questions. "
        "Keep it under 120 words. Use bullet points for changes, then a single question line."
    )
    u = {
        "user": {
            "name": user.name,
            "specialty": user.specialty,
            "city": user.city,
            "budget_usd": user.budget_usd,
            "days_off": user.days_off,
            "target_credits": user.target_credits,
            "remaining_credits": user.remaining_credits,
            "allow_live": user.allow_live,
            "professional_stage": user.professional_stage,
            "residency_completion_year": user.residency_completion_year,
            "memberships": user.memberships,
        },
        "plan": plan,
        "claims": [
            {
                "date": str(c.date),
                "credits": c.credits,
                "topic": c.topic,
                "text": c.source_text,
            }
            for c in claims
        ],
    }

    resp = call_responses(
        client,
        model="gpt-5",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(u)},
        ],
        temperature=0,
        max_output_tokens=700,
        reasoning_effort="medium",
    )
    text = response_text(resp)

    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            user = User(
                name="Doctor",
                specialty="Psychiatry",
                target_credits=30.0,
                remaining_credits=30.0,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
        elif user.id is None:
            session.add(user)
            session.commit()
            session.refresh(user)
        msg = AssistantMessage(user_id=user.id, content=text.strip()[:4000])
        session.add(msg)
        session.commit()

    return JSONResponse({"message": text})


@app.post("/assist/reply")
def assist_reply(answer: str = Form(...)):
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "missing OPENAI_API_KEY"}, status_code=400)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return RedirectResponse("/", status_code=303)
        current = {
            "budget_usd": user.budget_usd,
            "days_off": user.days_off,
            "allow_live": user.allow_live,
            "city": user.city,
            "specialty": user.specialty,
            "target_credits": user.target_credits,
            "professional_stage": user.professional_stage,
            "residency_completion_year": user.residency_completion_year,
            "memberships": user.memberships,
        }

    sys = (
        "Extract updated preference values from the user's reply. Return STRICT JSON with keys: "
        "budget_usd (number|null), days_off (integer|null), allow_live (boolean|null), city (string|null), "
        "specialty (string|null), target_credits (number|null), professional_stage (string|null), "
        "residency_completion_year (integer|null), memberships (array|string|null). If a value is not provided, set it to null."
    )
    user_msg = {
        "current": current,
        "reply": answer,
    }

    resp = call_responses(
        client,
        model="gpt-5",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user_msg)},
        ],
        temperature=0,
        max_output_tokens=600,
        reasoning_effort="medium",
    )
    text = response_text(resp)
    data = safe_json_loads(text) or {}

    updated_fields = []
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return RedirectResponse("/", status_code=303)
        # Apply updates when present
        if data.get("budget_usd") is not None:
            try:
                user.budget_usd = float(data["budget_usd"])  # type: ignore[assignment]
                updated_fields.append(f"budget=${int(user.budget_usd)}")
            except Exception:
                pass
        if data.get("days_off") is not None:
            try:
                user.days_off = int(data["days_off"])  # type: ignore[assignment]
                updated_fields.append(f"days_off={user.days_off}")
            except Exception:
                pass
        if data.get("allow_live") is not None:
            try:
                user.allow_live = bool(data["allow_live"])  # type: ignore[assignment]
                updated_fields.append(
                    f"allow_live={'Yes' if user.allow_live else 'No'}"
                )
            except Exception:
                pass
        if data.get("city") is not None:
            try:
                user.city = data["city"] or None  # type: ignore[assignment]
                updated_fields.append(f"city={user.city}")
            except Exception:
                pass
        if data.get("specialty") is not None:
            try:
                user.specialty = data["specialty"] or user.specialty  # type: ignore[assignment]
                updated_fields.append(f"specialty={user.specialty}")
            except Exception:
                pass
        if data.get("target_credits") is not None:
            try:
                user.target_credits = float(data["target_credits"])  # type: ignore[assignment]
                updated_fields.append(f"target_credits={user.target_credits}")
            except Exception:
                pass
        if data.get("professional_stage") is not None:
            stage = _normalize_stage(data.get("professional_stage"))
            user.professional_stage = stage
            updated_fields.append(f"stage={stage or 'cleared'}")
        if data.get("residency_completion_year") is not None:
            try:
                year_val = (
                    int(data["residency_completion_year"])
                    if data["residency_completion_year"]
                    else None
                )
            except Exception:
                year_val = None
            user.residency_completion_year = year_val
            updated_fields.append(
                f"residency_year={year_val if year_val else 'cleared'}"
            )
        if data.get("memberships") is not None:
            parsed_memberships = _parse_membership_input(data.get("memberships"))
            user.memberships = parsed_memberships
            updated_fields.append(
                "memberships="
                + (", ".join(parsed_memberships) if parsed_memberships else "cleared")
            )
        session.add(user)
        session.commit()

        # Save assistant confirmation message
        summary = (
            "Preferences updated: " + ", ".join(updated_fields)
            if updated_fields
            else "No changes detected."
        )
        session.add(AssistantMessage(user_id=user.id, content=summary))
        session.commit()

    return RedirectResponse("/", status_code=303)


@app.get("/preferences", response_class=HTMLResponse)
def preferences_form(request: Request):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(
            "preferences.html", {"request": request, "user": user}
        )


@app.post("/preferences")
def preferences_submit(
    name: str = Form(...),
    specialty: str = Form(...),
    city: str = Form(""),
    budget_usd: float = Form(...),
    days_off: int = Form(...),
    target_credits: float = Form(...),
    allow_live: bool = Form(False),
    prefer_live: bool = Form(False),
    residency_completion_year: str = Form(""),
    professional_stage: str = Form(""),
    memberships: str = Form(""),
):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return RedirectResponse("/", status_code=303)
        user.name = name or user.name
        user.specialty = specialty or user.specialty
        user.city = city or None
        user.budget_usd = budget_usd
        user.days_off = days_off
        user.target_credits = target_credits
        user.allow_live = allow_live
        user.prefer_live = prefer_live
        try:
            user.residency_completion_year = (
                int(residency_completion_year) if residency_completion_year else None
            )
        except Exception:
            user.residency_completion_year = None
        user.professional_stage = _normalize_stage(professional_stage)
        user.memberships = _parse_membership_input(memberships)
        session.add(user)
        session.commit()
    return RedirectResponse("/", status_code=303)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")


@app.post("/assist/command")
def assist_command(command: str = Form(...)):
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "missing OPENAI_API_KEY"}, status_code=400)
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # Ask model to produce a compact policy JSON
    sys = (
        "You convert natural language planning instructions into a compact JSON policy. "
        "Output ONLY JSON with these optional keys: "
        "avoid_terms (array of strings), prefer_topics (array of strings), diversity_weight (number 0..2), "
        "max_per_activity_fraction (number 0..1), prefer_live_override (true|false|null), budget_tolerance (number 0..0.3)."
    )
    try:
        resp = call_responses(
            client,
            model="gpt-5",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": command},
            ],
            temperature=0,
            max_output_tokens=600,
            reasoning_effort="medium",
        )
        text = response_text(resp)
        data = safe_json_loads(text) or {}
        # Sanitize types and defaults
        policy = {
            "avoid_terms": (
                list(filter(None, (data.get("avoid_terms") or [])))
                if isinstance(data.get("avoid_terms"), list)
                else []
            ),
            "prefer_topics": (
                list(filter(None, (data.get("prefer_topics") or [])))
                if isinstance(data.get("prefer_topics"), list)
                else []
            ),
            "diversity_weight": float(data.get("diversity_weight") or 0.0),
            "max_per_activity_fraction": float(
                data.get("max_per_activity_fraction") or 0.0
            )
            or None,
            "prefer_live_override": (
                data.get("prefer_live_override")
                if isinstance(data.get("prefer_live_override"), bool)
                else None
            ),
            "budget_tolerance": float(data.get("budget_tolerance") or 0.0),
        }
    except Exception:
        logging.exception("assist_command policy parse failed")
        return JSONResponse({"error": "policy_parse_failed"}, status_code=500)

    # Save policy and a summary message, then redirect home to apply it
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return RedirectResponse("/", status_code=303)
        apply_policy_payloads(
            [json.dumps(policy)],
            user,
            session,
            record_message=False,
        )

    return RedirectResponse("/?policy=1", status_code=303)


@app.get("/chat", response_class=HTMLResponse)
def chat_fragment(request: Request):
    with get_session() as session:
        user = session.exec(select(User)).first()
        query = (
            select(AssistantMessage)
            .order_by(AssistantMessage.created_at.desc())
            .limit(50)
        )
        if user:
            query = query.where(AssistantMessage.user_id == user.id)
        messages = list(session.exec(query))
        messages.reverse()
        filtered: list[AssistantMessage] = []
        for m in messages:
            content = (m.content or "").strip()
            if content.startswith("POLICY:"):
                continue
            if content.startswith("PATCH:"):
                continue
            if content.startswith("INTERNAL:"):
                continue
            filtered.append(m)
    return templates.TemplateResponse(
        "_chat_messages.html",
        {"request": request, "messages": filtered},
    )


@app.post("/chat/send", response_class=HTMLResponse)
def chat_send(request: Request, text: str = Form(...)):
    if not openai_client:
        return HTMLResponse(
            "<div class='bubble error'>Assistant unavailable: missing OPENAI_API_KEY.</div>",
            status_code=503,
        )

    try:
        normalized_text = text.strip().lower()
        should_ingest = False
        plan_update_needed = False
        ingest_success = False
        action_command: str | None = None
        force_plan_refresh = False
        activities_to_commit: set[int] = set()

        with get_session() as session:
            user = session.exec(select(User)).first()
            if not user:
                user = User(
                    name="Doctor",
                    specialty="Psychiatry",
                    target_credits=30.0,
                    remaining_credits=30.0,
                )
                session.add(user)
                session.commit()
                session.refresh(user)
            elif user.id is None:
                session.add(user)
                session.commit()
                session.refresh(user)

            plan_manager = PlanManager(session)
            user_message = AssistantMessage(
                user_id=user.id, role="user", content=text.strip()[:4000]
            )
            session.add(user_message)

            pending_notes: list[str] = []
            removal_note, removal_changed = _maybe_remove_claim_from_text(
                user, text, session
            )
            if removal_note:
                pending_notes.append(removal_note)
            if removal_changed:
                plan_update_needed = True
            note_text = _maybe_log_claim_from_text(user, text, session)
            if note_text:
                pending_notes.append(note_text)
                plan_update_needed = True
            note_text = _maybe_complete_activity_from_text(user, text, session)
            if note_text:
                pending_notes.append(note_text)

            policy_bundle = None
            plan_run = None

            removal_targets = _extract_remove_targets(text)
            if removal_targets:
                removal_hits: list[Activity] = []
                missing_removals: list[str] = []
                for target in removal_targets:
                    activity = find_activity_by_title(session, user, target)
                    if activity:
                        removal_hits.append(activity)
                    else:
                        missing_removals.append(target)
                if removal_hits:
                    if policy_bundle is None:
                        policy_bundle = load_policy_bundle(session, user)
                    if plan_run is None:
                        plan_run = plan_manager.ensure_plan(
                            user, "balanced", policy_bundle
                        )
                    plan_items = list(
                        session.exec(
                            select(PlanItem).where(PlanItem.plan_run_id == plan_run.id)
                        )
                    )
                    plan_item_lookup = {
                        item.activity_id: item
                        for item in plan_items
                        if item.activity_id
                    }
                    for activity in removal_hits:
                        item = plan_item_lookup.get(activity.id)
                        if item and item.committed:
                            item.committed = False
                            session.add(item)
                    removal_payload = json.dumps(
                        {"remove_titles": [activity.title for activity in removal_hits]}
                    )
                    apply_policy_payloads(
                        [removal_payload],
                        user,
                        session,
                        invalidate=False,
                        record_message=False,
                    )
                    session.flush()
                    PlanManager.invalidate_user_plans(
                        session, user.id, reason="plan_remove"
                    )
                    plan_update_needed = True
                    force_plan_refresh = True
                    removed_titles = ", ".join(
                        activity.title for activity in removal_hits
                    )
                    pending_notes.append(f"Removed from plan: {removed_titles}.")
                    plan_run = None
                    policy_bundle = None
                    pending_notes = [
                        note
                        for note in pending_notes
                        if note != "No logged credits to remove."
                    ]
                if missing_removals:
                    pending_notes.append(
                        "I couldn't find "
                        + ", ".join(missing_removals)
                        + " in the current plan—double-check the titles and I'll remove them."
                    )

            addition_targets = _extract_add_targets(text)
            if addition_targets:
                added_titles: list[str] = []
                missing_additions: list[str] = []
                for target in addition_targets:
                    pending_notes.append(f"Searching for details on {target}…")
                for target in addition_targets:
                    query_tokens = _title_tokens(target)
                    candidate = find_activity_by_title(session, user, target)
                    if not candidate:
                        try:
                            inserted = discover_activity_by_title(user, target)
                            if inserted > 0:
                                ingest_success = True
                        except Exception:
                            logging.exception(
                                "targeted discovery failed for %s", target
                            )
                        PlanManager.invalidate_user_plans(
                            session, user.id, reason="ingest_focus"
                        )
                        force_plan_refresh = True
                        plan_run = None
                        policy_bundle = None
                        candidate = find_activity_by_title(session, user, target)
                    if candidate:
                        if _is_candidate_match(candidate.title or "", query_tokens):
                            added_titles.append(candidate.title or target)
                            activities_to_commit.add(candidate.id)
                            plan_update_needed = True
                            force_plan_refresh = True
                        else:
                            missing_additions.append(target)
                    else:
                        missing_additions.append(target)
                if added_titles:
                    pending_notes.append(
                        "Added to plan: " + ", ".join(added_titles) + "."
                    )
                if missing_additions:
                    pending_notes.append(
                        "I couldn’t locate details for "
                        + ", ".join(missing_additions)
                        + " yet—share a link or more info and I’ll add it."
                    )

            if "keep this plan" in normalized_text:
                if policy_bundle is None:
                    policy_bundle = load_policy_bundle(session, user)
                if plan_run is None:
                    plan_run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
                plan_items = list(
                    session.exec(
                        select(PlanItem)
                        .where(PlanItem.plan_run_id == plan_run.id)
                        .order_by(PlanItem.position.asc())
                    )
                )
                newly_committed = 0
                for item in plan_items:
                    if not item.committed:
                        item.committed = True
                        session.add(item)
                        newly_committed += 1
                if newly_committed:
                    PlanManager.invalidate_user_plans(
                        session, user.id, reason="commit_all"
                    )
                session.commit()

                refreshed_run = plan_manager.ensure_plan(
                    user,
                    "balanced",
                    load_policy_bundle(session, user),
                    force_refresh=True,
                    reason="commit_all" if newly_committed else None,
                )
                plan, plan_summary, _ = plan_manager.serialize_plan(refreshed_run, user)
                confirmation_text = (
                    f"Locked in {len(plan)} activities; I'll keep them fixed while we fill remaining gaps."
                    if newly_committed
                    else "Everything in the current plan was already committed."
                )
                confirmation = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=confirmation_text[:4000],
                )
                session.add(confirmation)

                item_count = len(plan)
                total_cost_value = None
                if isinstance(plan_summary, dict):
                    total_cost_value = plan_summary.get("total_cost")
                cost_phrase = (
                    f"${total_cost_value:,.0f}"
                    if isinstance(total_cost_value, (int, float))
                    else "$0"
                )
                remaining_value = float(user.remaining_credits or 0.0)
                plan_note = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=(
                        f"Plan updated ({item_count} items, {cost_phrase} total, "
                        f"{remaining_value:.1f} remaining)."
                    ),
                )
                session.add(plan_note)
                session.commit()

                messages_to_render = [
                    SimpleNamespace(
                        role=user_message.role, content=user_message.content
                    ),
                    SimpleNamespace(
                        role=confirmation.role, content=confirmation.content
                    ),
                    SimpleNamespace(role=plan_note.role, content=plan_note.content),
                ]

                time.sleep(1.2)
                response = templates.TemplateResponse(
                    "_chat_messages_append.html",
                    {"request": request, "messages": messages_to_render},
                )
                response.headers["HX-Trigger"] = json.dumps({"plan-refresh": True})
                return response

            user_text_trimmed = text.strip()
            if user_text_trimmed.lower().startswith("remove "):
                target_text = user_text_trimmed[7:].strip().strip(".!")
                activity = find_activity_by_title(session, user, target_text)

                if not activity:
                    miss_msg = AssistantMessage(
                        user_id=user.id,
                        role="assistant",
                        content=(
                            "I couldn't find that activity in the current plan. "
                            "Double-check the title and I'll remove it."
                        ),
                    )
                    session.add(miss_msg)
                    session.commit()
                    messages_to_render = [
                        SimpleNamespace(
                            role=user_message.role, content=user_message.content
                        ),
                        SimpleNamespace(role=miss_msg.role, content=miss_msg.content),
                    ]
                    time.sleep(1.2)
                    return templates.TemplateResponse(
                        "_chat_messages_append.html",
                        {"request": request, "messages": messages_to_render},
                    )

                policy_bundle = load_policy_bundle(session, user)
                run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
                plan_items = list(
                    session.exec(
                        select(PlanItem)
                        .where(
                            PlanItem.plan_run_id == run.id,
                            PlanItem.activity_id == activity.id,
                        )
                        .order_by(PlanItem.position.asc())
                    )
                )

                for item in plan_items:
                    if item.committed:
                        item.committed = False
                        session.add(item)

                removal_payload = json.dumps({"remove_titles": [activity.title]})
                apply_policy_payloads(
                    [removal_payload],
                    user,
                    session,
                    invalidate=False,
                    record_message=False,
                )

                PlanManager.invalidate_user_plans(
                    session, user.id, reason="remove_activity"
                )
                session.commit()

                refreshed_run = plan_manager.ensure_plan(
                    user,
                    "balanced",
                    load_policy_bundle(session, user),
                    force_refresh=True,
                    reason="remove_activity",
                )
                plan, plan_summary, _ = plan_manager.serialize_plan(refreshed_run, user)

                cost_phrase = "$0"
                if isinstance(plan_summary, dict):
                    total_cost_value = plan_summary.get("total_cost")
                    if isinstance(total_cost_value, (int, float)):
                        cost_phrase = f"${total_cost_value:,.0f}"

                remaining_value = float(user.remaining_credits or 0.0)

                removal_msg = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=(f"Removed {activity.title}. Shall I find a substitute?"),
                )
                session.add(removal_msg)
                session.add(
                    AssistantMessage(
                        user_id=user.id,
                        role="assistant",
                        content=f"INTERNAL:SUBSTITUTE_REQUEST:{activity.id}",
                    )
                )

                plan_note = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=(
                        f"Plan updated ({len(plan)} items, {cost_phrase} total, "
                        f"{remaining_value:.1f} remaining)."
                    ),
                )
                session.add(plan_note)
                session.commit()

                messages_to_render = [
                    SimpleNamespace(
                        role=user_message.role, content=user_message.content
                    ),
                    SimpleNamespace(role=removal_msg.role, content=removal_msg.content),
                    SimpleNamespace(role=plan_note.role, content=plan_note.content),
                ]

                time.sleep(1.2)
                response = templates.TemplateResponse(
                    "_chat_messages_append.html",
                    {"request": request, "messages": messages_to_render},
                )
                response.headers["HX-Trigger"] = json.dumps({"plan-refresh": True})
                return response

            affirmative = _is_affirmative(text)
            negative = _is_negative(text)

            if affirmative:
                candidate_msg = _pop_internal_message(
                    session, user.id, "INTERNAL:SUBSTITUTE_CANDIDATE:"
                )
                if candidate_msg:
                    try:
                        candidate_id = int(candidate_msg.content.split(":", 2)[-1])
                    except Exception:
                        candidate_id = None

                    candidate_activity = (
                        session.get(Activity, candidate_id) if candidate_id else None
                    )
                    if not candidate_activity:
                        error_msg = AssistantMessage(
                            user_id=user.id,
                            role="assistant",
                            content="I couldn't locate that option anymore, but I can search again if you'd like.",
                        )
                        session.add(error_msg)
                        session.commit()
                        messages_to_render = [
                            SimpleNamespace(
                                role=user_message.role, content=user_message.content
                            ),
                            SimpleNamespace(
                                role=error_msg.role, content=error_msg.content
                            ),
                        ]
                        time.sleep(1.2)
                        return templates.TemplateResponse(
                            "_chat_messages_append.html",
                            {"request": request, "messages": messages_to_render},
                        )

                    policy_bundle = load_policy_bundle(session, user)
                    run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
                    plan_item = session.exec(
                        select(PlanItem)
                        .where(
                            PlanItem.plan_run_id == run.id,
                            PlanItem.activity_id == candidate_activity.id,
                        )
                        .limit(1)
                    ).first()

                    if plan_item is None:
                        current_max = session.exec(
                            select(func.max(PlanItem.position)).where(
                                PlanItem.plan_run_id == run.id
                            )
                        ).first()
                        next_position = int(current_max or 0) + 1
                        cost, _, pricing = _activity_cost_and_days(
                            user, candidate_activity
                        )
                        plan_item = PlanItem(
                            user_id=user.id,
                            activity_id=candidate_activity.id,
                            plan_run_id=run.id,
                            mode="balanced",
                            position=next_position,
                            chosen=True,
                            pricing_snapshot={
                                "cost": cost,
                                "base_cost": pricing.get("base_cost"),
                                "deadline_text": pricing.get("deadline_text"),
                                "deadline_urgency": pricing.get("deadline_urgency"),
                                "price_label": pricing.get("price_label"),
                                "notes": pricing.get("notes"),
                                "hybrid_available": pricing.get("hybrid_available"),
                            },
                            requirement_snapshot={
                                "tags": [],
                                "requirement_priority": False,
                            },
                            eligibility_status=(
                                "eligible"
                                if is_eligible(user, candidate_activity)
                                else "uncertain"
                            ),
                            notes=pricing.get("notes"),
                            committed=True,
                        )
                        session.add(plan_item)
                    else:
                        if not plan_item.committed:
                            plan_item.committed = True
                            session.add(plan_item)

                    PlanManager.invalidate_user_plans(
                        session, user.id, reason="commit_substitute"
                    )
                    session.commit()

                    refreshed_run = plan_manager.ensure_plan(
                        user,
                        "balanced",
                        load_policy_bundle(session, user),
                        force_refresh=True,
                        reason="commit_substitute",
                    )
                    plan, plan_summary, _ = plan_manager.serialize_plan(
                        refreshed_run, user
                    )

                    cost_phrase = "$0"
                    if isinstance(plan_summary, dict):
                        total_cost_value = plan_summary.get("total_cost")
                        if isinstance(total_cost_value, (int, float)):
                            cost_phrase = f"${total_cost_value:,.0f}"

                    remaining_value = float(user.remaining_credits or 0.0)

                    ack = AssistantMessage(
                        user_id=user.id,
                        role="assistant",
                        content=(
                            f"Added {candidate_activity.title} to your plan. Anything else you'd like to adjust?"
                        ),
                    )
                    session.add(ack)

                    plan_note = AssistantMessage(
                        user_id=user.id,
                        role="assistant",
                        content=(
                            f"Plan updated ({len(plan)} items, {cost_phrase} total, "
                            f"{remaining_value:.1f} remaining)."
                        ),
                    )
                    session.add(plan_note)
                    session.commit()

                    messages_to_render = [
                        SimpleNamespace(
                            role=user_message.role, content=user_message.content
                        ),
                        SimpleNamespace(role=ack.role, content=ack.content),
                        SimpleNamespace(role=plan_note.role, content=plan_note.content),
                    ]

                    time.sleep(1.2)
                    response = templates.TemplateResponse(
                        "_chat_messages_append.html",
                        {"request": request, "messages": messages_to_render},
                    )
                    response.headers["HX-Trigger"] = json.dumps({"plan-refresh": True})
                    return response

                request_msg = _pop_internal_message(
                    session, user.id, "INTERNAL:SUBSTITUTE_REQUEST:"
                )
                if request_msg:
                    try:
                        removed_id = int(request_msg.content.split(":", 2)[-1])
                    except Exception:
                        removed_id = None
                    removed_activity = (
                        session.get(Activity, removed_id) if removed_id else None
                    )
                    candidate = propose_substitute(session, user, removed_activity)
                    if candidate:
                        cost, _, pricing = _activity_cost_and_days(user, candidate)
                        cost_phrase = f"${cost:,.0f}" if cost else "Free"
                        message_text = (
                            f"I found {candidate.title} ({candidate.credits:.1f} cr, {cost_phrase}). "
                            "Add it to the plan?"
                        )
                        suggestion = AssistantMessage(
                            user_id=user.id,
                            role="assistant",
                            content=message_text[:4000],
                        )
                        session.add(suggestion)
                        session.add(
                            AssistantMessage(
                                user_id=user.id,
                                role="assistant",
                                content=f"INTERNAL:SUBSTITUTE_CANDIDATE:{candidate.id}",
                            )
                        )
                        session.commit()
                        messages_to_render = [
                            SimpleNamespace(
                                role=user_message.role, content=user_message.content
                            ),
                            SimpleNamespace(
                                role=suggestion.role, content=suggestion.content
                            ),
                        ]
                        time.sleep(1.2)
                        return templates.TemplateResponse(
                            "_chat_messages_append.html",
                            {"request": request, "messages": messages_to_render},
                        )
                    else:
                        no_sub_msg = AssistantMessage(
                            user_id=user.id,
                            role="assistant",
                            content=(
                                "I didn't find a good substitute yet, but I can broaden the search whenever you like."
                            ),
                        )
                        session.add(no_sub_msg)
                        session.commit()
                        messages_to_render = [
                            SimpleNamespace(
                                role=user_message.role, content=user_message.content
                            ),
                            SimpleNamespace(
                                role=no_sub_msg.role, content=no_sub_msg.content
                            ),
                        ]
                        time.sleep(1.2)
                        return templates.TemplateResponse(
                            "_chat_messages_append.html",
                            {"request": request, "messages": messages_to_render},
                        )

            if negative:
                candidate_msg = _pop_internal_message(
                    session, user.id, "INTERNAL:SUBSTITUTE_CANDIDATE:"
                )
                if candidate_msg:
                    decline = AssistantMessage(
                        user_id=user.id,
                        role="assistant",
                        content="No problem—I'll leave that slot open. Let me know if you want other ideas.",
                    )
                    session.add(decline)
                    session.commit()
                    messages_to_render = [
                        SimpleNamespace(
                            role=user_message.role, content=user_message.content
                        ),
                        SimpleNamespace(role=decline.role, content=decline.content),
                    ]
                    time.sleep(1.2)
                    return templates.TemplateResponse(
                        "_chat_messages_append.html",
                        {"request": request, "messages": messages_to_render},
                    )

                request_msg = _pop_internal_message(
                    session, user.id, "INTERNAL:SUBSTITUTE_REQUEST:"
                )
                if request_msg:
                    decline = AssistantMessage(
                        user_id=user.id,
                        role="assistant",
                        content="Got it—I’ll keep that slot empty for now.",
                    )
                    session.add(decline)
                    session.commit()
                    messages_to_render = [
                        SimpleNamespace(
                            role=user_message.role, content=user_message.content
                        ),
                        SimpleNamespace(role=decline.role, content=decline.content),
                    ]
                    time.sleep(1.2)
                    return templates.TemplateResponse(
                        "_chat_messages_append.html",
                        {"request": request, "messages": messages_to_render},
                    )

            if normalized_text in {"clear policy", "clear policies", "reset policy"}:
                cleared = clear_policies(session, user)
                policy_message = (
                    "Policy cleared. Planner reset."
                    if cleared
                    else "No active policy to clear."
                )
                confirm = AssistantMessage(
                    user_id=user.id, role="assistant", content=policy_message
                )
                session.add(confirm)

                policy_bundle = load_policy_bundle(session, user)
                plan_run = plan_manager.ensure_plan(
                    user,
                    "balanced",
                    policy_bundle,
                    force_refresh=True,
                    reason="policy_cleared",
                )
                plan_items, plan_summary, _ = plan_manager.serialize_plan(
                    plan_run, user
                )
                item_count = len(plan_items)
                total_cost_value = (
                    plan_summary.get("total_cost")
                    if isinstance(plan_summary, dict)
                    else None
                )
                cost_phrase = (
                    f"${total_cost_value:,.0f}"
                    if isinstance(total_cost_value, (int, float))
                    else "$0"
                )
                remaining_value = float(user.remaining_credits or 0.0)

                summary_msg = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content="Balanced plan refreshed.",
                )
                plan_note = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=(
                        f"Plan updated ({item_count} items, {cost_phrase} total, "
                        f"{remaining_value:.1f} remaining)."
                    ),
                )
                session.add(summary_msg)
                session.add(plan_note)
                session.commit()

                messages_to_render = [
                    SimpleNamespace(
                        role=user_message.role, content=user_message.content
                    ),
                    SimpleNamespace(role=confirm.role, content=confirm.content),
                    SimpleNamespace(role=summary_msg.role, content=summary_msg.content),
                    SimpleNamespace(role=plan_note.role, content=plan_note.content),
                ]
                response = templates.TemplateResponse(
                    "_chat_messages_append.html",
                    {"request": request, "messages": messages_to_render},
                )
                response.headers["HX-Trigger"] = json.dumps({"plan-refresh": True})
                time.sleep(1.2)
                return response

            snapshot = _state_snapshot(session, user)
            system_prompt = build_system_prompt(snapshot)

            user_payload = f"User said: {text}\n\nCurrent state snapshot:\n{json.dumps(snapshot, default=str)}"

            model_name = DEFAULT_ASSISTANT_MODEL
            if os.getenv("ENABLE_GPT5") and os.getenv("ENABLE_GPT5").lower() in {
                "1",
                "true",
                "yes",
            }:
                model_name = "gpt-5"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ]
            reasoning_effort = "medium" if model_name.startswith("gpt-5") else None
            response = call_responses(
                openai_client,
                model=model_name,
                messages=messages,
                temperature=0,
                max_output_tokens=1200,
                reasoning_effort=reasoning_effort,
            )

            reply = response_text(response)

            patch_payloads: list[str] = []
            policy_payloads: list[str] = []
            action_command = None
            visible_lines: list[str] = []
            for line in reply.splitlines():
                stripped = line.strip()
                if stripped.startswith("PATCH:"):
                    payload = stripped.removeprefix("PATCH:").strip()
                    if payload:
                        patch_payloads.append(payload)
                    continue
                if stripped.startswith("POLICY:"):
                    payload = stripped.removeprefix("POLICY:").strip()
                    if payload:
                        policy_payloads.append(payload)
                    continue
                if stripped.startswith("ACTION:"):
                    action_command = (
                        stripped.removeprefix("ACTION:").strip().lower() or None
                    )
                    if action_command in {"discover", "ingest", "refresh"}:
                        should_ingest = True
                    continue
                visible_lines.append(line)

            visible_reply = "\n".join(visible_lines).strip()
            if not visible_reply:
                visible_reply = "Updated the plan to match your request."

            if action_command in {"discover", "ingest", "refresh"}:
                should_ingest = True

            if removal_changed:
                removal_ack = "Cleared the logged credits you flagged and refreshed the plan around the remaining gaps."
                deny_phrases = (
                    "can't",
                    "cannot",
                    "unable",
                    "don't have the capability",
                    "do not have the capability",
                    "can't modify",
                    "cannot modify",
                    "can't delete",
                    "cannot delete",
                    "unable to delete",
                )
                reply_lower = visible_reply.lower()
                if any(phrase in reply_lower for phrase in deny_phrases):
                    visible_reply = removal_ack
                elif removal_ack.lower() not in reply_lower:
                    visible_reply = f"{visible_reply}\n\n{removal_ack}"

            if not patch_payloads and not policy_payloads:
                derived_patch, derived_policy = _derive_controls(text, snapshot)
                patch_payloads.extend(derived_patch)
                policy_payloads.extend(derived_policy)

            assistant_message = AssistantMessage(
                user_id=user.id, role="assistant", content=visible_reply[:4000]
            )
            session.add(assistant_message)

            extra_messages = _apply_patch_if_present(patch_payloads, user, session)
            if extra_messages:
                plan_update_needed = True
            if policy_payloads:
                apply_policy_payloads(
                    policy_payloads,
                    user,
                    session,
                    record_message=False,
                )
                plan_update_needed = True

            confirmations: list[AssistantMessage] = []
            for note_text in pending_notes:
                note_msg = AssistantMessage(
                    user_id=user.id, role="assistant", content=note_text[:4000]
                )
                session.add(note_msg)
                confirmations.append(note_msg)
            try:
                match = PATCH_RE.search(reply or "")
                if match:
                    patch = _safe_json(match.group(1))
                    if isinstance(patch, dict):
                        changed = _apply_patch_to_user(patch, session, user)
                        if changed:
                            plan_update_needed = True
                            confirm = AssistantMessage(
                                user_id=user.id,
                                role="assistant",
                                content=(
                                    "Applied: "
                                    + ", ".join(changed)
                                    + ". The plan will reflect these on refresh."
                                ),
                            )
                            session.add(confirm)
                            session.commit()
                            confirmations.append(confirm)
            except Exception:
                logging.exception("PATCH apply failed")

            try:
                note = _maybe_set_remaining_from_text(user, text, session)
                if note:
                    adjustment_message = AssistantMessage(
                        user_id=user.id, role="assistant", content=note
                    )
                    session.add(adjustment_message)
                    session.commit()
                    confirmations.append(adjustment_message)
            except Exception:
                logging.exception("Remaining adjustment failed")

            if should_ingest:
                trigger_reason = action_command or "keyword"
                logger.info(
                    "chat_send: triggering discovery (reason=%s)", trigger_reason
                )
                try:
                    ingest_psychiatry_online_ai(count=10, debug=False)
                    ingest_success = True
                except Exception:
                    logging.exception("ingest during chat failed")
                if ingest_success:
                    PlanManager.invalidate_user_plans(
                        session, user.id, reason="ingest_refresh"
                    )
                plan_update_needed = True

            if plan_update_needed:
                logger.info(
                    "chat_send: refreshing balanced plan (ingest_success=%s)",
                    ingest_success,
                )
                policy_bundle = load_policy_bundle(session, user)
                plan_run = plan_manager.ensure_plan(
                    user,
                    "balanced",
                    policy_bundle,
                    force_refresh=ingest_success or force_plan_refresh,
                    reason=(
                        "chat_refresh"
                        if (ingest_success or force_plan_refresh)
                        else None
                    ),
                )
                if activities_to_commit:
                    plan_items_db = list(
                        session.exec(
                            select(PlanItem).where(PlanItem.plan_run_id == plan_run.id)
                        )
                    )
                    commit_changed = False
                    plan_item_lookup = {
                        item.activity_id: item
                        for item in plan_items_db
                        if item.activity_id
                    }
                    missing_activity_ids = [
                        activity_id
                        for activity_id in activities_to_commit
                        if activity_id not in plan_item_lookup
                    ]
                    if missing_activity_ids:
                        current_max = session.exec(
                            select(func.max(PlanItem.position)).where(
                                PlanItem.plan_run_id == plan_run.id
                            )
                        ).first()
                        next_position = int(current_max or 0) + 1
                        for activity_id in missing_activity_ids:
                            activity = session.get(Activity, activity_id)
                            if not activity:
                                continue
                            cost, _, pricing = _activity_cost_and_days(user, activity)
                            new_item = PlanItem(
                                user_id=user.id,
                                activity_id=activity.id,
                                plan_run_id=plan_run.id,
                                mode="balanced",
                                position=next_position,
                                chosen=True,
                                pricing_snapshot={
                                    "cost": cost,
                                    "base_cost": pricing.get("base_cost"),
                                    "deadline_text": pricing.get("deadline_text"),
                                    "deadline_urgency": pricing.get("deadline_urgency"),
                                    "price_label": pricing.get("price_label"),
                                    "notes": pricing.get("notes"),
                                    "hybrid_available": pricing.get("hybrid_available"),
                                },
                                requirement_snapshot={
                                    "tags": [],
                                    "requirement_priority": False,
                                },
                                eligibility_status=(
                                    "eligible"
                                    if is_eligible(user, activity)
                                    else "uncertain"
                                ),
                                notes=pricing.get("notes"),
                                committed=True,
                            )
                            session.add(new_item)
                            next_position += 1
                            commit_changed = True
                    for item in plan_items_db:
                        if (
                            item.activity_id in activities_to_commit
                            and not item.committed
                        ):
                            item.committed = True
                            session.add(item)
                            commit_changed = True
                    if commit_changed:
                        session.commit()
                        PlanManager.invalidate_user_plans(
                            session, user.id, reason="commit_additions"
                        )
                        policy_bundle = load_policy_bundle(session, user)
                        plan_run = plan_manager.ensure_plan(
                            user,
                            "balanced",
                            policy_bundle,
                            force_refresh=True,
                            reason="commit_additions",
                        )

                plan_items, plan_summary, _ = plan_manager.serialize_plan(
                    plan_run, user
                )
                item_count = len(plan_items)
                total_cost_value = None
                if isinstance(plan_summary, dict):
                    total_cost_value = plan_summary.get("total_cost")
                if isinstance(total_cost_value, (int, float)):
                    cost_phrase = f"${total_cost_value:,.0f}"
                else:
                    cost_phrase = "$0"
                remaining_value = float(user.remaining_credits or 0.0)
                plan_note_message = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=(
                        f"Plan updated ({item_count} items, {cost_phrase} total, "
                        f"{remaining_value:.1f} remaining)."
                    ),
                )
                session.add(plan_note_message)
                confirmations.append(plan_note_message)
            messages_to_render = [
                SimpleNamespace(role=m.role, content=m.content)
                for m in [
                    user_message,
                    assistant_message,
                    *extra_messages,
                    *confirmations,
                ]
            ]

            session.commit()

        response = templates.TemplateResponse(
            "_chat_messages_append.html",
            {"request": request, "messages": messages_to_render},
        )
        if plan_update_needed:
            response.headers["HX-Trigger"] = json.dumps({"plan-refresh": True})
        time.sleep(1.2)
        return response
    except Exception as exc:
        logging.exception("chat_send failed")
        return HTMLResponse(
            f"<div class='bubble error'>Assistant error: {str(exc)[:200]}</div>",
            status_code=500,
        )


@app.post("/plan/remove", response_class=HTMLResponse)
def plan_remove(request: Request, title: str = Form(...)):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return HTMLResponse("", status_code=204)

        activity = find_activity_by_title(session, user, title)
        if not activity:
            return _refresh_plan_fragment_response(
                request,
                session,
                user,
                plan_mode="balanced",
                force_refresh=False,
            )

        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
        plan_items = list(
            session.exec(
                select(PlanItem)
                .where(
                    PlanItem.plan_run_id == run.id,
                    PlanItem.activity_id == activity.id,
                )
                .order_by(PlanItem.position.asc())
            )
        )

        for item in plan_items:
            if item.committed:
                item.committed = False
                session.add(item)

        removal_payload = json.dumps({"remove_titles": [activity.title]})
        apply_policy_payloads(
            [removal_payload],
            user,
            session,
            invalidate=False,
            record_message=False,
        )

        PlanManager.invalidate_user_plans(session, user.id, reason="plan_remove")
        session.commit()

        return _refresh_plan_fragment_response(
            request,
            session,
            user,
            plan_mode="balanced",
            force_refresh=True,
            reason="plan_remove",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/requirements/sync")
def requirements_sync():
    return JSONResponse(
        {"status": "not_implemented", "detail": "RAG sync scaffolding"},
        status_code=501,
    )


def _policy_clear_impl() -> list[dict[str, object]]:
    with get_session() as session:
        notes = list(
            session.exec(
                select(AssistantMessage).where(
                    AssistantMessage.content.ilike("%POLICY:%")
                )
            )
        )
        payload = [
            {
                "id": n.id,
                "role": n.role,
                "content": n.content,
                "created_at": n.created_at,
            }
            for n in notes
        ]
        for n in notes:
            session.delete(n)
        session.commit()
    logger.info("policy_clear: cleared=%d", len(payload))
    return payload


@app.post("/policy/clear")
def policy_clear_post():
    payload = _policy_clear_impl()
    return JSONResponse({"cleared": len(payload), "preview": payload[:3]})


@app.get("/policy/clear")
def policy_clear_get():
    _policy_clear_impl()
    return RedirectResponse("/", status_code=303)
