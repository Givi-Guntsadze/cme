from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # Load .env file before any other imports that need env vars

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

from .catalog.router import router as catalog_router
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
    reenrich_activities,
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


def _compose_requirement_gap_message(
    requirements: dict[str, object] | None
) -> tuple[str, str] | tuple[None, None]:
    if not requirements:
        return (None, None)
    pending_flags = requirements.get("pending") or {}
    pending_values = requirements.get("values") or {}
    if not isinstance(pending_flags, dict):
        pending_flags = {}
    if not isinstance(pending_values, dict):
        pending_values = {}

    gaps: list[str] = []
    focus_terms: list[str] = []

    if pending_flags.get("sa_cme"):
        needed = pending_values.get("sa_cme")
        try:
            needed = float(needed)
        except Exception:
            needed = None
        if needed and needed > 0.1:
            gaps.append(f"{needed:.1f} SA-CME credits")
            focus_terms.append("SA-CME")

    if pending_flags.get("patient_safety"):
        gaps.append("a patient safety activity")
        focus_terms.append("patient safety")

    pip_needed = pending_values.get("pip")
    pip_flag = pending_flags.get("pip")
    if pip_flag or (isinstance(pip_needed, (int, float)) and pip_needed):
        try:
            pip_count = int(float(pip_needed))
        except Exception:
            pip_count = 1 if pip_flag else 0
        if pip_flag or pip_count > 0:
            noun = "project" if pip_count == 1 else "projects"
            clause = f"{pip_count} PIP {noun}" if pip_count else "a PIP project"
            gaps.append(clause)
            focus_terms.append("PIP")

    if not gaps:
        return (None, None)

    gap_text = ", ".join(gaps[:-1]) + (" and " + gaps[-1] if len(gaps) > 1 else gaps[0])
    prompt = (
        f"We still need {gap_text}. What should I prioritize next "
        "— a national conference block, on-demand modules, or something else?"
    )
    return (prompt, ", ".join(focus_terms) or gap_text)


def _maybe_nudge_requirements(
    session, user: User, requirements: dict[str, object] | None
) -> bool:
    prompt, fingerprint = _compose_requirement_gap_message(requirements)
    if not prompt:
        return False

    recent = list(
        session.exec(
            select(AssistantMessage)
            .where(AssistantMessage.user_id == user.id)
            .order_by(AssistantMessage.created_at.desc())
            .limit(8)
        )
    )
    for msg in recent:
        content = (msg.content or "").strip()
        if not content:
            continue
        if fingerprint and fingerprint in content:
            return False
        if content == prompt:
            return False

    session.add(AssistantMessage(user_id=user.id, role="assistant", content=prompt))
    return True


def _queue_substitute_prompt(session, user: User, activity: Activity) -> None:
    # SILENT: Only send the internal signal to trigger the search, no chat message
    session.add(
        AssistantMessage(
            user_id=user.id,
            role="assistant",
            content=f"INTERNAL:SUBSTITUTE_REQUEST:{activity.id}",
        )
    )


def _queue_plan_reset_prompt(session, user: User, titles: list[str]) -> None:
    cleaned = [t for t in titles if t]
    preview = ", ".join(cleaned[:3])
    if len(cleaned) > 3:
        preview += ", and others"
    if preview:
        prefix = f"Cleared the previous recommendations ({preview})."
    else:
        prefix = "Cleared the previous recommendations."
    prompt = (
        f"{prefix} What should I scout next — national conferences, on-demand bundles, "
        "or something targeted like patient safety or SA-CME modules?"
    )
    session.add(AssistantMessage(user_id=user.id, role="assistant", content=prompt))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_db_and_tables()
    # Start the background scheduler for periodic source refresh
    from .services.scheduler import start_scheduler, stop_scheduler
    await start_scheduler()
    yield
    # Shutdown
    await stop_scheduler()


app = FastAPI(title="CME/MOC POC", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")
app.include_router(catalog_router)


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


# Patterns for extracting activity updates from user messages
ACTIVITY_UPDATE_PATTERNS = [
    # "X costs $Y" or "X will cost me $Y"
    re.compile(
        r"(?:the\s+)?(?P<title>[^$]+?)\s+(?:costs?|will\s+cost(?:\s+me)?|is|was)\s+\$(?P<cost>\d+(?:\.\d+)?)",
        re.I,
    ),
    # "$Y for X"
    re.compile(
        r"\$(?P<cost>\d+(?:\.\d+)?)\s+(?:for|to\s+(?:take|attend))\s+(?:the\s+)?(?P<title>[^.;!?]+)",
        re.I,
    ),
    # "update X to $Y" or "set X cost to $Y"
    re.compile(
        r"(?:update|set|change)\s+(?:the\s+)?(?P<title>[^$]+?)\s+(?:cost\s+)?(?:to|=)\s+\$(?P<cost>\d+(?:\.\d+)?)",
        re.I,
    ),
]

ELIGIBILITY_PATTERNS = [
    # "I am eligible for X" or "I'm eligible for X"
    re.compile(
        r"i(?:'m|\s+am)\s+eligible\s+(?:for\s+)?(?:the\s+)?(?P<title>[^.;!?]+)",
        re.I,
    ),
    # "X and I am eligible"
    re.compile(
        r"(?P<title>[^$]+?)\s+and\s+i(?:'m|\s+am)\s+eligible",
        re.I,
    ),
    # "eligible for X"
    re.compile(
        r"eligible\s+for\s+(?:the\s+)?(?P<title>[^.;!?]+)",
        re.I,
    ),
    # "change/set status to eligible for X" or "mark X as eligible"
    re.compile(
        r"(?:change|set|mark|update)\s+(?:.*?)\s+eligible\s+(?:for\s+)?(?P<title>[^.!?;]+)",
        re.I,
    ),
]

# Patterns for detecting discovery/search queries
DISCOVERY_PATTERNS = [
    # "find me X", "find X courses", "find X activities"
    re.compile(r"find\s+(?:me\s+)?(?P<query>.+?)(?:\s+courses?|\s+activities?|\s+cme)?$", re.I),
    # "search for X", "look for X"
    re.compile(r"(?:search|look)\s+for\s+(?P<query>.+?)(?:\s+courses?|\s+activities?)?$", re.I),
    # "show me X", "show X courses"
    re.compile(r"show\s+(?:me\s+)?(?P<query>.+?)(?:\s+courses?|\s+activities?|\s+cme)?$", re.I),
    # "what X courses/activities are available"
    re.compile(r"what\s+(?P<query>.+?)\s+(?:courses?|activities?|cme)\s+(?:are\s+)?(?:available|there)", re.I),
    # "any X courses", "any courses on X"
    re.compile(r"any\s+(?P<query>.+?)\s+(?:courses?|activities?|cme)", re.I),
    re.compile(r"any\s+(?:courses?|activities?|cme)\s+(?:on|about|for)\s+(?P<query>.+?)$", re.I),
]


def _extract_discovery_query(text: str) -> str | None:
    """
    Extract search query from user text if it looks like a discovery request.
    Returns the query string or None if not a discovery request.
    """
    if not text:
        return None
    
    for pattern in DISCOVERY_PATTERNS:
        match = pattern.search(text)
        if match:
            query = match.group("query").strip()
            # Clean up common trailing words
            query = re.sub(r"\s*(?:please|thanks?|thank\s+you)$", "", query, flags=re.I)
            if query and len(query) > 2:
                return query
    
    return None


def _extract_activity_update(text: str) -> dict | None:
    """
    Parse user text for activity update commands.
    Returns a dict with keys: title, cost (optional), eligible (optional)
    """
    if not text:
        return None
    
    result = {
        "title": None,
        "cost": None,
        "eligible": None,
    }
    
    # Check for cost updates
    for pattern in ACTIVITY_UPDATE_PATTERNS:
        match = pattern.search(text)
        if match:
            title = match.group("title")
            cost = match.group("cost")
            if title and cost:
                result["title"] = _clean_title_fragment(title)
                try:
                    result["cost"] = float(cost)
                except (TypeError, ValueError):
                    pass
                break
    
    # Check for eligibility updates
    for pattern in ELIGIBILITY_PATTERNS:
        match = pattern.search(text)
        if match:
            title = match.group("title")
            if title:
                title_clean = _clean_title_fragment(title)
                if result["title"] is None:
                    result["title"] = title_clean
                elif title_clean.lower() in result["title"].lower() or result["title"].lower() in title_clean.lower():
                    # Same or overlapping title - merge
                    pass
                result["eligible"] = True
                break
    
    # Also check for simple eligibility mention alongside the title
    if result["title"] and ("eligible" in text.lower() or "i am eligible" in text.lower()):
        result["eligible"] = True
    
    if result["title"]:
        return result
    return None


from .services.discovery import (
    find_activity_by_title, 
    _title_tokens, 
    _is_candidate_match, 
    _match_score,
    MIN_MATCH_SCORE
)
from .services.tools import TOOLS_SCHEMA, execute_tool_call


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


# _match_score is now imported from services.discovery


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


# find_activity_by_title is now imported from services.discovery


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

        pip_status = {"complete": False, "reason": None}
        for claim in claims:
            topic = (claim.topic or "").lower()
            if topic in {"pip_complete", "pip completion"}:
                pip_status = {
                    "complete": True,
                    "reason": "Documented pip_complete claim",
                }
                break

        validation = validate_full_cc(
            user, claims, requirements_data, pip_status=pip_status
        )

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
        force_refresh = request.query_params.get("refresh", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        response, _ = _prepare_plan_fragment(
            request,
            session,
            user,
            plan_mode=plan_mode,
            force_refresh=force_refresh,
            reason="fragment_refresh" if force_refresh else None,
        )
        return response


def _prepare_plan_fragment(
    request: Request,
    session,
    user: User,
    *,
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
    response = templates.TemplateResponse(
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
    return response, plan_requirements


def _refresh_plan_fragment_response(
    request: Request,
    session,
    user: User,
    plan_mode: str = "balanced",
    force_refresh: bool = True,
    reason: str | None = None,
):
    response, _ = _prepare_plan_fragment(
        request,
        session,
        user,
        plan_mode=plan_mode,
        force_refresh=force_refresh,
        reason=reason,
    )
    return response


@app.post("/plan/accept/{activity_id}", response_class=HTMLResponse)
@app.post("/plan/commit/{activity_id}", response_class=HTMLResponse)
def plan_accept(request: Request, activity_id: int):
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
        response, plan_requirements = _prepare_plan_fragment(
            request,
            session,
            user,
            plan_mode="balanced",
            force_refresh=True,
            reason="commit_toggle",
        )
        chat_refresh = _maybe_nudge_requirements(session, user, plan_requirements)
        if chat_refresh:
            session.commit()
            response.headers["HX-Trigger"] = json.dumps({"chat-refresh": True})
        return response


@app.post("/plan/uncommit/{activity_id}", response_class=HTMLResponse)
def plan_uncommit(request: Request, activity_id: int):
    return _handle_plan_reject(request, activity_id, find_substitute=False)


@app.post("/plan/reject/{activity_id}", response_class=HTMLResponse)
def plan_reject(request: Request, activity_id: int):
    return _handle_plan_reject(request, activity_id, find_substitute=False)


@app.post("/plan/reject_with_substitute/{activity_id}", response_class=HTMLResponse)
def plan_reject_with_substitute(request: Request, activity_id: int):
    return _handle_plan_reject(request, activity_id, find_substitute=True)


def _handle_plan_reject(
    request: Request, activity_id: int, *, find_substitute: bool
) -> HTMLResponse:
    chat_refresh = False
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
        activity = session.get(Activity, activity_id)
        if item.committed:
            item.committed = False
            session.add(item)
        if activity and activity.title:
            payload = json.dumps({"remove_titles": [activity.title]})
            apply_policy_payloads(
                [payload],
                user,
                session,
                invalidate=False,
                record_message=False,
            )
        PlanManager.invalidate_user_plans(session, user.id, reason="plan_reject")
        if find_substitute and activity:
             # SILENT SIGNAL: Triggers search without chat message
            _queue_substitute_prompt(session, user, activity)
            chat_refresh = True
        session.commit()
    with get_session() as session:
        user = session.exec(select(User)).first()
        response, plan_requirements = _prepare_plan_fragment(
            request,
            session,
            user,
            plan_mode="balanced",
            force_refresh=True,
            reason="plan_reject",
        )
        # REMOVED: Don't nudge via chat when user clicks buttons
        # nudged = _maybe_nudge_requirements(session, user, plan_requirements)
        # if nudged:
        #     session.commit()
        #     chat_refresh = True
        # if chat_refresh:
        #     response.headers["HX-Trigger"] = json.dumps({"chat-refresh": True})
        return response


@app.post("/plan/accept_all", response_class=HTMLResponse)
def plan_accept_all(request: Request):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return HTMLResponse("", status_code=204)
        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
        items = list(
            session.exec(
                select(PlanItem)
                .where(PlanItem.plan_run_id == run.id)
                .order_by(PlanItem.position.asc())
            )
        )
        changed = False
        for item in items:
            if not item.committed:
                item.committed = True
                session.add(item)
                changed = True
        if changed:
            PlanManager.invalidate_user_plans(session, user.id, reason="accept_all")
        session.commit()
    with get_session() as session:
        user = session.exec(select(User)).first()
        response, plan_requirements = _prepare_plan_fragment(
            request,
            session,
            user,
            plan_mode="balanced",
            force_refresh=True,
            reason="accept_all",
        )
        # REMOVED: Don't nudge via chat when user clicks buttons
        # nudged = _maybe_nudge_requirements(session, user, plan_requirements)
        # if nudged:
        #     session.commit()
        #     response.headers["HX-Trigger"] = json.dumps({"chat-refresh": True})
        return response


@app.post("/plan/reject_all", response_class=HTMLResponse)
def plan_reject_all(request: Request):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return HTMLResponse("", status_code=204)
        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        run = plan_manager.ensure_plan(user, "balanced", policy_bundle)
        items = list(
            session.exec(
                select(PlanItem)
                .where(PlanItem.plan_run_id == run.id)
                .order_by(PlanItem.position.asc())
            )
        )
        titles: list[str] = []
        for item in items:
            activity = session.get(Activity, item.activity_id)
            if activity and activity.title:
                titles.append(activity.title)
            if item.committed:
                item.committed = False
                session.add(item)
        if titles:
            payload = json.dumps({"remove_titles": titles})
            apply_policy_payloads(
                [payload],
                user,
                session,
                invalidate=False,
                record_message=False,
            )
        PlanManager.invalidate_user_plans(session, user.id, reason="reject_all")
        # REMOVED: Don't add chat messages when user clicks buttons
        # _queue_plan_reset_prompt(session, user, titles)
        session.commit()
    with get_session() as session:
        user = session.exec(select(User)).first()
        response, plan_requirements = _prepare_plan_fragment(
            request,
            session,
            user,
            plan_mode="balanced",
            force_refresh=True,
            reason="reject_all",
        )
        # REMOVED: Don't nudge via chat when user clicks buttons  
        # nudged = _maybe_nudge_requirements(session, user, plan_requirements)
        # if nudged:
        #     session.commit()
        # response.headers["HX-Trigger"] = json.dumps({"chat-refresh": True})
        return response


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


@app.api_route("/reenrich", methods=["GET", "POST"])
async def reenrich_endpoint(request: Request = None):
    """Re-enrich existing activities using Perplexity to fill in missing data."""
    logger = logging.getLogger(__name__)
    logger.info("reenrich called")
    try:
        limit = 50
        if request is not None:
            qp = dict(request.query_params)
            try:
                limit = int(qp.get("limit", "50"))
            except (TypeError, ValueError):
                limit = 50
        
        result = await asyncio.to_thread(reenrich_activities, limit=limit)
        
        # If there are activities that still need manual input, post a message to the chat
        needs_manual = result.get("needs_manual", [])
        if needs_manual:
            with get_session() as session:
                user = session.exec(select(User)).first()
                if user:
                    # Build a helpful message for the user
                    lines = ["I couldn't find complete pricing or eligibility info for some activities:"]
                    for item in needs_manual[:5]:  # Limit to 5
                        missing_str = " and ".join(item["missing"])
                        lines.append(f"• **{item['title']}** (missing: {missing_str})")
                    lines.append("")
                    lines.append("You can update these by telling me, for example:")
                    lines.append('_"The Opioid CME Courses costs $125 and I am eligible."_')
                    
                    message = AssistantMessage(
                        user_id=user.id,
                        role="assistant",
                        content="\n".join(lines)
                    )
                    session.add(message)
                    session.commit()
        
        return JSONResponse(result)
    except Exception as e:
        logger.exception("reenrich failed")
        return JSONResponse(
            {"error": "reenrich_failed", "detail": str(e)}, status_code=500
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

    with get_session() as session:
        # 1. User Setup
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

        # 2. Log User Message
        user_message = AssistantMessage(
            user_id=user.id, role="user", content=text.strip()[:4000]
        )
        session.add(user_message)
        session.commit()

        normalized_text = text.strip().lower()

        # --- LEGACY FAST PATHS (Slash Commands) ---

        # "Keep this plan" command
        if "keep this plan" in normalized_text and "policy" not in normalized_text:
            policy_bundle = load_policy_bundle(session, user)
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

        # "Clear policy" command
        if "clear policy" in normalized_text or "reset policy" in normalized_text:
            apply_policy_payloads(
                [json.dumps({"clear_all": True})], user, session, invalidate=True
            )
            cleared_msg = AssistantMessage(
                user_id=user.id,
                role="assistant",
                content="I've cleared your planning policies (topics, filters). We're starting fresh.",
            )
            session.add(cleared_msg)
            session.commit()
            return templates.TemplateResponse(
                "_chat_messages_append.html",
                {
                    "request": request,
                    "messages": [
                        SimpleNamespace(
                            role=user_message.role, content=user_message.content
                        ),
                        SimpleNamespace(
                            role=cleared_msg.role, content=cleared_msg.content
                        ),
                    ],
                },
            )

        # --- AGENTIC LOOP ---
        state = _state_snapshot(session, user)
        system_prompt = build_system_prompt(state)

        # History
        history = session.exec(
            select(AssistantMessage)
            .where(AssistantMessage.user_id == user.id)
            .order_by(AssistantMessage.created_at.desc())
            .limit(10)
        ).all()
        history = list(reversed(history))

        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            role = m.role if m.role in ["user", "assistant"] else "assistant"
            messages.append({"role": role, "content": m.content})

        context_note = f"\n[System Context]: Current Plan Status: {json.dumps(state['plan_summary'])}"
        messages.append({"role": "user", "content": text + context_note})

        final_response_text = "I'm thinking..."
        tools_called = False

        # Agent Loop (Max 5 turns)
        for _ in range(5):
            completion = openai_client.chat.completions.create(
                model=DEFAULT_ASSISTANT_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )
            msg = completion.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                tools_called = True
                for tool_call in msg.tool_calls:
                    fname = tool_call.function.name
                    args_str = tool_call.function.arguments
                    logger.info(f"Tool Call: {fname}({args_str})")
                    try:
                        args = json.loads(args_str)
                        result = execute_tool_call(fname, args, session, user)
                    except Exception as e:
                        result = f"Error executing {fname}: {str(e)}"
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                final_response_text = msg.content
                break
        
        # Save Assistant Response
        final_msg = AssistantMessage(
            user_id=user.id,
            role="assistant",
            content=final_response_text or "Done.",
        )
        session.add(final_msg)
        session.commit()

        # Trigger plan refresh if tools were used
        headers = {}
        if tools_called:
            headers["HX-Trigger"] = "plan-refresh"

        # Build the response payload
        messages_to_render = [
            SimpleNamespace(role=user_message.role, content=user_message.content),
            SimpleNamespace(role=final_msg.role, content=final_msg.content),
        ]
        logger.info(f"chat_send: returning {len(messages_to_render)} messages")
        return templates.TemplateResponse(
            "_chat_messages_append.html",
            {
                "request": request,
                "messages": messages_to_render,
            },
            headers=headers
        )




def _state_snapshot(session, user: User) -> dict:
    """Collects context for the LLM."""
    pm = PlanManager(session)
    # Ensure plan exists to get valid summary
    try:
        run = pm.ensure_plan(user, "balanced", policy_bundle=None)
        items = session.exec(select(PlanItem).where(PlanItem.plan_run_id == run.id).order_by(PlanItem.position)).all()
        logger.info(f"_state_snapshot: Found {len(items)} plan items for run {run.id}")
        
        # Fetch activities for each item
        plan_summary = []
        for i in items:
            if i.activity_id:
                activity = session.get(Activity, i.activity_id)
                if activity:
                    status = 'Committed' if i.committed else 'Candidate'
                    plan_summary.append(f"{i.position}. {activity.title} (${activity.cost_usd}) [{status}]")
        
        logger.info(f"_state_snapshot: plan_summary has {len(plan_summary)} items")
    except Exception as e:
        logger.exception(f"_state_snapshot error: {e}")
        plan_summary = ["(No plan available)"]

    return {
        "user_name": user.name,
        "target_credits": user.target_credits,
        "remaining_credits": user.remaining_credits,
        "budget": user.budget_usd,
        "plan_summary": plan_summary,
        "days_off": user.days_off
    }

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


@app.post("/source/monitor")
def source_monitor(request: Request, url: str = Form(...)):
    """
    Add a URL to the monitored sources for Apify scraping.
    
    The domain is extracted from the URL and stored in ScrapeSource.
    """
    from urllib.parse import urlparse
    from .models import ScrapeSource
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        if not domain:
            return JSONResponse({"error": "Invalid URL"}, status_code=400)
        
        # Use full URL
        base_url = url.strip()
        
        with get_session() as session:
            # Check if already monitored
            existing = session.exec(
                select(ScrapeSource).where(ScrapeSource.domain == domain)
            ).first()
            
            if existing:
                return JSONResponse({
                    "status": "already_monitored",
                    "domain": domain,
                    "message": f"'{domain}' is already being monitored."
                })
            
            # Add new source
            new_source = ScrapeSource(
                domain=domain,
                url=base_url,
                enabled=True,
            )
            session.add(new_source)
            session.commit()
            
            return JSONResponse({
                "status": "success",
                "domain": domain,
                "message": f"Now monitoring '{domain}'. New activities will appear after the next scheduled crawl."
            })
    except Exception as e:
        logger.exception("Failed to add monitored source")
        return JSONResponse({"error": str(e)}, status_code=500)
