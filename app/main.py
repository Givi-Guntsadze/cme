from __future__ import annotations
import os
import json
import logging
import re
import asyncio
from textwrap import dedent
from types import SimpleNamespace
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from datetime import date, datetime

from .db import create_db_and_tables, get_session
from .models import (
    User,
    Claim,
    AssistantMessage,
    Activity,
    CompletedActivity,
)
from .requirements import (
    load_abpn_psychiatry_requirements,
    validate_against_requirements,
)
from .parser import parse_message
from .ingest import ingest_psychiatry_online_ai, safe_json_loads
from openai import OpenAI
from .services.plan import (
    PlanManager,
    active_policy_status,
    apply_policy_payloads,
    clear_policies,
    load_policy_bundle,
    policy_for_mode,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_db_and_tables()
    yield
    # Shutdown: nothing for now


app = FastAPI(title="CME/MOC POC", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
DEFAULT_ASSISTANT_MODEL = os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o-mini")

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
    run = plan_manager.ensure_plan(user, "variety", policy_bundle)
    plan_entries, _, _ = plan_manager.serialize_plan(run, user)
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
              "variety"?: OBJECT
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
        response = openai_client.chat.completions.create(
            model=DEFAULT_ASSISTANT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": control_system},
                {"role": "user", "content": json.dumps(user_payload, default=str)},
            ],
        )
    except Exception:
        logging.exception("control derivation failed")
        return ([], [])

    text = (response.choices[0].message.content or "").strip()
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
    plan_mode = "variety"

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
                "is_dev": bool(
                    os.getenv("ENV") in ("dev", "development")
                    or str(os.getenv("DEBUG")) in ("1", "true", "True")
                ),
            },
        )


@app.get("/fragment/plan", response_class=HTMLResponse)
def plan_fragment(request: Request, plan_mode: str = Query("variety")):
    plan_mode = "variety"
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
            "plan_mode": request.query_params.get("plan_mode", "variety"),
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
    has_google_key = bool(os.getenv("GOOGLE_API_KEY"))
    has_google_cx = bool(os.getenv("GOOGLE_CSE_ID"))
    logger.info(
        "ingest called. env GOOGLE_API_KEY=%s GOOGLE_CSE_ID=%s",
        has_google_key,
        has_google_cx,
    )
    try:
        # threshold from env (optional)
        min_results = int(os.getenv("INGEST_MIN_RESULTS", "12"))
        debug = False
        if request is not None:
            qp = dict(request.query_params)
            debug = str(qp.get("debug", "0")) in ("1", "true", "True")
        result = await ingest_psychiatry_online_ai(count=min_results, debug=debug)
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
    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse({"error": "missing OPENAI_API_KEY"}, status_code=400)
    from openai import OpenAI

    client = OpenAI()
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return JSONResponse({"error": "no_user"}, status_code=400)
        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        run = plan_manager.ensure_plan(user, "variety", policy_bundle)
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

    resp = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(u)},
        ],
    )
    text = getattr(resp, "output_text", "") or ""

    with get_session() as session:
        user = session.exec(select(User)).first()
        msg = AssistantMessage(user_id=user.id, content=text.strip()[:4000])
        session.add(msg)
        session.commit()

    return JSONResponse({"message": text})


@app.post("/assist/reply")
def assist_reply(answer: str = Form(...)):
    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse({"error": "missing OPENAI_API_KEY"}, status_code=400)

    from openai import OpenAI

    client = OpenAI()
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

    resp = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user_msg)},
        ],
    )
    text = getattr(resp, "output_text", "") or ""
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
    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse({"error": "missing OPENAI_API_KEY"}, status_code=400)
    from openai import OpenAI

    client = OpenAI()

    # Ask model to produce a compact policy JSON
    sys = (
        "You convert natural language planning instructions into a compact JSON policy. "
        "Output ONLY JSON with these optional keys: "
        "avoid_terms (array of strings), prefer_topics (array of strings), diversity_weight (number 0..2), "
        "max_per_activity_fraction (number 0..1), prefer_live_override (true|false|null), budget_tolerance (number 0..0.3)."
    )
    try:
        resp = client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": command},
            ],
        )
        text = getattr(resp, "output_text", "") or ""
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
        apply_policy_payloads([json.dumps(policy)], user, session)

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
        keywords_trigger = {
            "discover",
            "add more",
            "find more",
            "update plan",
            "add activities",
            "more activities",
        }
        normalized_text = text.strip().lower()
        should_ingest = any(k in normalized_text for k in keywords_trigger)

        with get_session() as session:
            user = session.exec(select(User)).first()
            if not user:
                user = User()
                session.add(user)
                session.commit()
                session.refresh(user)

            plan_manager = PlanManager(session)
            user_message = AssistantMessage(
                user_id=user.id, role="user", content=text.strip()[:4000]
            )
            session.add(user_message)

            pending_notes: list[str] = []
            note_text = _maybe_log_claim_from_text(user, text, session)
            if note_text:
                pending_notes.append(note_text)
            note_text = _maybe_complete_activity_from_text(user, text, session)
            if note_text:
                pending_notes.append(note_text)

            if normalized_text in {"clear policy", "clear policies", "reset policy"}:
                cleared = clear_policies(session, user)
                response_text = (
                    "Policy cleared. Planner reset."
                    if cleared
                    else "No active policy to clear."
                )
                confirm = AssistantMessage(
                    user_id=user.id, role="assistant", content=response_text
                )
                session.add(confirm)
                plan_manager = PlanManager(session)
                policy_bundle = load_policy_bundle(session, user)
                summary_text, variety_count, cheapest_count = (
                    plan_manager.summarize_modes(
                        user, policy_bundle, force_refresh=True
                    )
                )
                summary_msg = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=summary_text[:4000],
                )
                plan_note = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=(
                        f"Plan updated ({variety_count} variety items, "
                        f"{cheapest_count} cheapest)."
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
                return templates.TemplateResponse(
                    "_chat_messages_append.html",
                    {"request": request, "messages": messages_to_render},
                )

            snapshot = _state_snapshot(session, user)

            system_prompt = dedent(
                """
                You are a CME planning assistant for a board-certified US physician.

                Behavior:

                Be concise and helpful.

                Explain why the top plan activities were selected (cost per credit, budget, days_off, allow_live).

                Ask at most 1–2 clarifying questions only when genuinely useful.

                When the user expresses preference updates (e.g., “online only”, “budget 200”, “days off 1”, “target 30”):

                Acknowledge briefly in natural language.

                Append a single machine-readable line:
                PATCH: {"only_changed_fields_here": "..."}
                • Only include keys among: budget_usd, days_off, allow_live, city, specialty,
                  target_credits, professional_stage, residency_completion_year, memberships
                • Use correct JSON types (numbers for budgets/targets, booleans for allow_live, strings for city/specialty)
                • memberships should be an array of strings when provided
                • No backticks, no code fences, no extra text on that line

                If the user indicates remaining credits explicitly (e.g., “remaining 17”
                or “I have 12 left”), still reply normally; no patch is needed for this.
                (The system will adjust remaining by logging a claim.)

                If you suggest plan policies, you may optionally add another machine line:
                POLICY: {"plan_mode":"variety","notes":"avoid subscriptions",
                "weights":{"diversity":0.4}}
                Keep policies compact and only if you actually changed them.

                Never lead with raw control JSON. First provide the human explanation, then append control lines if needed.
                Do not repeat the PATCH or POLICY content in prose. Do not wrap control lines in code fences.
                """
            ).strip()

            user_payload = f"User said: {text}\n\nCurrent state snapshot:\n{json.dumps(snapshot, default=str)}"

            model_name = DEFAULT_ASSISTANT_MODEL
            if os.getenv("ENABLE_GPT5") and os.getenv("ENABLE_GPT5").lower() in {
                "1",
                "true",
                "yes",
            }:
                model_name = "gpt-5"

            response = openai_client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
            )

            reply = (response.choices[0].message.content or "").strip()

            patch_payloads: list[str] = []
            policy_payloads: list[str] = []
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
                visible_lines.append(line)

            visible_reply = "\n".join(visible_lines).strip()
            if not visible_reply:
                visible_reply = "Updated the plan to match your request."

            if not patch_payloads and not policy_payloads:
                derived_patch, derived_policy = _derive_controls(text, snapshot)
                patch_payloads.extend(derived_patch)
                policy_payloads.extend(derived_policy)

            assistant_message = AssistantMessage(
                user_id=user.id, role="assistant", content=visible_reply[:4000]
            )
            session.add(assistant_message)

            extra_messages = _apply_patch_if_present(patch_payloads, user, session)
            apply_policy_payloads(policy_payloads, user, session)

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
                            should_ingest = True
                            confirm = AssistantMessage(
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
                        role="assistant", content=note
                    )
                    session.add(adjustment_message)
                    session.commit()
                    confirmations.append(adjustment_message)
            except Exception:
                logging.exception("Remaining adjustment failed")

            ingest_success = False
            if should_ingest:
                try:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        new_loop = asyncio.new_event_loop()
                        try:
                            asyncio.set_event_loop(new_loop)
                            new_loop.run_until_complete(
                                ingest_psychiatry_online_ai(count=10, debug=False)
                            )
                        finally:
                            asyncio.set_event_loop(loop)
                            new_loop.close()
                    else:
                        asyncio.run(ingest_psychiatry_online_ai(count=10, debug=False))
                    ingest_success = True
                except Exception:
                    logging.exception("ingest during chat failed")
                if ingest_success:
                    PlanManager.invalidate_user_plans(
                        session, user.id, reason="ingest_refresh"
                    )

            policy_bundle = load_policy_bundle(session, user)
            plan_summary, variety_count, cheapest_count = plan_manager.summarize_modes(
                user, policy_bundle, force_refresh=ingest_success
            )

            if should_ingest:
                assistant_message.content = plan_summary[:4000]
            else:
                assistant_message.content = (
                    (assistant_message.content or "").rstrip() + "\n\n" + plan_summary
                )[:4000]

            plan_note_needed = bool(
                confirmations or extra_messages or patch_payloads or should_ingest
            )
            if plan_note_needed:
                plan_note = AssistantMessage(
                    user_id=user.id,
                    role="assistant",
                    content=(
                        f"Plan updated ({variety_count} variety items, "
                        f"{cheapest_count} cheapest)."
                    ),
                )
                session.add(plan_note)
                confirmations.append(plan_note)

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

        return templates.TemplateResponse(
            "_chat_messages_append.html",
            {"request": request, "messages": messages_to_render},
        )
    except Exception as exc:
        logging.exception("chat_send failed")
        return HTMLResponse(
            f"<div class='bubble error'>Assistant error: {str(exc)[:200]}</div>",
            status_code=500,
        )


@app.post("/plan/remove")
def plan_remove(title: str = Form(...)):
    # Append the title to policy.remove_titles and save as a new POLICY message
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return RedirectResponse("/", status_code=303)
        msgs = list(
            session.exec(
                select(AssistantMessage).where(AssistantMessage.user_id == user.id)
            )
        )
        policy = {}
        for m in reversed(msgs):
            if isinstance(m.content, str) and m.content.startswith("POLICY:"):
                try:
                    policy = json.loads(m.content.removeprefix("POLICY:")) or {}
                except Exception:
                    policy = {}
                break
        remove_list = list(policy.get("remove_titles") or [])
        if title not in remove_list:
            remove_list.append(title)
        policy["remove_titles"] = remove_list
        session.add(
            AssistantMessage(user_id=user.id, content="POLICY:" + json.dumps(policy))
        )
        session.add(AssistantMessage(user_id=user.id, content=f"Removed: {title}"))
        PlanManager.invalidate_user_plans(session, user.id, reason="plan_remove")
        session.commit()
    return RedirectResponse("/", status_code=303)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/requirements/sync")
def requirements_sync():
    return JSONResponse(
        {"status": "not_implemented", "detail": "RAG sync scaffolding"},
        status_code=501,
    )


@app.post("/policy/clear", response_class=HTMLResponse)
def policy_clear(request: Request):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return HTMLResponse("", status_code=204)
        cleared = clear_policies(session, user)
        message = (
            "Policy cleared. Planner reset."
            if cleared
            else "No active policy to clear."
        )
        note = AssistantMessage(user_id=user.id, role="assistant", content=message)
        session.add(note)
        plan_manager = PlanManager(session)
        policy_bundle = load_policy_bundle(session, user)
        summary_text, variety_count, cheapest_count = plan_manager.summarize_modes(
            user, policy_bundle, force_refresh=True
        )
        summary_msg = AssistantMessage(
            user_id=user.id,
            role="assistant",
            content=summary_text[:4000],
        )
        plan_note = AssistantMessage(
            user_id=user.id,
            role="assistant",
            content=(
                f"Plan updated ({variety_count} variety items, {cheapest_count} cheapest)."
            ),
        )
        session.add(summary_msg)
        session.add(plan_note)
        session.commit()

    messages = [
        SimpleNamespace(role=note.role, content=note.content),
        SimpleNamespace(role=summary_msg.role, content=summary_msg.content),
        SimpleNamespace(role=plan_note.role, content=plan_note.content),
    ]
    return templates.TemplateResponse(
        "_chat_messages_append.html",
        {"request": request, "messages": messages},
    )
