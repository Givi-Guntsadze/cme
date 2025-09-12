from __future__ import annotations
import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from .db import create_db_and_tables, get_session, purge_seed_activities
from .models import User, Claim, AssistantMessage
from .planner import build_plan, build_plan_with_policy
from .parser import parse_message
from .ingest import ingest_psychiatry_online_ai, safe_json_loads


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_db_and_tables()
    try:
        purge_seed_activities()
    except Exception:
        pass
    yield
    # Shutdown: nothing for now


app = FastAPI(title="CME/MOC POC", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
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
                    "is_dev": bool(
                        os.getenv("ENV") in ("dev", "development")
                        or str(os.getenv("DEBUG")) in ("1", "true", "True")
                    ),
                },
            )

        total_claimed = sum(
            c.credits
            for c in session.exec(select(Claim).where(Claim.user_id == user.id))
        )
        user.remaining_credits = max(user.target_credits - total_claimed, 0.0)
        session.add(user)
        session.commit()

        msgs = list(
            session.exec(
                select(AssistantMessage).where(AssistantMessage.user_id == user.id)
            )
        )
        # Look for the latest policy message
        policy = None
        for m in reversed(msgs):
            if isinstance(m.content, str) and m.content.startswith("POLICY:"):
                try:
                    policy = json.loads(m.content.removeprefix("POLICY:"))
                except Exception:
                    policy = None
                break

        if policy:
            chosen, _, _, _ = build_plan_with_policy(user, session, policy)
        else:
            chosen, _, _, _ = build_plan(user, session)

        plan = [
            {
                "title": a.title,
                "provider": a.provider,
                "credits": a.credits,
                "cost": a.cost_usd,
                "modality": a.modality,
                "city": a.city,
                "url": a.url,
                "summary": a.summary,
                "source": a.source,
            }
            for a in chosen
        ]
        claims = list(session.exec(select(Claim).where(Claim.user_id == user.id)))
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "plan": plan,
                "claims": claims,
                "assistant_messages": msgs,
                "is_dev": bool(
                    os.getenv("ENV") in ("dev", "development")
                    or str(os.getenv("DEBUG")) in ("1", "true", "True")
                ),
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
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "plan": [],
            "claims": claims,
            "assistant_messages": msgs,
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
            if len(result) == 3:
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
        # Build current plan snapshot using same policy logic as home
        msgs = list(
            session.exec(
                select(AssistantMessage).where(AssistantMessage.user_id == user.id)
            )
        )
        policy = None
        for m in reversed(msgs):
            if isinstance(m.content, str) and m.content.startswith("POLICY:"):
                try:
                    policy = json.loads(m.content.removeprefix("POLICY:"))
                except Exception:
                    policy = None
                break
        if policy:
            chosen, total_credits, total_cost, days_used = build_plan_with_policy(
                user, session, policy
            )
        else:
            chosen, total_credits, total_cost, days_used = build_plan(user, session)
        plan = [
            {
                "title": a.title,
                "provider": a.provider,
                "credits": a.credits,
                "cost": a.cost_usd,
                "modality": a.modality,
                "city": a.city,
            }
            for a in chosen
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
        }

    sys = (
        "Extract updated preference values from the user's reply. Return STRICT JSON with keys: "
        "budget_usd (number|null), days_off (integer|null), allow_live (boolean|null), city (string|null), "
        "specialty (string|null), target_credits (number|null). If a value is not provided, set it to null."
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
        session.add(
            AssistantMessage(user_id=user.id, content="POLICY:" + json.dumps(policy))
        )
        # brief echo message
        summary = (
            "Applied policy: avoid="
            + ", ".join(policy["avoid_terms"])
            + "; prefer="
            + ", ".join(policy["prefer_topics"])
            + f"; diversity_weight={policy['diversity_weight']}"
        )
        session.add(AssistantMessage(user_id=user.id, content=summary))
        session.commit()

    return RedirectResponse("/?policy=1", status_code=303)


@app.get("/chat", response_class=HTMLResponse)
def chat(request: Request):
    with get_session() as session:
        user = session.exec(select(User)).first()
        msgs = (
            list(
                session.exec(
                    select(AssistantMessage).where(AssistantMessage.user_id == user.id)
                )
            )
            if user
            else []
        )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "chat_only": True,
            "assistant_messages": msgs,
            "plan": [],
            "claims": [],
            "is_dev": bool(
                os.getenv("ENV") in ("dev", "development")
                or str(os.getenv("DEBUG")) in ("1", "true", "True")
            ),
        },
    )


@app.post("/chat/send")
def chat_send(text: str = Form(...)):
    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse({"error": "missing OPENAI_API_KEY"}, status_code=400)
    from openai import OpenAI

    client = OpenAI()
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return JSONResponse({"error": "no_user"}, status_code=400)
        # Save user message
        session.add(AssistantMessage(user_id=user.id, role="user", content=text[:4000]))
        # Build snapshot
        chosen, total_credits, total_cost, days_used = build_plan(user, session)
        plan = [
            {
                "title": a.title,
                "provider": a.provider,
                "credits": a.credits,
                "cost": a.cost_usd,
                "modality": a.modality,
                "city": a.city,
            }
            for a in chosen[:10]
        ]
        snapshot = {
            "user": {
                "budget_usd": user.budget_usd,
                "days_off": user.days_off,
                "target_credits": user.target_credits,
                "remaining_credits": user.remaining_credits,
                "allow_live": user.allow_live,
            },
            "plan": plan,
        }
    sys = (
        "You are a concise CME planner assistant. Explain reasons, ask at most 2 clarifying questions. "
        "If the user expresses preferences (budget, days_off, allow_live, city, specialty, target_credits), "
        "reply with a short confirmation and a minimal JSON patch under key 'patch'."
    )
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(snapshot)},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    content = getattr(resp, "output_text", "") or ""
    data = safe_json_loads(content) or {}
    message_text = content
    patch = data.get("patch") if isinstance(data, dict) else None

    with get_session() as session:
        user = session.exec(select(User)).first()
        if patch and isinstance(patch, dict):
            # Apply simple patches
            try:
                if "budget_usd" in patch and patch["budget_usd"] is not None:
                    user.budget_usd = float(patch["budget_usd"])  # type: ignore
                if "days_off" in patch and patch["days_off"] is not None:
                    user.days_off = int(patch["days_off"])  # type: ignore
                if "allow_live" in patch and patch["allow_live"] is not None:
                    user.allow_live = bool(patch["allow_live"])  # type: ignore
                if "city" in patch:
                    user.city = patch["city"] or None  # type: ignore
                if "specialty" in patch:
                    user.specialty = patch["specialty"] or user.specialty  # type: ignore
                if "target_credits" in patch and patch["target_credits"] is not None:
                    user.target_credits = float(patch["target_credits"])  # type: ignore
                session.add(user)
                session.commit()
            except Exception:
                pass
        # Save assistant message
        session.add(
            AssistantMessage(
                user_id=user.id, role="assistant", content=message_text[:4000]
            )
        )
        session.commit()
    return RedirectResponse("/chat", status_code=303)


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
        session.commit()
    return RedirectResponse("/", status_code=303)


@app.get("/health")
def health():
    return {"status": "ok"}
