from __future__ import annotations
import os
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from .db import create_db_and_tables, get_session, purge_seed_activities
from .models import User, Claim, AssistantMessage
from .planner import build_plan
from .parser import parse_message
from .ingest import ingest_psychiatry_online_ai, safe_json_loads

app = FastAPI(title="CME/MOC POC")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    # Remove legacy seed rows
    try:
        purge_seed_activities()
    except Exception:
        pass


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
                },
            )

        total_claimed = sum(
            c.credits
            for c in session.exec(select(Claim).where(Claim.user_id == user.id))
        )
        user.remaining_credits = max(user.target_credits - total_claimed, 0.0)
        session.add(user)
        session.commit()

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
        msgs = list(
            session.exec(
                select(AssistantMessage).where(AssistantMessage.user_id == user.id)
            )
        )
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "user": user,
                "plan": plan,
                "claims": claims,
                "assistant_messages": msgs,
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
        },
    )


@app.api_route("/ingest", methods=["GET", "POST"])
def ingest():
    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse({"error": "missing OPENAI_API_KEY"}, status_code=400)
    try:
        min_results = int(os.getenv("INGEST_MIN_RESULTS", "12"))
        (
            total,
            used_fallback,
            g_added,
            ai_added,
            cand_count,
            extracted_count,
            google_ok,
        ) = ingest_psychiatry_online_ai(count=min_results)
        return JSONResponse(
            {
                "ingested": total,
                "google_inserted": g_added,
                "ai_inserted": ai_added,
                "used_fallback": used_fallback,
                "candidates": cand_count,
                "extracted": extracted_count,
                "google_ok": bool(google_ok),
            }
        )
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception:
        return JSONResponse({"error": "ingest_failed"}, status_code=500)


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
        # Build current plan snapshot
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
