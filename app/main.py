from __future__ import annotations
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select

from .db import create_db_and_tables, get_session
from .seed import seed_activities
from .models import User, Claim
from .planner import build_plan
from .parser import parse_message

app = FastAPI(title="CME/MOC POC")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    with get_session() as s:
        seed_activities(s)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    with get_session() as session:
        user = session.exec(select(User)).first()
        if not user:
            return templates.TemplateResponse(
                "index.html", {"request": request, "user": None, "plan": []}
            )
        # recompute remaining from claims
        total_claimed = sum(
            c.credits
            for c in session.exec(select(Claim).where(Claim.user_id == user.id))
        )
        user.remaining_credits = max(user.target_credits - total_claimed, 0.0)
        session.add(user)
        session.commit()
        plan, _, _, _ = _plan_for(user, session)
        return templates.TemplateResponse(
            "index.html", {"request": request, "user": user, "plan": plan}
        )


def _plan_for(user: User, session):
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
    return plan, total_credits, total_cost, days_used


@app.post("/setup")
def setup(
    name: str = Form("Doctor"),
    specialty: str = Form("Psychiatry"),
    city: str = Form(""),
    budget_usd: float = Form(...),
    days_off: int = Form(...),
    target_credits: float = Form(...),
    allow_live: bool = Form(False),
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
    return templates.TemplateResponse(
        "index.html", {"request": request, "user": user, "plan": [], "claims": claims}
    )
