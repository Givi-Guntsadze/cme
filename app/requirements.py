from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from .models import User, Claim, RequirementsSnapshot
from .db import get_session
from sqlalchemy import select as sa_select

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent / "config" / "abpn_psychiatry_requirements.json"


@dataclass
class RequirementRules:
    cycle_years: int = 0
    total_credits: int = 0
    annual_minimum: int = 0
    safety_or_ethics_min: int = 0
    audit_evidence_notes: str = ""


@dataclass
class Requirements:
    board: str = "ABPN"
    specialty: str = "Psychiatry"
    version: str = "unknown"
    sources: List[str] = field(default_factory=list)
    rules: RequirementRules = field(default_factory=RequirementRules)


SAFE_DEFAULT = Requirements()


def _safe_requirements() -> dict:
    return {
        "board": SAFE_DEFAULT.board,
        "specialty": SAFE_DEFAULT.specialty,
        "version": SAFE_DEFAULT.version,
        "sources": SAFE_DEFAULT.sources,
        "rules": {
            "cycle_years": SAFE_DEFAULT.rules.cycle_years,
            "total_credits": SAFE_DEFAULT.rules.total_credits,
            "annual_minimum": SAFE_DEFAULT.rules.annual_minimum,
            "safety_or_ethics_min": SAFE_DEFAULT.rules.safety_or_ethics_min,
            "audit_evidence_notes": SAFE_DEFAULT.rules.audit_evidence_notes,
        },
    }


def load_abpn_psychiatry_requirements() -> dict:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("requirements JSON must be a dict")
            # Basic validation
            data.setdefault("board", SAFE_DEFAULT.board)
            data.setdefault("specialty", SAFE_DEFAULT.specialty)
            data.setdefault("version", SAFE_DEFAULT.version)
            data.setdefault("sources", [])
            if not isinstance(data["sources"], list):
                data["sources"] = []
            data.setdefault("rules", {})
            if not isinstance(data["rules"], dict):
                data["rules"] = {}
            rules = data["rules"]
            rules.setdefault("cycle_years", 0)
            rules.setdefault("total_credits", 0)
            rules.setdefault("annual_minimum", 0)
            rules.setdefault("safety_or_ethics_min", 0)
            rules.setdefault("audit_evidence_notes", "")
            return data
    except FileNotFoundError:
        LOGGER.warning("Requirements config missing at %s", CONFIG_PATH)
    except json.JSONDecodeError:
        LOGGER.exception("Failed to parse requirements JSON")
    except Exception:
        LOGGER.exception("Unexpected error loading requirements")
    return _safe_requirements()


def requirements_version(req: dict) -> str:
    return str(req.get("version") or SAFE_DEFAULT.version)


def requirements_sources(req: dict) -> List[str]:
    sources = req.get("sources") or []
    if not isinstance(sources, list):
        return []
    return [str(s) for s in sources]


def requirements_rules(req: dict) -> dict:
    rules = req.get("rules") or {}
    if not isinstance(rules, dict):
        return {}
    return rules


def classify_topic(claim: Claim) -> Optional[str]:
    text = " ".join(
        filter(
            None,
            [
                claim.topic or "",
                claim.source_text or "",
            ],
        )
    ).lower()
    keywords = {
        "ethics": "ethic",
        "safety": "patient safety",
        "risk": "risk",
        "opioid": "opioid safety",
    }
    for label, needle in keywords.items():
        if needle in text:
            if label == "ethics":
                return "ethics"
            return "safety"
    if "ethic" in text:
        return "ethics"
    if "safety" in text:
        return "safety"
    return None


def _annual_credits(claims: List[Claim]) -> Dict[int, float]:
    buckets: Dict[int, float] = {}
    for claim in claims:
        year = claim.date.year if isinstance(claim.date, date) else None
        if not year:
            continue
        buckets[year] = buckets.get(year, 0.0) + float(claim.credits or 0.0)
    return buckets


def validate_against_requirements(user: User, claims: List[Claim], req: dict) -> dict:
    rules = requirements_rules(req)
    target_total = int(rules.get("total_credits") or 0)
    if getattr(user, "target_credits", 0) and user.target_credits > 0:
        target_total = int(user.target_credits)
    earned_total = sum(float(c.credits or 0.0) for c in claims)
    remaining_total = max(target_total - earned_total, 0.0)

    checks: List[dict] = []

    # Total credits check
    short_total = target_total - earned_total
    status_total = "ok"
    if short_total > 0:
        status_total = "warn" if short_total < 1.0 else "fail"
    checks.append(
        {
            "label": "Total credits",
            "status": status_total,
            "detail": f"{earned_total:.1f} of {target_total} credits earned",
        }
    )

    # Annual minimum (current year only for now)
    annual_min = float(rules.get("annual_minimum") or 0)
    if annual_min > 0:
        buckets = _annual_credits(claims)
        current_year = date.today().year
        current_total = buckets.get(current_year, 0.0)
        short_year = annual_min - current_total
        status_year = "ok"
        if short_year > 0:
            status_year = "warn" if short_year < 1.0 else "fail"
        detail = f"{current_total:.1f} of {annual_min} credits earned in {current_year}"
        checks.append(
            {
                "label": "Annual minimum",
                "status": status_year,
                "detail": detail,
            }
        )

    # Safety/Ethics minimum combined bucket
    safety_min = float(rules.get("safety_or_ethics_min") or 0)
    if safety_min > 0:
        bucket_total = 0.0
        for claim in claims:
            category = classify_topic(claim)
            if category in {"ethics", "safety"}:
                bucket_total += float(claim.credits or 0.0)
        short_bucket = safety_min - bucket_total
        status_bucket = "ok"
        if short_bucket > 0:
            status_bucket = "warn" if short_bucket < 1.0 else "fail"
        detail = f"{bucket_total:.1f} of {safety_min} credits from safety/ethics"
        checks.append(
            {
                "label": "Safety/Ethics minimum",
                "status": status_bucket,
                "detail": detail,
            }
        )

    return {
        "summary": {
            "target_total": target_total,
            "earned_total": earned_total,
            "remaining_total": remaining_total,
        },
        "checks": checks,
        "meta": {
            "version": requirements_version(req),
            "board": req.get("board", SAFE_DEFAULT.board),
            "specialty": req.get("specialty", SAFE_DEFAULT.specialty),
            "sources": requirements_sources(req),
        },
    }


def compute_content_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_requirements_snapshot(board: str, specialty: str, data: dict) -> None:
    payload = {
        "board": board,
        "specialty": specialty,
        "version": str(data.get("version") or "unknown"),
        "effective_date": None,
        "source_urls": data.get("sources") or [],
        "rules": data.get("rules") or {},
        "content_hash": compute_content_hash(json.dumps(data, sort_keys=True)),
    }
    with get_session() as session:
        snapshot = RequirementsSnapshot(**payload)
        session.add(snapshot)
        session.commit()


def latest_requirements_snapshot(board: str, specialty: str) -> Optional[dict]:
    with get_session() as session:
        stmt = (
            sa_select(RequirementsSnapshot)
            .where(
                RequirementsSnapshot.board == board,
                RequirementsSnapshot.specialty == specialty,
            )
            .order_by(RequirementsSnapshot.created_at.desc())
        )
        row = session.exec(stmt).first()
        if not row:
            return None
        return {
            "board": row.board,
            "specialty": row.specialty,
            "version": row.version,
            "source_urls": row.source_urls or [],
            "rules": row.rules or {},
            "content_hash": row.content_hash,
            "created_at": row.created_at,
        }
