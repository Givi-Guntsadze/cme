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
from .knowledge import get as get_knowledge_base

LOGGER = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).parent / "config" / "abpn_psychiatry_requirements.json"


SA_CME_KEYWORDS = [
    "sa-cme",
    "sa cme",
    "self assessment",
    "self-assessment",
    "self assessment cme",
    "self-assessment cme",
    "knowledge self-assessment",
    "moc self-assessment",
    "moc self assessment",
    "ksa",
    "self assessment module",
    "sam",
]

PIP_KEYWORDS = [
    "pip",
    "performance improvement",
    "practice improvement",
    "quality improvement",
    "chart review",
    "reassessment",
    "improvement plan",
]


PATIENT_SAFETY_KEYWORDS = [
    "patient safety",
    "risk management",
    "error prevention",
    "quality improvement",
    "safety culture",
    "safety training",
]

REQUIREMENT_LABELS = {
    "patient_safety": "Patient Safety Activity",
    "sa_cme": "Self-Assessment CME",
    "pip": "Performance Improvement (PIP)",
}


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


def _normalize_requirements_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Requirements payload must be a dict")

    data = dict(payload)
    data.setdefault("board", SAFE_DEFAULT.board)
    data.setdefault("specialty", SAFE_DEFAULT.specialty)
    data.setdefault("version", SAFE_DEFAULT.version)

    sources = data.get("sources") or []
    if not isinstance(sources, list):
        sources = [sources]
    data["sources"] = [str(src) for src in sources if src]

    rules_input = data.get("rules") or {}
    if not isinstance(rules_input, dict):
        rules_input = {}
    rules = dict(rules_input)

    def _int_or(default_key: str, candidate) -> int:
        try:
            return int(candidate)
        except (TypeError, ValueError):
            return int(rules.get(default_key) or 0)

    total = rules.get("total_credits")
    if total is None:
        total = _int_or("total_credits", rules.get("cme_total_per_cycle"))
    rules["total_credits"] = int(total or 0)

    try:
        rules["cycle_years"] = int(rules.get("cycle_years") or 0)
    except (TypeError, ValueError):
        rules["cycle_years"] = 0

    if "annual_minimum" not in rules:
        rules["annual_minimum"] = _int_or(
            "annual_minimum", rules.get("annual_minimum_credits")
        )
    try:
        rules["annual_minimum"] = int(rules.get("annual_minimum") or 0)
    except (TypeError, ValueError):
        rules["annual_minimum"] = 0

    safety_block = rules.get("patient_safety_activity")
    safety_min = rules.get("safety_or_ethics_min")
    if isinstance(safety_block, dict) and safety_block.get("required"):
        rules["safety_or_ethics_min"] = max(int(safety_min or 0), 1)
    else:
        try:
            rules["safety_or_ethics_min"] = int(safety_min or 0)
        except (TypeError, ValueError):
            rules["safety_or_ethics_min"] = 0

    try:
        rules["sa_cme_min_per_cycle"] = float(rules.get("sa_cme_min_per_cycle") or 0.0)
    except (TypeError, ValueError):
        rules["sa_cme_min_per_cycle"] = 0.0

    try:
        rules["pip_required_per_cycle"] = int(rules.get("pip_required_per_cycle") or 0)
    except (TypeError, ValueError):
        rules["pip_required_per_cycle"] = 0

    rules["audit_evidence_notes"] = str(rules.get("audit_evidence_notes") or "")

    data["rules"] = rules
    return data


def _load_requirements_from_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("requirements JSON must be a dict")
    return data


def load_requirements_for(board: str, specialty: str) -> dict:
    kb = get_knowledge_base(board, specialty)
    if kb:
        try:
            data = kb.load_requirements()
            return _normalize_requirements_payload(data)
        except Exception:
            LOGGER.exception("knowledge base load failed for %s %s", board, specialty)
    if board.lower() == "abpn" and specialty.lower() == "psychiatry":
        try:
            data = _load_requirements_from_file(CONFIG_PATH)
            return _normalize_requirements_payload(data)
        except Exception:
            LOGGER.exception("Failed to read ABPN psychiatry requirements file")
    return _safe_requirements()


def load_abpn_psychiatry_requirements() -> dict:
    return load_requirements_for("ABPN", "Psychiatry")


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


def _claim_text(claim: Claim) -> str:
    parts = [
        claim.topic or "",
        claim.source_text or "",
    ]
    return " ".join(filter(None, parts)).lower()


def classify_topic(claim: Claim) -> Optional[str]:
    text = _claim_text(claim)
    if not text:
        return None
    if "ethic" in text:
        return "ethics"
    safety_needles = list(PATIENT_SAFETY_KEYWORDS) + ["opioid safety", "risk", "safety"]
    for needle in safety_needles:
        if needle in text:
            return "safety"
    return None


def is_sa_cme_claim(claim: Claim) -> bool:
    text = _claim_text(claim)
    if not text:
        return False
    return any(keyword in text for keyword in SA_CME_KEYWORDS)


def is_pip_claim(claim: Claim) -> bool:
    text = _claim_text(claim)
    if not text:
        return False
    return any(keyword in text for keyword in PIP_KEYWORDS)


def infer_requirement_tags_from_text(*parts: object) -> set[str]:
    text = " ".join(str(p).lower() for p in parts if p).strip()
    tags: set[str] = set()
    if not text:
        return tags
    if any(keyword in text for keyword in PATIENT_SAFETY_KEYWORDS):
        tags.add("patient_safety")
    if any(keyword in text for keyword in SA_CME_KEYWORDS):
        tags.add("sa_cme")
    if any(keyword in text for keyword in PIP_KEYWORDS):
        tags.add("pip")
    return tags


def sa_cme_credit_sum(claims: List[Claim]) -> float:
    total = 0.0
    for claim in claims:
        if is_sa_cme_claim(claim):
            total += float(claim.credits or 0.0)
    return total


def pip_activity_count(claims: List[Claim]) -> int:
    count = 0
    for claim in claims:
        if is_pip_claim(claim):
            count += 1
    return count


def _status_for_gap(shortfall: float, warn_threshold: float = 1.0) -> str:
    if shortfall <= 0:
        return "ok"
    if shortfall < warn_threshold:
        return "warn"
    return "fail"


def _clinically_inactive(user: User) -> bool:
    stage = str(getattr(user, "professional_stage", "") or "").lower()
    return stage in {"non_clinical", "inactive", "retired", "non-clinical"}


def _pip_complete_from_claims(claims: List[Claim]) -> tuple[bool, str | None]:
    topics = {str(claim.topic or "").lower() for claim in claims}
    if "pip_complete" in topics or "pip completion" in topics:
        return True, "Documented pip_complete claim"
    if {"pip_step_a", "pip_step_c"}.issubset(topics):
        return True, "PIP steps A and C logged"
    if pip_activity_count(claims) > 0:
        return True, "PIP activity logged"
    return False, None


def validate_full_cc(
    user: User,
    claims: List[Claim],
    req: dict,
    pip_status: Optional[dict] = None,
) -> dict:
    rules = requirements_rules(req)
    target_total = int(
        rules.get("total_credits") or rules.get("cme_total_per_cycle") or 0
    )
    if getattr(user, "target_credits", 0) and user.target_credits > 0:
        target_total = int(user.target_credits)

    earned_total = sum(float(c.credits or 0.0) for c in claims)
    short_total = max(target_total - earned_total, 0.0)
    total_status = _status_for_gap(target_total - earned_total)
    pillars = {
        "cme": {
            "label": "CME Total",
            "earned": earned_total,
            "target": target_total,
            "status": total_status,
            "detail": (
                f"{earned_total:.1f} of {target_total} CME credits earned"
                if target_total > 0
                else f"{earned_total:.1f} CME credits logged"
            ),
        }
    }

    gaps: List[str] = []
    if total_status != "ok" and target_total > 0:
        gaps.append(f"CME short by {short_total:.1f}")

    sa_min = float(rules.get("sa_cme_min_per_cycle") or 0.0)
    sa_total = sa_cme_credit_sum(claims)
    short_sa = max(sa_min - sa_total, 0.0)
    sa_status = "ok" if sa_min <= 0 else _status_for_gap(sa_min - sa_total)
    pillars["sa_cme"] = {
        "label": "Self-Assessment CME",
        "earned": sa_total,
        "target": sa_min,
        "status": sa_status,
        "detail": (
            f"{sa_total:.1f} of {sa_min:.1f} SA-CME credits documented"
            if sa_min > 0
            else f"{sa_total:.1f} SA-CME credits logged"
        ),
    }
    if sa_status != "ok" and sa_min > 0:
        gaps.append(f"SA-CME short by {short_sa:.1f}")

    pip_required = int(rules.get("pip_required_per_cycle") or 0)
    pip_complete = bool(pip_status.get("complete")) if pip_status else False
    pip_reason = None
    if not pip_complete:
        pip_complete, pip_reason = _pip_complete_from_claims(claims)
    if not pip_complete and _clinically_inactive(user):
        pip_complete = True
        pip_reason = "Clinically inactive"

    if pip_status and pip_status.get("reason") and pip_complete:
        pip_reason = pip_status.get("reason")

    if pip_required <= 0:
        pip_status_label = "ok"
        pip_detail = "PIP not required for this cycle"
    else:
        pip_status_label = "ok" if pip_complete else "fail"
        if pip_complete:
            pip_detail = pip_reason or "Performance Improvement documented"
        else:
            pip_detail = f"Need {pip_required} completed PIP activity"

    pillars["pip"] = {
        "label": "Performance Improvement (PIP)",
        "earned": 1 if pip_complete else 0,
        "target": pip_required,
        "status": pip_status_label,
        "detail": pip_detail,
    }
    if pip_status_label != "ok" and pip_required > 0:
        gaps.append("PIP missing")

    safety_block = rules.get("patient_safety_activity")
    safety_required = bool(
        isinstance(safety_block, dict) and safety_block.get("required")
    )
    safety_completed = False
    if safety_required:
        for claim in claims:
            if classify_topic(claim) == "safety":
                safety_completed = True
                break

    if not safety_required:
        safety_status = "ok"
        safety_detail = "Patient safety activity not required"
    else:
        safety_status = "ok" if safety_completed else "fail"
        safety_detail = (
            "Patient safety activity logged"
            if safety_completed
            else "No patient safety activity logged yet"
        )

    pillars["patient_safety"] = {
        "label": "Patient Safety Activity",
        "earned": 1 if safety_completed else 0,
        "target": 1 if safety_required else 0,
        "status": safety_status,
        "detail": safety_detail,
    }
    if safety_status != "ok" and safety_required:
        gaps.append("Patient safety activity pending")

    return {
        "summary": {
            "target_total": target_total,
            "earned_total": earned_total,
            "remaining_total": max(target_total - earned_total, 0.0),
        },
        "pillars": pillars,
        "gaps": gaps,
        "meta": {
            "version": requirements_version(req),
            "board": req.get("board", SAFE_DEFAULT.board),
            "specialty": req.get("specialty", SAFE_DEFAULT.specialty),
            "sources": requirements_sources(req),
        },
    }


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
    target_total = int(
        rules.get("total_credits") or rules.get("cme_total_per_cycle") or 0
    )
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

    # Self-Assessment CME minimum
    sa_min = float(rules.get("sa_cme_min_per_cycle") or 0)
    if sa_min > 0:
        sa_total = sa_cme_credit_sum(claims)
        short_sa = sa_min - sa_total
        status_sa = "ok"
        if short_sa > 0:
            status_sa = "warn" if short_sa < 1.0 else "fail"
        detail = f"{sa_total:.1f} of {sa_min} SA-CME credits documented"
        checks.append(
            {
                "label": "Self-Assessment CME",
                "status": status_sa,
                "detail": detail,
            }
        )

    # Performance Improvement (PIP) requirement
    pip_required = int(rules.get("pip_required_per_cycle") or 0)
    if pip_required > 0:
        pip_completed = pip_activity_count(claims)
        short_pip = pip_required - pip_completed
        status_pip = "ok" if short_pip <= 0 else "fail"
        detail = f"{pip_completed} of {pip_required} PIP activities documented"
        checks.append(
            {
                "label": "Performance Improvement (PIP)",
                "status": status_pip,
                "detail": detail,
            }
        )

    # Patient safety activity requirement (one-time per cycle)
    safety_block = rules.get("patient_safety_activity")
    if isinstance(safety_block, dict) and safety_block.get("required"):
        safety_completed = False
        for claim in claims:
            if classify_topic(claim) == "safety":
                safety_completed = True
                break
        detail = (
            "At least one patient safety activity completed"
            if safety_completed
            else "No patient safety activity logged yet"
        )
        status = "ok" if safety_completed else "warn"
        if safety_block.get("portal_determined") and not safety_completed:
            detail += "; verify status in ABPN Physician Portal"
        checks.append(
            {
                "label": "Patient safety activity",
                "status": status,
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
