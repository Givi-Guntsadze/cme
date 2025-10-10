import re
from datetime import date, timedelta
from typing import Optional, Tuple

from .openai_helpers import call_responses, response_text
from .env import get_secret

# Patterns
_CREDIT_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:credit(?:s)?|cr(?:eds?)?|hour(?:s)?)\b", re.I),
    re.compile(
        r"(?:earn(?:ed)?|log(?:ged)?|claim(?:ed)?|finish(?:ed)?|complete(?:d)?)\s*(\d+(?:\.\d+)?)\s*cme\b",
        re.I,
    ),
    re.compile(r"cme\s*(?:credit(?:s)?|hour(?:s)?)\s*(\d+(?:\.\d+)?)", re.I),
]
_TOPIC_RE = re.compile(r"ethic|safety|opioid|psychopharm", re.I)
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _looks_like_year(raw: str) -> bool:
    if not raw.isdigit():
        return False
    value = int(raw)
    return 1800 <= value <= 2200


FUTURE_INTENT_HINTS = {
    "want",
    "rather",
    "prefer",
    "looking",
    "searching",
    "need",
    "should",
    "must",
    "hoping",
    "plan to",
    "planning",
    "would like",
    "going to",
}

COMPLETION_HINTS = {
    "earn",
    "earned",
    "log",
    "logged",
    "claim",
    "claimed",
    "finish",
    "finished",
    "complete",
    "completed",
    "attended",
    "took",
    "received",
    "got",
}


CORRECTION_KEYWORDS = {
    "delete",
    "remove",
    "undo",
    "erase",
    "clear",
    "rollback",
    "roll back",
    "fix",
    "correct",
    "clearing",
    "reset",
    "strip",
    "drop",
    "mistake",
    "mistaken",
    "incorrect",
    "wrong",
    "error",
    "accident",
    "accidental",
    "shouldn't be",
    "should not be",
    "shouldn't have",
    "should not have",
    "false",
    "bad data",
}

NEGATED_ACTION_PATTERN = re.compile(
    r"\b(?:didn['’]?t|did not|haven['’]?t|have not|never|wasn['’]?t|was not)\b"
    r".{0,60}?\b(?:earn|log|claim|complete|finish|receive)\b",
    re.I | re.S,
)
DO_NOT_ACTION_PATTERN = re.compile(
    r"\b(?:do(?:n't| not)|please\s+don't|should(?:n't| not))\s+"
    r"(?:log|count|record|add)\b",
    re.I,
)
CORRECTION_WITH_CREDITS_PATTERN = re.compile(
    r"\b(?:remove|delete|clear|erase|undo|fix|correct|adjust|drop)\b"
    r".{0,40}\b(?:credit|log|entry)\b",
    re.I | re.S,
)


def _has_completion_language(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in COMPLETION_HINTS)


def _has_future_language(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in FUTURE_INTENT_HINTS)


def _is_correction_or_negation(text: str) -> bool:
    lowered = text.lower()
    if (
        "credit" in lowered
        or "credits" in lowered
        or "log" in lowered
        or "entry" in lowered
    ) and any(keyword in lowered for keyword in CORRECTION_KEYWORDS):
        return True
    if NEGATED_ACTION_PATTERN.search(text):
        return True
    if DO_NOT_ACTION_PATTERN.search(text):
        return True
    if CORRECTION_WITH_CREDITS_PATTERN.search(text):
        return True
    return False


def _extract_credits(text: str) -> float:
    lowered = text.lower()
    for pattern in _CREDIT_PATTERNS:
        match = pattern.search(lowered)
        if not match:
            continue
        raw_value = match.group(1)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        # Guard against capturing cycle years like "2023 CME"
        if value.is_integer() and _looks_like_year(raw_value):
            continue
        # Very large values are almost always mistakes
        if value > 200:
            continue
        return value
    return 0.0


def _regex_parse(text: str) -> Tuple[float, Optional[str], date]:
    credits = _extract_credits(text)

    tm = _TOPIC_RE.search(text)
    topic = tm.group(0).lower() if tm else None

    dm = _DATE_RE.search(text)
    if dm:
        d = date.fromisoformat(dm.group(1))
    elif "yesterday" in text.lower():
        d = date.today() - timedelta(days=1)
    else:
        d = date.today()

    return credits, topic, d


def _ai_parse(text: str) -> Tuple[float, Optional[str], date]:
    # Optional: uses OpenAI if key is present
    from openai import OpenAI

    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    prompt = (
        "Extract CME update as JSON with fields: credits (number), "
        "topic (string|null), date (YYYY-MM-DD; default today). "
        f"Text: {text!r}"
    )
    resp = call_responses(
        client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_output_tokens=150,
    )
    import json

    content = response_text(resp)
    data = json.loads(content)
    c = float(data.get("credits", 0) or 0)
    t = data.get("topic") or None
    d = date.fromisoformat(data.get("date")) if data.get("date") else date.today()
    return c, t, d


def parse_message(text: str) -> Tuple[float, Optional[str], date]:
    if not text:
        return 0.0, None, date.today()

    if _is_correction_or_negation(text):
        _, topic, parsed_date = _regex_parse(text)
        return 0.0, topic, parsed_date

    # Try regex first (fast & deterministic)
    c, t, d = _regex_parse(text)
    if c > 0:
        if _has_future_language(text) and not _has_completion_language(text):
            return 0.0, t, d
        return c, t, d

    # If regex fails, try AI when key is present
    if get_secret("OPENAI_API_KEY"):
        try:
            return _ai_parse(text)
        except Exception:
            pass

    return 0.0, t, d
