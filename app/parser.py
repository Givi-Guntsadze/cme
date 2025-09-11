import os
import re
from datetime import date, timedelta
from typing import Optional, Tuple

# Patterns
_CRED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:credit(?:s)?|cr(?:edits?)?|cme)\b", re.I)
_TOPIC_RE = re.compile(r"ethic|safety|opioid|psychopharm", re.I)
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _regex_parse(text: str) -> Tuple[float, Optional[str], date]:
    m = _CRED_RE.search(text)
    credits = float(m.group(1)) if m else 0.0

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

    client = OpenAI()  # reads OPENAI_API_KEY from env
    prompt = (
        "Extract CME update as JSON with fields: credits (number), "
        "topic (string|null), date (YYYY-MM-DD; default today). "
        f"Text: {text!r}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    import json

    content = resp.choices[0].message.content
    data = json.loads(content)
    c = float(data.get("credits", 0) or 0)
    t = data.get("topic") or None
    d = date.fromisoformat(data.get("date")) if data.get("date") else date.today()
    return c, t, d


def parse_message(text: str) -> Tuple[float, Optional[str], date]:
    # Try regex first (fast & deterministic)
    c, t, d = _regex_parse(text)
    if c > 0:
        return c, t, d

    # If regex fails, try AI when key is present
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _ai_parse(text)
        except Exception:
            pass

    return 0.0, t, d
