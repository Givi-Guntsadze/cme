import re
from datetime import date, timedelta
from typing import Optional, Tuple

_CRED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:credit|cr|cme)\b", re.I)
_TOPIC_RE = re.compile(r"ethic|safety|opioid|psychopharm", re.I)
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_message(text: str) -> Tuple[float, Optional[str], date]:
    # credits
    m = _CRED_RE.search(text)
    credits = float(m.group(1)) if m else 0.0

    # topic
    tm = _TOPIC_RE.search(text)
    topic = tm.group(0).lower() if tm else None

    # date
    dm = _DATE_RE.search(text)
    if dm:
        d = date.fromisoformat(dm.group(1))
    elif "yesterday" in text.lower():
        d = date.today() - timedelta(days=1)
    else:
        d = date.today()

    return credits, topic, d
