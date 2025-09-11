# app/ingest.py
import os
import re
import json
import logging
from datetime import date
from typing import List, Optional, Tuple

import httpx
from sqlmodel import select

from .db import get_session
from .models import Activity, User

logger = logging.getLogger(__name__)


def _redact_key(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"(key=)[^&\s]+", r"\1REDACTED", text, flags=re.I)
    text = re.sub(r"(GOOGLE_API_KEY:\s*)\S+", r"\1REDACTED", text)
    return text


# --- Google Programmable Search (Primary) ---


def fetch_google_cse(query: str, num: int = 10, start: Optional[int] = None):
    key = os.getenv("GOOGLE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    if not (key and cx):
        raise RuntimeError("Google CSE not configured")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": key, "cx": cx, "num": max(1, min(int(num or 10), 10))}
    if start is not None:
        params["start"] = max(1, int(start))
    try:
        resp = httpx.get(url, params=params, timeout=20)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        safe_url = _redact_key(str(e.request.url))
        logger.error("Google CSE failed %s (%s)", safe_url, e.response.status_code)
        raise RuntimeError("Google CSE request failed") from None
    except Exception as e:
        logger.exception("CSE error: %s", _redact_key(str(e)))
        raise RuntimeError("Google CSE request failed") from None
    data = resp.json()
    items = data.get("items", [])
    return [
        {
            "title": it.get("title", ""),
            "link": it.get("link", ""),
            "snippet": it.get("snippet", ""),
        }
        for it in items
    ]


def fetch_google_cse_multi(query: str, total: int = 20) -> List[dict]:
    results: List[dict] = []
    start = 1
    while len(results) < total:
        batch = fetch_google_cse(query, num=10, start=start)
        if not batch:
            break
        # Dedup by link
        seen = {r.get("link") for r in results}
        for it in batch:
            if it.get("link") not in seen:
                results.append(it)
        start += 10
        if start > 91:  # API max
            break
    return results[:total]


# --- Heuristics (used only for minimal fallback parsing) ---
CRED_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:AMA PRA)?\s*Category\s*1(?:\s*Credit)?(?:\(s\))?", re.I
)
ALT_CRED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*credit", re.I)
COST_RE = re.compile(r"\$(\d{1,4}(?:\.\d{2})?)")
MODALITY_RE = re.compile(r"\bonline|virtual|on-demand|webinar|live\b", re.I)


def _openai_client():
    from openai import OpenAI

    return OpenAI()


# Robust JSON parsing helper
# Returns a dict on success or None on failure


def safe_json_loads(text: str) -> Optional[dict]:
    try:
        t = (text or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
            t = re.sub(r"\s*```$", "", t)
        try:
            return json.loads(t)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", t)
            return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def _build_query_for_user(user: Optional[User]) -> str:
    base = "psychiatry CME AMA PRA Category 1 credit"
    if not user:
        return base + " online OR webinar OR on-demand"
    if user.allow_live:
        live_part = 'conference OR symposium OR in-person OR "live CME"'
        if user.city:
            live_part += f" {user.city}"
        return f"{base} ({live_part} OR online OR webinar OR on-demand)"
    else:
        return base + " online OR webinar OR on-demand OR virtual"


def _build_queries_for_user(user: Optional[User]) -> List[str]:
    base = "psychiatry CME AMA PRA Category 1 credit"
    city = user.city if user and user.city else None
    queries: List[str] = []
    # Online-focused
    queries.append(base + " online OR webinar OR on-demand")
    # Provider-oriented
    queries.append(base + " site:ama-assn.org OR site:accme.org OR site:medscape.org")
    # Live-focused when allowed
    if user and user.allow_live:
        live_terms = 'conference OR symposium OR in-person OR "live CME"'
        q = f"{base} {live_terms}"
        queries.append(q + (f" {city}" if city else ""))
    # City-focused general
    if city:
        queries.append(f"{base} CME {city}")
    # Specialty hint
    if user and user.specialty:
        queries.append(f"{user.specialty} {base}")
    # Remove dupes while preserving order
    seen = set()
    uniq: List[str] = []
    for q in queries:
        if q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq


def _insert_items(items: List[dict]) -> int:
    """Insert parsed items into DB with dedupe by URL, else (title, provider)."""
    added = 0
    with get_session() as s:
        for it in items:
            try:
                title = (it.get("title") or "").strip()
                provider = (it.get("provider") or "").strip()
                credits = float(it.get("credits") or 0)
                cost = float(it.get("cost_usd") or 0)
                modality = (it.get("modality") or "online").lower()
                city = it.get("city") or None
                days_required = int(
                    it.get("days_required") or (0 if modality == "online" else 1)
                )
                start_date_s = it.get("start_date") or None
                end_date_s = it.get("end_date") or None
                start_date = date.fromisoformat(start_date_s) if start_date_s else None
                end_date = date.fromisoformat(end_date_s) if end_date_s else None
                url = (it.get("url") or "").strip() or None
                summary = it.get("summary") or None
                source = (it.get("source") or None) or "web"

                if not title or credits <= 0:
                    continue

                # Prefer URL-based dedupe
                exists = None
                if url:
                    exists = s.exec(select(Activity).where(Activity.url == url)).first()
                if not exists:
                    exists = s.exec(
                        select(Activity).where(
                            Activity.title == title, Activity.provider == provider
                        )
                    ).first()
                if exists:
                    continue

                a = Activity(
                    title=title[:200],
                    provider=provider[:120] or (url or "")[:120],
                    credits=credits,
                    cost_usd=cost,
                    modality=(
                        "online"
                        if modality.startswith("on")
                        or modality.startswith("web")
                        or modality == "virtual"
                        else ("live" if modality == "live" else "online")
                    ),
                    city=city,
                    days_required=days_required,
                    start_date=start_date,
                    end_date=end_date,
                    url=url,
                    summary=summary,
                    source=source,
                )
                s.add(a)
                added += 1
            except Exception as e:
                logger.debug("insert skip: %s", e)
                continue
        s.commit()
    return added


def _extract_with_openai_from_candidates(
    candidates: List[dict], count: int
) -> List[dict]:
    """Use OpenAI to turn Google CSE candidates into strict JSON CME items."""
    if not candidates:
        return []

    client = _openai_client()
    schema_note = (
        "Return STRICT JSON ONLY (no markdown, no commentary) with this schema: "
        '{"items": [{"title": "string", "provider": "string", "credits": 0, '
        '"cost_usd": 0, "modality": "online|live", "city": null, '
        '"start_date": "YYYY-MM-DD"|null, "end_date": "YYYY-MM-DD"|null, '
        '"days_required": 0|1|2|null, "url": "string", "summary": "string|null", "source": "web"}]}'
    )
    guidance = (
        "Only include items awarding AMA PRA Category 1 Credits. "
        "Set url to the candidate's link. If a field is unknown, use null (not an empty string). "
        "Provide a single-sentence summary of the activity (<=160 chars)."
    )

    content = (
        f"You are extracting structured CME activities from candidate web results. {guidance}\n"
        f"Limit to about {count} best items.\n\n"
        f"Candidates (JSON with title, link, snippet):\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"{schema_note}"
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=content,
        temperature=0,
    )

    # Parse JSON defensively
    text = getattr(resp, "output_text", None)
    if not text:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
        text = "\n".join(parts)

    data = safe_json_loads(text or "")
    if data is None:
        print("ingest: JSON load failed from candidates")
        return []

    items = data.get("items") or []
    if not isinstance(items, list):
        return []

    # Ensure URL present by mapping title->link if missing; filter credits>0
    link_map = {(it.get("title") or "").strip(): it.get("link") for it in candidates}
    cleaned: List[dict] = []
    for it in items:
        try:
            it = dict(it or {})
            it.setdefault("url", link_map.get((it.get("title") or "").strip(), ""))
            it.setdefault("source", "web")
            if not it.get("summary"):
                # use snippet as fallback summary
                snip = next(
                    (
                        c.get("snippet")
                        for c in candidates
                        if (c.get("title") or "").strip()
                        == (it.get("title") or "").strip()
                    ),
                    "",
                )
                it["summary"] = snip or None
            # credits must be > 0
            cval = it.get("credits")
            credits = float(cval) if cval is not None else 0.0
            if credits <= 0:
                continue
            it["credits"] = credits
            cleaned.append(it)
        except Exception as e:
            print(f"ingest: dropping item due to parse error: {e}")
            continue

    return cleaned


def _fallback_with_openai_web_search(count: int) -> List[dict]:
    """Use OpenAI web_search tool to find additional CME items."""
    client = _openai_client()
    prompt = (
        "Find up-to-date CME opportunities in psychiatry that award AMA PRA Category 1 Credits. "
        "Return STRICT JSON with top-level 'items' only, with fields: title, provider, credits (number), cost_usd (number|null), "
        "modality ('online'|'live'), city (string|null), start_date (YYYY-MM-DD|null), end_date (YYYY-MM-DD|null), "
        f"days_required (0|1|2|null), url, summary (string|null), source ('ai'). Limit to about {count} high-quality results."
    )
    resp = client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search"}],
        input=prompt,
        temperature=0,
    )

    text = getattr(resp, "output_text", None)
    if not text:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
        text = "\n".join(parts)

    data = safe_json_loads(text or "")
    items = (data or {}).get("items") or []
    return items if isinstance(items, list) else []


def ingest_psychiatry_online_ai(
    count: int = 10,
) -> Tuple[int, bool, int, int, int, int, bool]:
    """
    Primary: Use preference-aware multi-query Google CSE to gather candidates, then OpenAI to extract structured CME items.
    Fallback: If fewer than `count` valid items inserted, use OpenAI web_search tool to supplement.

    Returns (total_inserted, used_fallback, google_inserted, ai_inserted, candidates_count, extracted_count, google_ok).
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY")

    total_added = 0
    used_fallback = False
    google_added = 0
    ai_added = 0
    candidates_count = 0
    extracted_count = 0
    google_ok = False

    # Load user for query shaping
    with get_session() as s:
        user = s.exec(select(User)).first()

    # Phase 1: Google CSE candidates -> OpenAI extraction
    candidates: List[dict] = []
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        queries = _build_queries_for_user(user)
        try:
            for q in queries:
                batch = fetch_google_cse_multi(q, total=20)
                if batch:
                    google_ok = True
                    # Dedup by link across queries
                    seen_links = {c.get("link") for c in candidates}
                    for it in batch:
                        if it.get("link") not in seen_links:
                            candidates.append(it)
            candidates_count = len(candidates)
        except Exception as e:
            logger.error("google ingestion failed: %s", _redact_key(str(e)))
            candidates = []
            candidates_count = 0

    if candidates:
        items = _extract_with_openai_from_candidates(candidates, count)
        extracted_count = len(items)
        added = _insert_items(items)
        google_added += added
        total_added += added

    # Phase 2: Fallback to OpenAI web_search if not enough
    if total_added < count:
        used_fallback = True
        more = _fallback_with_openai_web_search(count - total_added)
        ai_added2 = _insert_items(more)
        ai_added += ai_added2
        total_added += ai_added2

    return (
        total_added,
        used_fallback,
        google_added,
        ai_added,
        candidates_count,
        extracted_count,
        google_ok,
    )


# NOTE: Bing Web Search API was retired for new keys (Aug 2025). Legacy code removed.
