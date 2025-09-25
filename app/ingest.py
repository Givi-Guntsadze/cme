# app/ingest.py
import os
import re
import json
import logging
from datetime import date
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from sqlmodel import select

from .db import get_session
from .models import Activity, User

logger = logging.getLogger(__name__)

# In-memory page cache for a single ingest run
_page_cache: Dict[str, str] = {}


def _redact_key(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"(key=)[^&\s]+", r"\1REDACTED", text, flags=re.I)
    text = re.sub(r"(GOOGLE_API_KEY:\s*)\S+", r"\1REDACTED", text)
    return text


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


def fetch_page(url: str, timeout: int = 20) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CMEBot/1.0)"}
        if url in _page_cache:
            return _page_cache[url]
        r = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        if r.status_code == 200:
            _page_cache[url] = r.text
            return r.text
        logger.info("page fetch non-200 for %s: %s", _redact_key(url), r.status_code)
        return None
    except Exception as e:
        logger.exception("page fetch failed: %s", _redact_key(f"{url} {e}"))
        return None


def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:15000]
    except Exception:
        logger.exception("html_to_text failed")
        return ""


async def ai_extract_from_text(text: str, url: str) -> Optional[dict]:
    try:
        from openai import OpenAI

        client = OpenAI()
        schema = (
            "Return STRICT JSON only with keys: title, provider, credits (number), cost_usd (number|null), "
            "modality ('online'|'live'), city (string|null), start_date (YYYY-MM-DD|null), end_date (YYYY-MM-DD|null), "
            "days_required (integer|null), url (string), hybrid_available (boolean|null), "
            "pricing_options (array of objects with label, cost_usd, deadline (YYYY-MM-DD|null), "
            "conditions {membership, stage, max_years_post_residency}), "
            "eligible_institutions (array of strings|null), eligible_groups (array of strings|null), "
            "membership_required (string|null), open_to_public (boolean|null)."
        )
        prompt = (
            f"Extract CME info from the following page text. {schema} URL: {url}. "
            "If provider isn't explicit, use the page domain as provider."
        )
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        data = safe_json_loads(getattr(resp, "output_text", "") or "")
        return data or None
    except Exception:
        logger.exception("ai_extract_from_text failed for %s", _redact_key(url))
        return None


def is_valid_record(r: Dict[str, Any]) -> bool:
    try:
        title = (r.get("title") or "").strip()
        provider = (r.get("provider") or "").strip()
        url = (r.get("url") or "").strip()
        if not provider and url:
            try:
                provider = urlparse(url).netloc
                if provider:
                    r["provider"] = provider
            except Exception:
                provider = ""
        credits = float(r.get("credits") or 0)
        if not title or not provider or credits <= 0:
            return False
        cost_raw = r.get("cost_usd")
        cost_known = False
        if isinstance(cost_raw, str):
            text = cost_raw.strip()
            if not text:
                cost_raw = None
            elif text.lower() in {"free", "complimentary", "no cost", "included"}:
                cost_raw = 0.0
        if cost_raw is not None and cost_raw != "":
            try:
                cost_val = float(cost_raw)
                if cost_val < 0:
                    return False
                r["cost_usd"] = cost_val
                cost_known = True
            except (TypeError, ValueError):
                return False
        if not cost_known:
            return False
        if (r.get("modality") or "").lower() == "online" and r.get("days_required") in (
            None,
            "",
        ):
            r["days_required"] = 0
        return True
    except Exception:
        return False


# OpenAI client factory
def _openai_client():
    from openai import OpenAI

    return OpenAI()


# --- Google Programmable Search (Primary) ---


def fetch_google_cse(
    query: str, num: int = 10, start: Optional[int] = None
) -> List[dict]:
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
        data = resp.json()
        items = data.get("items", [])
        return [
            {
                "title": it.get("title", ""),
                "link": it.get("link", ""),
                "displayLink": it.get("displayLink", ""),
                "snippet": it.get("snippet", ""),
            }
            for it in items
        ]
    except httpx.HTTPStatusError as e:
        # Full traceback + redacted URL
        safe_url = _redact_key(str(e.request.url) if e.request else url)
        logger.exception(
            "Google CSE failed %s (%s)",
            safe_url,
            getattr(e.response, "status_code", "?"),
        )
        raise RuntimeError("Google CSE request failed") from None
    except Exception as e:
        logger.exception("CSE error: %s", _redact_key(str(e)))
        raise RuntimeError("Google CSE request failed") from None


def fetch_google_cse_multi(query: str, total: int = 20) -> List[dict]:
    results: List[dict] = []
    start = 1
    # cap to ~30 items to reduce quota
    target = max(1, min(total, 30))
    while len(results) < target:
        batch = fetch_google_cse(query, num=10, start=start)
        if not batch:
            break
        # Dedup by link within candidate pool
        seen = {r.get("link") for r in results}
        for it in batch:
            if it.get("link") not in seen:
                results.append(it)
        start += 10
        if start > 91:
            break
    return results[:target]


# --- Heuristics ---
CRED_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:AMA PRA)?\s*Category\s*1(?:\s*Credit)?(?:\(s\))?", re.I
)
ALT_CRED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*credit", re.I)
COST_RE = re.compile(r"\$(\d{1,4}(?:\.\d{2})?)")
MODALITY_RE = re.compile(r"\bonline|virtual|on-demand|webinar|live\b", re.I)


def _split_list(text: str) -> List[str]:
    parts = []
    working = text.replace(" and ", ",")
    for chunk in working.split(","):
        item = chunk.strip(" .")
        if item:
            parts.append(item)
    return parts


def _parse_eligibility(snippet: Optional[str]) -> Dict[str, Any]:
    base_text = (snippet or "").strip()
    result: Dict[str, Any] = {
        "eligibility_text": base_text or None,
        "eligible_institutions": [],
        "eligible_groups": [],
        "membership_required": None,
        "open_to_public": True,
    }
    if not base_text:
        return result

    lowered = base_text.lower()

    m = re.search(r"open to ([^.;]+?) at ([^.;]+)", base_text, flags=re.I)
    if m:
        groups_raw = m.group(1)
        inst_raw = m.group(2)
        groups = _split_list(groups_raw)
        institutions = _split_list(inst_raw)
        if groups:
            result["eligible_groups"] = groups
        if institutions:
            result["eligible_institutions"] = institutions

    membership_match = re.search(
        r"([A-Za-z0-9&\- ]+?)\s+members? only", base_text, flags=re.I
    )
    if membership_match:
        membership = membership_match.group(1).strip(" .")
        if membership:
            result["membership_required"] = membership

    if any(
        phrase in lowered
        for phrase in (
            "open registration",
            "public welcome",
            "open to the public",
            "general public",
        )
    ):
        result["open_to_public"] = True

    if (
        result["eligible_institutions"]
        or result["eligible_groups"]
        or result["membership_required"]
    ):
        result["open_to_public"] = False

    return result


def _normalize_pricing_options(raw: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not raw:
        return normalized
    options = raw if isinstance(raw, list) else []
    if not isinstance(options, list):
        return normalized

    for opt in options:
        if not isinstance(opt, dict):
            continue
        label = str(opt.get("label") or "").strip() or "Alternate pricing"
        try:
            cost_val = float(opt.get("cost_usd"))
        except (TypeError, ValueError):
            continue
        if cost_val < 0:
            continue
        deadline_raw = opt.get("deadline")
        deadline_iso = None
        if deadline_raw:
            try:
                deadline_iso = date.fromisoformat(str(deadline_raw)).isoformat()
            except Exception:
                deadline_iso = None

        conditions = (
            opt.get("conditions") if isinstance(opt.get("conditions"), dict) else {}
        )
        membership = conditions.get("membership") or conditions.get("memberships")
        if membership and not isinstance(membership, list):
            membership = [membership]
        membership_list = []
        for item in membership or []:
            text = str(item).strip()
            if text:
                membership_list.append(text)

        stage = conditions.get("stage") or conditions.get("stages")
        if stage and not isinstance(stage, list):
            stage = [stage]
        stage_list = []
        for item in stage or []:
            text = str(item).strip()
            if text:
                stage_list.append(text)

        max_years = conditions.get("max_years_post_residency")
        try:
            max_years_val = float(max_years) if max_years is not None else None
        except (TypeError, ValueError):
            max_years_val = None

        normalized.append(
            {
                "label": label,
                "cost_usd": cost_val,
                "deadline": deadline_iso,
                "conditions": {
                    k: v
                    for k, v in {
                        "membership": membership_list,
                        "stage": stage_list,
                        "max_years_post_residency": max_years_val,
                    }.items()
                    if v
                },
            }
        )

    return normalized


def _build_queries_for_user(user: Optional[User]) -> List[str]:
    base = "psychiatry CME AMA PRA Category 1 credit"
    city = user.city if user and user.city else None
    queries: List[str] = []
    queries.append(base + " online OR webinar OR on-demand")
    queries.append(base + " site:ama-assn.org OR site:accme.org OR site:medscape.org")
    if user and user.allow_live:
        live_terms = 'conference OR symposium OR in-person OR "live CME"'
        queries.append(f"{base} {live_terms}" + (f" {city}" if city else ""))
    if city:
        queries.append(f"{base} CME {city}")
    if user and user.specialty:
        queries.append(f"{user.specialty} {base}")
    # unique preserve order
    seen = set()
    uniq: List[str] = []
    for q in queries:
        if q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq


def _insert_items(items: List[dict]) -> tuple[int, Dict[str, int]]:
    """Insert parsed items into DB with dedupe by (title, provider) only."""

    added = 0
    by_source: Dict[str, int] = {}

    with get_session() as s:
        for it in items:
            try:
                title = (it.get("title") or "").strip()
                provider_in = (it.get("provider") or "").strip()
                url = (it.get("url") or "").strip() or None
                # Fallback provider to domain if missing
                provider = provider_in or (urlparse(url).netloc if url else "")
                credits = float(it.get("credits") or 0)
                modality = (it.get("modality") or "online").lower()
                try:
                    cost = (
                        float(it.get("cost_usd"))
                        if it.get("cost_usd") is not None
                        else 0.0
                    )
                except (TypeError, ValueError):
                    cost = 0.0
                city = it.get("city") or None
                start_date_s = it.get("start_date") or None
                end_date_s = it.get("end_date") or None
                start_date = date.fromisoformat(start_date_s) if start_date_s else None
                end_date = date.fromisoformat(end_date_s) if end_date_s else None
                summary = it.get("summary") or None
                days_required = int(
                    it.get("days_required") or (0 if modality == "online" else 1)
                )
                pricing_options = _normalize_pricing_options(it.get("pricing_options"))
                if pricing_options:
                    try:
                        highest = max(
                            float(opt.get("cost_usd") or 0) for opt in pricing_options
                        )
                    except ValueError:
                        highest = 0.0
                    if highest and highest > cost:
                        cost = highest

                hybrid_flag = it.get("hybrid_available")
                if isinstance(hybrid_flag, str):
                    hybrid_available = hybrid_flag.strip().lower() in {
                        "true",
                        "1",
                        "yes",
                        "both",
                        "hybrid",
                        "online and in-person",
                    }
                else:
                    hybrid_available = bool(hybrid_flag)

                if not hybrid_available and "hybrid" in modality:
                    hybrid_available = True

                # Validation
                if not title or not provider or credits <= 0:
                    continue

                exists = s.exec(
                    select(Activity).where(
                        Activity.title == title, Activity.provider == provider
                    )
                ).first()
                if exists:
                    continue

                def _coerce_list(value) -> List[str]:
                    if not value:
                        return []
                    if isinstance(value, list):
                        return [str(v).strip() for v in value if str(v).strip()]
                    if isinstance(value, str):
                        v = value.strip()
                        return [v] if v else []
                    return []

                raw_eligibility = (
                    it.get("eligibility_text")
                    or it.get("snippet")
                    or it.get("summary")
                    or ""
                )
                eligibility_data = _parse_eligibility(raw_eligibility)

                incoming_institutions = _coerce_list(it.get("eligible_institutions"))
                if incoming_institutions:
                    eligibility_data["eligible_institutions"] = incoming_institutions
                    eligibility_data["open_to_public"] = False

                incoming_groups = _coerce_list(it.get("eligible_groups"))
                if incoming_groups:
                    eligibility_data["eligible_groups"] = incoming_groups
                    eligibility_data["open_to_public"] = False

                membership_req = it.get("membership_required")
                if membership_req:
                    eligibility_data["membership_required"] = str(
                        membership_req
                    ).strip()
                    eligibility_data["open_to_public"] = False

                if it.get("open_to_public") is not None:
                    eligibility_data["open_to_public"] = bool(it.get("open_to_public"))

                has_structured_restriction = bool(
                    eligibility_data["eligible_institutions"]
                    or eligibility_data["eligible_groups"]
                    or eligibility_data["membership_required"]
                )

                if has_structured_restriction:
                    parts: List[str] = []
                    if eligibility_data["eligible_institutions"]:
                        parts.append(
                            "Institutions: "
                            + ", ".join(eligibility_data["eligible_institutions"])
                        )
                    if eligibility_data["eligible_groups"]:
                        parts.append(
                            "Groups: " + ", ".join(eligibility_data["eligible_groups"])
                        )
                    if eligibility_data["membership_required"]:
                        parts.append(
                            "Membership: " + eligibility_data["membership_required"]
                        )
                    eligibility_text_value = "; ".join(parts) or (
                        eligibility_data["eligibility_text"] or raw_eligibility or None
                    )
                else:
                    eligibility_text_value = None

                source_val = (it.get("source") or "web").strip().lower() or "web"

                a = Activity(
                    title=title[:200],
                    provider=provider[:120],
                    credits=credits,
                    cost_usd=cost,
                    modality=(
                        "online"
                        if modality.startswith("on")
                        or modality in {"virtual", "asynchronous"}
                        else (
                            "live"
                            if (
                                modality == "live"
                                or "hybrid" in modality
                                or "in-person" in modality
                            )
                            else "online"
                        )
                    ),
                    city=city,
                    days_required=days_required,
                    start_date=start_date,
                    end_date=end_date,
                    url=url,
                    summary=summary,
                    source=source_val,
                    eligibility_text=eligibility_text_value,
                    eligible_institutions=eligibility_data["eligible_institutions"],
                    eligible_groups=eligibility_data["eligible_groups"],
                    membership_required=eligibility_data["membership_required"],
                    open_to_public=eligibility_data["open_to_public"],
                    hybrid_available=hybrid_available,
                    pricing_options=pricing_options,
                )
                s.add(a)
                added += 1
                by_source[source_val] = by_source.get(source_val, 0) + 1
            except Exception:
                logger.exception(
                    "insert skip (item=%s)",
                    {k: it.get(k) for k in ("title", "provider", "url")},
                )
                continue
        s.commit()
    return added, by_source


def _extract_with_openai_from_candidates(
    candidates: List[dict], count: int
) -> List[dict]:
    if not candidates:
        return []
    # If no API key, skip this pass and let deepfetch handle it
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("OPENAI_API_KEY not set; skipping first-pass AI extraction")
        return []
    client = _openai_client()
    try:
        content = (
            "You are extracting structured CME activities from candidate web results. "
            "Use the provided title/snippet/URL to infer provider (use domain if unsure) and credits if stated. "
            'Return STRICT JSON: {"items":[{"title":str,"provider":str,"credits":number,'
            '"cost_usd":number|null,"modality":"online|live","city":str|null,'
            '"start_date":"YYYY-MM-DD"|null,"end_date":"YYYY-MM-DD"|null,'
            '"days_required":0|1|2|null,"url":str,"summary":str|null,"hybrid_available":boolean|null,"pricing_options":array|null}]}. '
            f"Limit to about {count}. Candidates JSON follows.\n"
            f"{json.dumps(candidates, ensure_ascii=False)}"
        )
        resp = client.responses.create(model="gpt-4o-mini", input=content)
        text = getattr(resp, "output_text", "") or ""
        data = safe_json_loads(text)
        if not data:
            logger.warning("AI extraction JSON parse failed; skipping.")
            return []
        items = data.get("items") or []
        if not isinstance(items, list):
            return []
        # Filter/clean with fallbacks
        cleaned: List[dict] = []
        link_map = {(c.get("title") or "").strip(): c.get("link") for c in candidates}
        snippet_map = {
            (c.get("title") or "").strip(): c.get("snippet") for c in candidates
        }
        display_map = {
            (c.get("title") or "").strip(): c.get("displayLink") for c in candidates
        }
        for it in items:
            try:
                it = dict(it or {})
                title = (it.get("title") or "").strip()
                it.setdefault("url", link_map.get(title, ""))
                snippet = snippet_map.get(title, "") or ""
                lowered_snippet = snippet.lower()
                # Provider fallback to domain or displayLink
                if not it.get("provider"):
                    url_val = it.get("url") or ""
                    prov = (
                        urlparse(url_val).netloc
                        if url_val
                        else (display_map.get(title) or "")
                    )
                    if prov:
                        it["provider"] = prov
                if it.get("cost_usd") in (None, ""):
                    price_match = COST_RE.search(snippet)
                    if price_match:
                        try:
                            it["cost_usd"] = float(price_match.group(1))
                        except Exception:
                            pass
                    elif any(
                        phrase in lowered_snippet
                        for phrase in ("free", "no cost", "complimentary", "included")
                    ):
                        it["cost_usd"] = 0.0
                # Credits fallback from snippet/title heuristics
                if not it.get("credits") or float(it.get("credits") or 0) <= 0:
                    text_blob = f"{snippet} {title}"
                    m = CRED_RE.search(text_blob) or ALT_CRED_RE.search(text_blob)
                    if m:
                        it["credits"] = float(m.group(1))
                # Modality fallback
                if not it.get("modality"):
                    mm = MODALITY_RE.search(snippet)
                    it["modality"] = mm.group(0).lower() if mm else "online"
                if "eligibility_text" not in it or not it.get("eligibility_text"):
                    it["eligibility_text"] = snippet
                if "snippet" not in it or not it.get("snippet"):
                    it["snippet"] = snippet
                # Validate here
                credits = float(it.get("credits") or 0)
                provider = (it.get("provider") or "").strip()
                if credits <= 0 or not provider or not title:
                    continue
                eligibility_preview = _parse_eligibility(it.get("eligibility_text"))
                has_restriction = bool(
                    eligibility_preview["eligible_institutions"]
                    or eligibility_preview["eligible_groups"]
                    or eligibility_preview["membership_required"]
                )
                has_cost = it.get("cost_usd") not in (None, "")
                if not has_cost or not has_restriction:
                    it["_force_deepfetch"] = True
                cleaned.append(it)
            except Exception:
                logger.exception("drop item due to parse error")
                continue
        logger.info(
            "AI extracted from Google candidates: %d valid of %d",
            len(cleaned),
            len(items),
        )
        return cleaned
    except Exception:
        logger.exception("AI extraction failed")
        return []


def _fallback_with_openai_web_search(count: int) -> List[dict]:
    client = _openai_client()
    try:
        prompt = (
            "Find up-to-date CME opportunities in psychiatry that award AMA PRA Category 1 Credits. "
            "Return STRICT JSON with top-level 'items' only, with fields: title, provider, credits (number), "
            "cost_usd (number|null), modality ('online'|'live'), city (string|null), start_date (YYYY-MM-DD|null), "
            "end_date (YYYY-MM-DD|null), days_required (0|1|2|null), url, summary (string|null), "
            "hybrid_available (boolean|null), pricing_options (array), eligible_institutions (array|null), "
            "eligible_groups (array|null), membership_required (string|null), open_to_public (boolean|null)."
        )
        resp = client.responses.create(
            model="gpt-4o-mini", tools=[{"type": "web_search"}], input=prompt
        )
        text = getattr(resp, "output_text", "") or ""
        data = safe_json_loads(text) or {}
        items = data.get("items") or []
        if not isinstance(items, list):
            return []
        out: List[dict] = []
        for it in items:
            try:
                credits = float(it.get("credits") or 0)
                if credits <= 0:
                    continue
                it["credits"] = credits
                it.setdefault("source", "ai")
                out.append(it)
            except Exception:
                logger.exception("drop ai-fallback item parse error")
                continue
        return out
    except Exception:
        logger.exception("AI web_search fallback failed")
        return []


async def ingest_psychiatry_online_ai(count: int = 10, debug: bool = False):
    """Ingest CME using Google CSE (primary) with AI extraction; deep-fetch pages when snippets lack credits.
    Falls back to AI web_search only if still below threshold.

    Returns (ingested_count, used_fallback) by default, or a debug dict when debug=True.
    """
    total_added = 0
    used_fallback = False

    # for counters
    cse_raw = 0
    first_pass_valid = 0
    deepfetch_attempted = 0
    deepfetch_valid = 0

    try:
        with get_session() as s:
            user = s.exec(select(User)).first()
    except Exception:
        logger.exception("Failed to load user; proceeding with default queries")
        user = None

    # Google phase: candidates
    candidates: List[dict] = []
    try:
        if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
            queries = _build_queries_for_user(user)
            for q in queries:
                try:
                    batch = fetch_google_cse_multi(q, total=count * 2)
                    if batch:
                        # dedup by link across batches
                        seen = {c.get("link") for c in candidates}
                        for it in batch:
                            if it.get("link") not in seen:
                                candidates.append(it)
                except Exception:
                    logger.exception("Google query failed: %s", _redact_key(q))
            cse_raw = len(candidates)
            logger.info("Google candidates: %d", cse_raw)
        else:
            logger.warning("Google CSE env not configured; skipping primary search")
    except Exception:
        logger.exception("Google phase failed")

    # First extraction pass from candidates
    items: List[dict] = []
    try:
        if candidates:
            items = _extract_with_openai_from_candidates(candidates, count * 2)
        else:
            logger.warning("Google returned 0 candidates; will consider fallback")
    except Exception:
        logger.exception("First extraction pass failed")
        items = []

    # If first pass produced nothing, seed prelims from raw candidates for deepfetch
    if not items and candidates:
        prelim: List[dict] = []
        for c in candidates[:30]:  # cap
            prelim.append(
                {
                    "title": c.get("title") or "",
                    "url": c.get("link") or "",
                    "provider": c.get("displayLink")
                    or (urlparse(c.get("link") or "").netloc if c.get("link") else ""),
                    "credits": 0,
                    "eligibility_text": c.get("snippet") or "",
                    "snippet": c.get("snippet") or "",
                    "_force_deepfetch": True,
                }
            )
    else:
        # Validate items and count first-pass valid
        prelim = []
        for it in items:
            try:
                if is_valid_record(it):
                    first_pass_valid += 1
                    prelim.append(it)
                else:
                    prelim.append(it)  # keep partial for deepfetch
            except Exception:
                continue

    # Deep fetch pass for partials
    max_deepfetch = int(os.getenv("INGEST_MAX_DEEPFETCH", "20"))
    deepfetch_budget = max_deepfetch
    improved: List[dict] = []

    # Pre-load set for dedupe check on insert
    def key_of(r: dict) -> str:
        return (
            r.get("title", "").strip().lower()
            + "|"
            + (r.get("provider", "").strip().lower())
        )

    for it in prelim:
        try:
            force_deepfetch = bool(it.get("_force_deepfetch"))
            valid_initial = is_valid_record(it)
            if valid_initial and not force_deepfetch:
                improved.append(it)
                continue
            if deepfetch_budget <= 0:
                if valid_initial and not force_deepfetch:
                    improved.append(it)
                continue
            url = it.get("url") or it.get("link")
            if not url:
                continue
            deepfetch_budget -= 1
            deepfetch_attempted += 1
            html = fetch_page(url)
            if not html:
                continue
            text = html_to_text(html)
            if not text:
                continue
            r2 = await ai_extract_from_text(text, url=url)
            if r2 and is_valid_record(r2):
                if not r2.get("eligibility_text"):
                    r2["eligibility_text"] = it.get("eligibility_text") or it.get(
                        "snippet", ""
                    )
                if not r2.get("snippet"):
                    r2["snippet"] = it.get("snippet") or ""
                deepfetch_valid += 1
                improved.append(r2)
            elif valid_initial:
                improved.append(it)
        except Exception:
            logger.exception(
                "deepfetch failed for %s",
                _redact_key(str(it.get("url") or it.get("link") or "")),
            )
            continue

    # Insert valid, dedupe by (title, provider)
    seen_pairs = set()
    to_insert: List[dict] = []
    for r in improved:
        if not is_valid_record(r):
            continue
        key = key_of(r)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        to_insert.append(r)

    added_from_google, source_counts = _insert_items(to_insert)
    total_added += added_from_google
    cumulative_sources: Dict[str, int] = dict(source_counts)
    logger.info(
        "ingest counters: cse_raw=%d first_pass_valid=%d deepfetch_attempted=%d deepfetch_valid=%d inserted=%d",
        cse_raw,
        first_pass_valid,
        deepfetch_attempted,
        deepfetch_valid,
        total_added,
    )

    # Fallback if still below target
    min_results = int(os.getenv("INGEST_MIN_RESULTS", str(count)))
    if total_added < min_results:
        used_fallback = True
        logger.info("Fallback triggered: OpenAI web_search")
        try:
            more = _fallback_with_openai_web_search(min_results - total_added)
            added_more, fallback_counts = _insert_items(more)
            total_added += added_more
            for key, value in fallback_counts.items():
                cumulative_sources[key] = cumulative_sources.get(key, 0) + value
        except Exception:
            logger.exception("Fallback phase failed")

    if debug:
        return {
            "cse_raw": cse_raw,
            "first_pass_valid": first_pass_valid,
            "deepfetch_attempted": deepfetch_attempted,
            "deepfetch_valid": deepfetch_valid,
            "inserted": total_added,
            "fallback_used": used_fallback,
            "by_source": cumulative_sources,
        }

    web_count = cumulative_sources.get("web", 0) + cumulative_sources.get("seed", 0)
    ai_count = cumulative_sources.get("ai", 0)
    return total_added, used_fallback, web_count, ai_count
