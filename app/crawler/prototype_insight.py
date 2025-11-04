"""
Prototype crawler + enricher for the Insight CME Institute sample provider.

The goal is to exercise the proposed catalog schema, highlight missing fields,
and demonstrate how raw HTML is converted into structured activity payloads.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
import json
import re

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "app.crawler"

DATA_DIR = Path(__file__).parent / "samples"
SAMPLE_HTML = DATA_DIR / "insight_psych.html"

MONTH_LOOKUP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

DATE_RANGE_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\s*[-&]\s*(\d{1,2}),\s*(\d{4})",
    re.I,
)

DATE_SIMPLE_PATTERN = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?",
    re.I,
)

PRICE_PATTERN = re.compile(r"\$([\d,]+(?:\.\d{2})?)")


@dataclass
class PricingTier:
    tier_name: str
    price_amount: float
    currency: str = "USD"
    price_type: str = "base"
    eligibility_notes: Optional[str] = None
    discount_end: Optional[date] = None


@dataclass
class CreditDetail:
    credit_type: str
    credit_quantity: float
    notes: Optional[str] = None


@dataclass
class EligibilityRequirement:
    requirement_type: str
    requirement_value: str
    notes: Optional[str] = None


@dataclass
class CommitmentDetail:
    seat_time_hours: Optional[float] = None
    total_time_hours: Optional[float] = None
    completion_window_days: Optional[int] = None
    notes: Optional[str] = None


@dataclass
class NormalizedActivity:
    provider_id: str
    provider_name: str
    activity_title: str
    canonical_url: str
    modality: str
    format: str
    start_date: Optional[date]
    end_date: Optional[date]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    summary: str
    credit_details: List[CreditDetail] = field(default_factory=list)
    pricing_tiers: List[PricingTier] = field(default_factory=list)
    eligibility: List[EligibilityRequirement] = field(default_factory=list)
    commitment: CommitmentDetail = field(default_factory=CommitmentDetail)
    requirement_codes: List[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    field_gaps: List[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        """Convert to schema-friendly payload."""
        return {
            "activity": {
                "provider_slug": self.provider_id,
                "provider_name": self.provider_name,
                "title": self.activity_title,
                "url": self.canonical_url,
                "modality": self.modality,
                "format": self.format,
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "city": self.city,
                "state": self.state,
                "country": self.country,
                "summary": self.summary,
            },
            "credit_types": [asdict(c) for c in self.credit_details],
            "pricing_tiers": [asdict(p) for p in self.pricing_tiers],
            "eligibility": [asdict(e) for e in self.eligibility],
            "commitment": asdict(self.commitment),
            "requirement_mappings": self.requirement_codes,
            "provenance": self.provenance,
            "field_gaps": self.field_gaps,
        }


def load_sample_html(path: Path = SAMPLE_HTML) -> str:
    return path.read_text(encoding="utf-8")


def fetch_remote_html(source_url: str, timeout: int = 20) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CMECatalogBot/1.0)"}
        resp = httpx.get(
            source_url, timeout=timeout, follow_redirects=True, headers=headers
        )
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.warning("Remote fetch failed for %s: %s", source_url, exc)
        return None


def expand_date_ranges(text: str) -> str:
    def replace_range(match: re.Match[str]) -> str:
        month, start_day, end_day, year = match.groups()
        return f"{month} {start_day}, {year} and " f"{month} {end_day}, {year}"

    return DATE_RANGE_PATTERN.sub(replace_range, text)


def extract_dates(text: str) -> List[date]:
    normalized = expand_date_ranges(text)
    normalized = normalized.replace("&", " and ")
    dates: List[date] = []
    last_year: Optional[int] = None

    for match in DATE_SIMPLE_PATTERN.finditer(normalized):
        month_name, day_raw, year_raw = match.groups()
        month_num = MONTH_LOOKUP[month_name.lower()]
        day = int(day_raw)
        if year_raw:
            year = int(year_raw)
            last_year = year
        elif last_year:
            year = last_year
        else:
            # If no year is visible anywhere yet, skip this date.
            continue
        try:
            dates.append(date(year, month_num, day))
        except ValueError:
            continue
    return dates


def normalize_modality(raw: Optional[str]) -> str:
    value = (raw or "").strip().lower()
    if value in {"hybrid", "online", "live"}:
        return value
    if "virtual" in value and "live" in value:
        return "live"
    if "self-paced" in value or "on-demand" in value:
        return "online"
    return "live"


def parse_location(
    segments: Iterable[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    for segment in reversed(list(segments)):
        cleaned = segment.strip()
        cleaned = cleaned.replace("•", " ")
        lowered = cleaned.lower()
        for marker in (" in ", " at ", " on "):
            if marker in lowered:
                idx = lowered.rfind(marker)
                cleaned = cleaned[idx + len(marker) :]
                break
        cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()
        if "," in cleaned:
            city_part, state_part = cleaned.rsplit(",", 1)
            city = city_part.strip()
            state = re.sub(r"[^A-Za-z]", "", state_part.strip().split(" ")[0])
            if len(state) == 2 and state.isalpha():
                return city or None, state.upper(), "USA"
    return None, None, None


def parse_pricing_items(items: List[str]) -> List[PricingTier]:
    tiers: List[PricingTier] = []
    for item in items:
        prices = PRICE_PATTERN.findall(item)
        if not prices:
            continue
        primary_price = float(prices[0].replace(",", ""))
        tier_name = (
            item.split(":")[0].strip() if ":" in item else item.split("(")[0].strip()
        )
        tier = PricingTier(
            tier_name=tier_name or "General",
            price_amount=primary_price,
            eligibility_notes=item.strip(),
        )
        # If there is a "thru" or "until" clause, attempt to capture discount end date.
        discount_match = DATE_SIMPLE_PATTERN.search(item)
        if discount_match:
            month_name, day_raw, year_raw = discount_match.groups()
            if year_raw:
                tier.discount_end = date(
                    int(year_raw),
                    MONTH_LOOKUP[month_name.lower()],
                    int(day_raw),
                )
        tiers.append(tier)

        # Additional embedded prices (e.g., member pricing)
        if len(prices) > 1:
            for idx, price in enumerate(prices[1:], start=1):
                tiers.append(
                    PricingTier(
                        tier_name=f"Alt price {idx}",
                        price_amount=float(price.replace(",", "")),
                        eligibility_notes=f"Derived from: {item.strip()}",
                    )
                )
    return tiers


def parse_commitment(text: Optional[str]) -> CommitmentDetail:
    if not text:
        return CommitmentDetail()
    hours = [
        float(x.replace(",", "")) for x in re.findall(r"(\d+(?:\.\d+)?)\s*hours", text)
    ]
    seat_time = hours[0] if hours else None
    total_time = sum(hours) if hours else None
    return CommitmentDetail(
        seat_time_hours=seat_time,
        total_time_hours=total_time,
        notes=text.strip(),
    )


def parse_eligibility(text: Optional[str]) -> List[EligibilityRequirement]:
    if not text:
        return []
    chunks = [c.strip() for c in re.split(r"[.;]", text) if c.strip()]
    requirements: List[EligibilityRequirement] = []
    for chunk in chunks:
        if "licensed" in chunk.lower():
            requirements.append(
                EligibilityRequirement(
                    requirement_type="licensure",
                    requirement_value="active medical license",
                    notes=chunk,
                )
            )
        elif "membership" in chunk.lower():
            requirements.append(
                EligibilityRequirement(
                    requirement_type="membership",
                    requirement_value="AACME membership",
                    notes=chunk,
                )
            )
        elif "patient panel" in chunk.lower():
            requirements.append(
                EligibilityRequirement(
                    requirement_type="practice_status",
                    requirement_value="active patient panel",
                    notes=chunk,
                )
            )
        else:
            requirements.append(
                EligibilityRequirement(
                    requirement_type="other",
                    requirement_value=chunk,
                )
            )
    return requirements


def infer_requirement_codes(credit_type: str) -> List[str]:
    normalized = credit_type.lower()
    if "sa-cme" in normalized:
        return ["abpn.sa_cme"]
    if "pip" in normalized:
        return ["abpn.pip"]
    if "patient safety" in normalized:
        return ["abpn.patient_safety"]
    return ["abpn.general"]


def build_activity_records(html: str) -> List[NormalizedActivity]:
    soup = BeautifulSoup(html, "lxml")
    section = soup.select_one("#catalog")
    provider_name = (
        section["data-provider-name"]
        if section and section.has_attr("data-provider-name")
        else "Unknown Provider"
    )
    provider_slug = re.sub(r"[^a-z0-9]+", "-", provider_name.lower()).strip("-")

    activities: List[NormalizedActivity] = []
    cards = soup.select(".course-card")
    for card in cards:
        title_el = card.select_one(".course-title a")
        summary_el = card.select_one(".course-summary")
        dates_text = (
            card.select_one(".course-dates").get_text(" ", strip=True)
            if card.select_one(".course-dates")
            else ""
        )
        timing_text = (
            card.select_one(".course-timing").get_text(" ", strip=True)
            if card.select_one(".course-timing")
            else None
        )
        eligibility_text = (
            card.select_one(".course-eligibility").get_text(" ", strip=True)
            if card.select_one(".course-eligibility")
            else None
        )

        date_candidates = extract_dates(dates_text)
        start_date = date_candidates[0] if date_candidates else None
        end_date = date_candidates[-1] if len(date_candidates) > 1 else start_date

        segments = [seg for seg in dates_text.split("•") if seg.strip()]
        city, state, country = parse_location(segments)

        pricing_items = [
            li.get_text(" ", strip=True) for li in card.select(".course-pricing li")
        ]
        pricing_tiers = parse_pricing_items(pricing_items)

        credit_quantity = float(card.get("data-credit-quantity", "0") or 0)
        credit_type = card.get("data-credit-type", "AMA PRA Cat 1")
        credit_notes = ""
        if (
            summary_el
            and "patient-safety" in summary_el.get_text(" ", strip=True).lower()
        ):
            credit_notes = "Includes patient safety content"

        activity = NormalizedActivity(
            provider_id=provider_slug,
            provider_name=provider_name,
            activity_title=(
                title_el.get_text(strip=True) if title_el else "Untitled Activity"
            ),
            canonical_url=(
                title_el["href"] if title_el and title_el.has_attr("href") else ""
            ),
            modality=normalize_modality(card.get("data-modality")),
            format=(card.get("data-format") or "conference").lower(),
            start_date=start_date,
            end_date=end_date,
            city=city,
            state=state,
            country=country,
            summary=summary_el.get_text(" ", strip=True) if summary_el else "",
            credit_details=(
                [
                    CreditDetail(
                        credit_type=credit_type,
                        credit_quantity=credit_quantity,
                        notes=credit_notes or None,
                    )
                ]
                if credit_quantity
                else []
            ),
            pricing_tiers=pricing_tiers,
            eligibility=parse_eligibility(eligibility_text),
            commitment=parse_commitment(timing_text),
            requirement_codes=infer_requirement_codes(credit_type),
            provenance={
                "source_type": "html",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "source_hint": (
                    title_el["href"]
                    if title_el and title_el.has_attr("href")
                    else "sample"
                ),
            },
        )

        activity.field_gaps = identify_gaps(activity)
        activities.append(activity)
    return activities


def identify_gaps(activity: NormalizedActivity) -> List[str]:
    gaps: List[str] = []
    if not activity.start_date:
        gaps.append("start_date")
    if not activity.end_date:
        gaps.append("end_date")
    if not activity.city or not activity.state:
        gaps.append("location")
    if not activity.pricing_tiers:
        gaps.append("pricing_tiers")
    if not activity.credit_details:
        gaps.append("credit_details")
    if activity.commitment.total_time_hours is None:
        gaps.append("commitment.total_time_hours")
    if not activity.eligibility and activity.modality != "online":
        gaps.append("eligibility")
    return gaps


def run_prototype(
    source_url: Optional[str] = None,
    *,
    sync_db: bool = False,
    suppress_output: bool = False,
) -> List[dict]:
    if source_url:
        html = fetch_remote_html(source_url)
        if not html:
            logger.warning("Falling back to bundled sample HTML after fetch failure.")
    else:
        html = None
    if not source_url:
        source_url = os.getenv("INSIGHT_CME_SOURCE_URL")
        if source_url and not html:
            html = fetch_remote_html(source_url)
            if not html:
                logger.warning(
                    "Falling back to bundled sample HTML after env fetch failure."
                )
                source_url = None
    if not html:
        html = load_sample_html()

    activities = build_activity_records(html)
    payload = [activity.to_payload() for activity in activities]

    if sync_db:
        from app.catalog.bridge import sync_catalog_to_activity_table
        from app.catalog.service import upsert_activity_bundle
        from app.db import create_db_and_tables, get_session

        create_db_and_tables()
        with get_session() as session:
            for bundle in payload:
                upsert_activity_bundle(session, bundle)
            session.commit()
            synced = sync_catalog_to_activity_table(session)
            logger.info(
                "Catalog sync applied %d rows into legacy Activity table", synced
            )

    if not suppress_output:
        print(json.dumps(payload, indent=2, default=str))

    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototype crawl & enrichment for Insight CME Institute."
    )
    parser.add_argument(
        "--source-url",
        help="Optional live HTML endpoint (Apify actor, provider listing, etc.).",
    )
    parser.add_argument(
        "--sync-db",
        action="store_true",
        help="Persist normalized activities into the catalog tables and legacy Activity table.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress JSON output (useful when only syncing to the database).",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    run_prototype(
        args.source_url,
        sync_db=args.sync_db,
        suppress_output=args.no_print,
    )
