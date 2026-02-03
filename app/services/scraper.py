"""
Apify scraping integration service.

Uses Apify's Website Content Crawler Actor to robustly scrape CME provider sites.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import date
from typing import Any

import httpx

from ..models import Activity
from ..env import get_secret

logger = logging.getLogger(__name__)

APIFY_API_BASE = "https://api.apify.com/v2"
WEBSITE_CONTENT_CRAWLER_ACTOR = "apify~website-content-crawler"


def get_apify_token() -> str | None:
    """Get Apify API token from environment."""
    return get_secret("APIFY_TOKEN")


async def run_website_crawler(
    urls: list[str],
    max_crawl_depth: int = 2,
    max_pages: int = 50,
) -> dict[str, Any]:
    """
    Trigger Apify Website Content Crawler Actor.

    Args:
        urls: List of start URLs to crawl
        max_crawl_depth: How deep to follow links (default: 2)
        max_pages: Maximum pages to crawl per run (default: 50)

    Returns:
        Dict with 'run_id' and 'status' on success, or 'error' on failure
    """
    token = get_apify_token()
    if not token:
        logger.error("APIFY_TOKEN not configured")
        return {"error": "APIFY_TOKEN not configured"}

    # Debug: Log token prefix to verify it's loaded
    logger.info("Using APIFY_TOKEN: %s...", token[:8] if len(token) > 8 else "SHORT")

    actor_input = {
        "startUrls": [{"url": url} for url in urls],
        "maxCrawlDepth": max_crawl_depth,
        "maxPagesPerCrawl": max_pages,
        # GENTLE SETTINGS - prioritize compatibility over cleanliness
        "crawlerType": "playwright:firefox",  # More stable for legacy sites
        "proxyConfiguration": {"useApifyProxy": True},
        # Minimal stripping - let LLM handle the noise
        "removeElementsCssSelector": "script, style, noscript, svg",
        "clickElementsCssSelector": "",  # Don't try to expand things
        "htmlTransformer": "readableText",
        "readableTextCharThreshold": 50,  # Lower threshold = more content
        "aggressivePrune": False,
        "debugMode": False,
        "debugLog": False,
        "saveHtml": False,
        "saveMarkdown": True,
        "saveFiles": False,
        "saveScreenshots": False,
        "maxScrollHeightPixels": 3000,  # Less scrolling = fewer timeouts
        "dynamicContentWaitSecs": 15,   # Wait longer for slow ASP.NET sites
        "removeCookieWarnings": True,
        "expandIframes": False,  # Iframes often cause hangs
    }

    api_url = f"{APIFY_API_BASE}/acts/{WEBSITE_CONTENT_CRAWLER_ACTOR}/runs"
    logger.info("Calling Apify API: %s", api_url)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Start the actor run
            response = await client.post(
                api_url,
                params={"token": token},
                json=actor_input,
            )
            
            # Debug: Log full response details
            logger.info("Apify response status: %s", response.status_code)
            logger.debug("Apify response body: %s", response.text[:500] if response.text else "empty")
            
            response.raise_for_status()
            data = response.json()
            run_id = data.get("data", {}).get("id")

            if not run_id:
                logger.error("No run_id returned from Apify: %s", data)
                return {"error": "No run_id returned"}

            logger.info("Started Apify run: %s for URLs: %s", run_id, urls)
            return {
                "run_id": run_id,
                "status": data.get("data", {}).get("status", "RUNNING"),
            }
        except httpx.HTTPStatusError as e:
            logger.error("Apify API error: %s - %s", e.response.status_code, e.response.text)
            return {"error": f"API error: {e.response.status_code}"}
        except Exception as e:
            logger.exception("Failed to start Apify run")
            return {"error": str(e)}


async def get_run_status(run_id: str) -> dict[str, Any]:
    """Check the status of an Apify run."""
    token = get_apify_token()
    if not token:
        return {"error": "APIFY_TOKEN not configured"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                f"{APIFY_API_BASE}/actor-runs/{run_id}",
                params={"token": token},
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            return {
                "run_id": run_id,
                "status": data.get("status"),
                "dataset_id": data.get("defaultDatasetId"),
            }
        except Exception as e:
            logger.exception("Failed to get run status for %s", run_id)
            return {"error": str(e)}


async def wait_for_run(run_id: str, poll_interval: float = 5.0, timeout: float = 600.0) -> dict[str, Any]:
    """
    Wait for an Apify run to complete.

    Args:
        run_id: The Apify run ID
        poll_interval: Seconds between status checks
        timeout: Maximum seconds to wait

    Returns:
        Final run status dict
    """
    elapsed = 0.0
    while elapsed < timeout:
        status = await get_run_status(run_id)
        if status.get("error"):
            return status

        run_status = status.get("status", "")
        if run_status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
            return status

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    return {"error": "Timeout waiting for run to complete", "run_id": run_id}


async def fetch_actor_results(run_id: str) -> list[dict[str, Any]]:
    """
    Retrieve results from a completed Apify run.

    Args:
        run_id: The Apify run ID

    Returns:
        List of scraped page data dicts
    """
    token = get_apify_token()
    if not token:
        return []

    # First get the dataset ID from the run
    status = await get_run_status(run_id)
    dataset_id = status.get("dataset_id")
    if not dataset_id:
        logger.error("No dataset_id found for run %s", run_id)
        return []

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(
                f"{APIFY_API_BASE}/datasets/{dataset_id}/items",
                params={"token": token, "format": "json"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.exception("Failed to fetch results for run %s", run_id)
            return []


def normalize_to_activity(scraped_data: dict[str, Any], source_domain: str = "web") -> Activity | None:
    """
    Map Apify Website Content Crawler output to Activity model.

    The crawler returns items with: url, title, text, markdown, etc.
    We try to extract structured CME activity data.

    Args:
        scraped_data: Single item from Apify results
        source_domain: Domain source for the activity

    Returns:
        Activity instance or None if data is insufficient
    """
    url = scraped_data.get("url", "")
    title = scraped_data.get("title", "").strip()
    text = scraped_data.get("text", "") or scraped_data.get("markdown", "")

    if not title or len(title) < 10:
        return None

    # Default values - these would be enriched by AI/Perplexity later
    return Activity(
        title=title[:500],
        provider=source_domain,
        credits=0.0,  # To be enriched
        cost_usd=0.0,  # To be enriched
        modality="online",  # Default, to be enriched
        url=url[:2000] if url else None,
        summary=text[:1000] if text else None,
        source="web",
        open_to_public=True,
    )


def ai_extract_activities_from_markdown(
    markdown: str,
    page_url: str,
    source_domain: str = "web",
    source_tag: str = "web",
) -> list[Activity]:
    """
    Use OpenAI to extract structured CME activities from raw markdown content.

    Args:
        markdown: Raw markdown/text from a scraped page
        page_url: URL of the source page
        source_domain: Domain for provider attribution

    Returns:
        List of Activity objects extracted by the LLM
    """
    import json
    from openai import OpenAI

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping AI extraction")
        return []

    # Truncate markdown to avoid token limits (roughly 12k chars ~ 3k tokens)
    truncated = markdown[:12000] if len(markdown) > 12000 else markdown

    prompt = f"""You are a CME (Continuing Medical Education) activity extractor.

Analyze the following webpage content and extract all CME activities/courses you can find.

For each activity, extract:
- title: The name of the CME activity/course
- credits: Number of CME credits (as a float, e.g., 1.0, 2.5). Use 0 if not specified.
- credit_type: Type of credits (e.g., "AMA PRA Category 1", "SA-CME", "MOC", "Patient Safety"). Use "CME" if not specified.
- cost_usd: Cost in USD (as a float). Use 0 if free or not specified.
- modality: "online", "live", or "hybrid"
- description: Brief summary of the activity (1-2 sentences)
- url: Direct URL to the activity if found in the content, otherwise null

Return a JSON array of objects. If no CME activities are found, return an empty array [].
Only include actual educational activities, not general website pages or navigation content.

SOURCE URL: {page_url}

WEBPAGE CONTENT:
{truncated}

RESPOND ONLY WITH VALID JSON (no markdown, no explanation):"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap for extraction
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
        )

        raw_output = response.choices[0].message.content.strip()

        # Clean potential markdown wrapping
        if raw_output.startswith("```"):
            lines = raw_output.split("\n")
            raw_output = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        extracted = json.loads(raw_output)
        if not isinstance(extracted, list):
            extracted = [extracted] if extracted else []

        activities = []
        for item in extracted:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()
            if not title or len(title) < 5:
                continue

            activity = Activity(
                title=title[:500],
                provider=source_domain,
                credits=float(item.get("credits", 0) or 0),
                cost_usd=float(item.get("cost_usd", 0) or 0),
                modality=item.get("modality", "online"),
                url=item.get("url") or page_url,
                summary=item.get("description", "")[:1000] if item.get("description") else None,
                source=source_tag,
                open_to_public=True,
            )
            activities.append(activity)

        logger.info("AI extracted %d activities from %s", len(activities), page_url)
        return activities

    except json.JSONDecodeError as e:
        logger.warning("AI extraction JSON parse failed for %s: %s", page_url, e)
        return []
    except Exception as e:
        logger.exception("AI extraction failed for %s: %s", page_url, e)
        return []


async def crawl_and_extract_activities(
    urls: list[str],
    max_pages: int = 30,
) -> list[Activity]:
    """
    High-level function: crawl URLs and return extracted activities.

    Args:
        urls: List of CME provider URLs to crawl
        max_pages: Max pages per run

    Returns:
        List of Activity objects (need enrichment for credits/cost)
    """
    result = await run_website_crawler(urls, max_pages=max_pages)

    if result.get("error"):
        logger.error("Crawl failed: %s", result["error"])
        return []

    run_id = result.get("run_id")
    if not run_id:
        return []

    # Wait for completion
    final_status = await wait_for_run(run_id)
    if final_status.get("status") != "SUCCEEDED":
        logger.error("Crawl run did not succeed: %s", final_status)
        return []

    # Fetch and normalize results
    raw_items = await fetch_actor_results(run_id)
    activities = []

    for item in raw_items:
        url = item.get("url", "")
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc or "unknown"
        except Exception:
            domain = "unknown"

        activity = normalize_to_activity(item, source_domain=domain)
        if activity:
            activities.append(activity)

    logger.info("Extracted %d activities from %d pages", len(activities), len(raw_items))
    
    # Index into vector store for RAG search
    if activities:
        try:
            from . import vectordb
            activity_dicts = [
                {
                    "id": f"{act.provider}_{hash(act.title) % 10**8}",
                    "title": act.title,
                    "description": act.summary or "",
                    "provider": act.provider or "",
                    "credits": act.credits or 0,
                    "cost": act.cost_usd or 0,
                    "tags": [],
                }
                for act in activities
            ]
            indexed_count = vectordb.add_activities(activity_dicts)
            logger.info("Indexed %d activities into vector store", indexed_count)
        except Exception as e:
            logger.warning("Failed to index activities into vector store: %s", e)
    
    return activities
