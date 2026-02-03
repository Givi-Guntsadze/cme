"""
Script to fetch results from specific Apify runs and ingest them into the local database.
Uses AI (OpenAI) to extract structured CME activities from raw markdown content.

Run with: python scripts/ingest_apify_runs.py [run_id1] [run_id2] ...
If no run IDs provided, it can list recent successful runs.
"""
import asyncio
import sys
import argparse
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.services.scraper import fetch_actor_results, get_apify_token, ai_extract_activities_from_markdown
from app.ingest import _insert_items
from app.db import get_session
from sqlmodel import select
from app.models import Activity
import httpx


def load_abpn_urls() -> set:
    """Load ABPN-approved URLs to tag them correctly."""
    abpn_file = Path(__file__).parent.parent / "data" / "abpn_urls.txt"
    if not abpn_file.exists():
        return set()
    
    urls = set()
    with open(abpn_file, "r") as f:
        for line in f:
            url = line.strip()
            if url:
                # Normalize: store just the domain for matching
                try:
                    parsed = urlparse(url)
                    # Store the full URL for exact match
                    urls.add(url)
                    # Also store domain for partial matching
                    urls.add(parsed.netloc)
                except:
                    pass
    return urls


# Load ABPN URLs at module level
ABPN_URLS = load_abpn_urls()
print(f"Loaded {len(ABPN_URLS)} ABPN reference URLs for source tagging")


async def list_recent_runs(limit=10):
    token = get_apify_token()
    if not token:
        print("Error: APIFY_TOKEN not found.")
        return []
        
    url = "https://api.apify.com/v2/actor-runs"
    params = {"token": token, "desc": 1, "limit": limit, "status": "SUCCEEDED"}
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json().get("data", {}).get("items", [])
            print(f"\n--- Recent SUCCEEDED Runs (Last {limit}) ---")
            for run in data:
                print(f"ID: {run['id']} | Started: {run['startedAt']} | Status: {run['status']}")
            return [r['id'] for r in data]
        else:
            print(f"Failed to list runs: {resp.status_code}")
            return []


def is_abpn_url(url: str) -> bool:
    """Check if a URL is from the ABPN-approved list."""
    if url in ABPN_URLS:
        return True
    try:
        domain = urlparse(url).netloc
        if domain in ABPN_URLS:
            return True
    except:
        pass
    return False


async def process_runs(run_ids):
    total_new = 0
    total_extracted = 0
    
    for run_id in run_ids:
        print(f"\n{'='*60}")
        print(f"Processing Run ID: {run_id}")
        print('='*60)
        
        try:
            # 1. Fetch raw items from Apify
            raw_items = await fetch_actor_results(run_id)
            print(f"  Fetched {len(raw_items)} pages from Apify.")
            
            if not raw_items:
                print("  No items to process.")
                continue

            all_activities = []

            # 2. Process each page with AI extraction
            for i, item in enumerate(raw_items):
                page_url = item.get("url", "unknown")
                markdown = item.get("markdown") or item.get("text") or ""
                
                # Skip pages with no content
                if len(markdown) < 100:
                    continue
                
                # Get domain for provider attribution
                try:
                    domain = urlparse(page_url).netloc or "unknown"
                except:
                    domain = "unknown"

                # Determine source tag (abpn vs web)
                source_tag = "abpn" if is_abpn_url(page_url) else "web"

                print(f"  [{i+1}/{len(raw_items)}] Extracting from: {page_url[:60]}... [{source_tag}]", end=" ")
                
                # AI extraction with correct source tag
                activities = ai_extract_activities_from_markdown(
                    markdown=markdown,
                    page_url=page_url,
                    source_domain=domain,
                    source_tag=source_tag,
                )
                
                print(f"Found {len(activities)} activities")
                all_activities.extend(activities)
            
            print(f"\n  Total activities extracted: {len(all_activities)}")
            total_extracted += len(all_activities)

            # 3. Convert to dicts for insertion
            if all_activities:
                activities_dicts = [act.model_dump() for act in all_activities]
                
                # 4. Ingest into DB (handles deduplication)
                inserted_count, stats = _insert_items(activities_dicts)
                print(f"  New items inserted: {inserted_count}")
                print(f"  Stats: {stats}")
                total_new += inserted_count
            else:
                print("  No activities found to insert.")

        except Exception as e:
            print(f"Error processing run {run_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total activities extracted by AI: {total_extracted}")
    print(f"Total NEW activities inserted into database: {total_new}")

def main():
    parser = argparse.ArgumentParser(description="Ingest Apify runs into CME database with AI extraction")
    parser.add_argument("run_ids", metavar="RUN_ID", type=str, nargs="*", help="Apify Run IDs to ingest")
    parser.add_argument("--list", action="store_true", help="List recent successful runs")
    
    args = parser.parse_args()

    if args.list or not args.run_ids:
        asyncio.run(list_recent_runs())
        if not args.run_ids:
            print("\nUsage: python scripts/ingest_apify_runs.py [RUN_ID ...]")
            return

    if args.run_ids:
        asyncio.run(process_runs(args.run_ids))

if __name__ == "__main__":
    main()
