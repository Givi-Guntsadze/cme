"""
Batch crawler script with MEMORY MANAGEMENT.
Triggers Apify crawls for sources in sources.txt, limiting concurrency to avoid 402 errors.
Run with: python scripts/crawl_all_sources.py
"""
import asyncio
import sys
import httpx
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.services.scraper import run_website_crawler, get_apify_token

# CONFIGURATION
MAX_CONCURRENT_RUNS = 3  # 3 * 8GB = 24GB (Safe under 32GB limit)
POLL_INTERVAL_SEC = 10   # Check status every 10 seconds

async def check_run_status(client, run_id, token):
    """Returns 'RUNNING', 'SUCCEEDED', 'FAILED', etc."""
    try:
        url = f"https://api.apify.com/v2/actor-runs/{run_id}"
        resp = await client.get(url, params={"token": token})
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return data.get("status")
    except Exception:
        pass
    return "UNKNOWN"

async def main():
    token = get_apify_token()
    if not token:
        print("ERROR: APIFY_TOKEN not found in .env")
        return

    # Use ABPN URLs instead of general sources
    sources_file = Path(__file__).parent.parent / "data" / "abpn_urls.txt"
    if not sources_file.exists():
        print(f"ERROR: {sources_file} not found")
        print("Run 'python scripts/extract_abpn_sources.py' first")
        return
    
    # Read URLs
    all_urls = []
    with open(sources_file, "r") as f:
        for line in f:
            url = line.strip()
            if url and not url.startswith("#"):
                all_urls.append(url)
    
    # Process all ABPN URLs
    # Support for batching via CLI args
    start_idx = 0
    end_idx = None
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    args, _ = parser.parse_known_args()
    
    start_idx = args.start
    end_idx = args.end
    
    # Slice the list
    if end_idx is not None:
        pending_urls = all_urls[start_idx:end_idx]
        print(f"Processing batch: indices {start_idx} to {end_idx} (Total: {len(pending_urls)})")
    else:
        pending_urls = all_urls[start_idx:]
        print(f"Processing batch: indices {start_idx} to END (Total: {len(pending_urls)})")
    
    print(f"Processing with MAX_CONCURRENT_RUNS = {MAX_CONCURRENT_RUNS}")
    print("=" * 60)

    active_runs = {}  # run_id -> url
    completed_runs = []
    failed_starts = []

    async with httpx.AsyncClient() as client:
        # Loop until everything is processed
        while pending_urls or active_runs:
            
            # 1. Fill slots if available
            while len(active_runs) < MAX_CONCURRENT_RUNS and pending_urls:
                url = pending_urls.pop(0)
                print(f"▶ Triggering: {url}")
                try:
                    result = await run_website_crawler(
                        urls=[url],
                        max_crawl_depth=2,
                        max_pages=30,
                    )
                    
                    if "run_id" in result:
                        run_id = result["run_id"]
                        print(f"   ✓ Started (ID: {run_id})")
                        active_runs[run_id] = url
                    else:
                        print(f"   ✗ Start Failed: {result.get('error')}")
                        failed_starts.append((url, result.get('error')))
                except Exception as e:
                    print(f"   ✗ Exception: {e}")
                    failed_starts.append((url, str(e)))
                
                # Tiny pause to be nice to API
                await asyncio.sleep(2)

            # 2. Poll active runs
            if not active_runs:
                break

            # Check statuses
            print(f"   [Status Check] Active: {len(active_runs)} | Pending: {len(pending_urls)} ... ", end="\r")
            
            finished_ids = []
            for run_id, url in active_runs.items():
                status = await check_run_status(client, run_id, token)
                if status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"]:
                    print(f"\n   ● Finished: {url} (Status: {status})")
                    completed_runs.append((url, status, run_id))
                    finished_ids.append(run_id)
            
            # Remove finished
            for rid in finished_ids:
                del active_runs[rid]
            
            # Wait before next poll
            await asyncio.sleep(POLL_INTERVAL_SEC)

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(completed_runs)} completed runs, {len(failed_starts)} failed to start")
    
    if failed_starts:
        print("\nFailed to start:")
        for url, err in failed_starts:
            print(f" - {url}: {err}")

    print("\nCheck Apify Console for verification of 'SUCCEEDED' runs.")

if __name__ == "__main__":
    asyncio.run(main())
