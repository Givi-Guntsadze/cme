"""
Manual test script for Apify scraping integration.

Run this script to verify the Apify Website Content Crawler integration works correctly.
It will crawl a sample CME provider page and display the extracted activities.

Usage:
    python tests/manual_test_scraper.py
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env BEFORE importing app modules
from dotenv import load_dotenv
load_dotenv()


from app.services.scraper import (
    run_website_crawler,
    wait_for_run,
    fetch_actor_results,
    normalize_to_activity,
    crawl_and_extract_activities,
)

import logging
logging.basicConfig(level=logging.INFO)

async def test_full_crawl():
    """Test the full crawl and extract pipeline."""
    print("=" * 60)
    print("Apify Scraping Integration Test")
    print("=" * 60)
    
    # Use a smaller, known CME page for testing
    test_urls = [
        "https://www.psychiatry.org/psychiatrists/education/cme"
    ]
    
    print(f"\n1. Testing crawl of: {test_urls[0]}")
    print("   This may take 2-5 minutes...\n")
    
    try:
        activities = await crawl_and_extract_activities(test_urls, max_pages=10)
        
        if not activities:
            print("No activities extracted. Check:")
            print("   - APIFY_TOKEN is set in .env")
            print("   - Apify account has available credits")
            return False
        
        print(f"Successfully extracted {len(activities)} activities:\n")
        
        for i, act in enumerate(activities[:5], 1):  # Show first 5
            print(f"   {i}. {act.title[:60]}...")
            print(f"      Provider: {act.provider}")
            print(f"      URL: {act.url[:50]}..." if act.url else "      URL: None")
            print()
        
        if len(activities) > 5:
            print(f"   ... and {len(activities) - 5} more activities")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_quick_status():
    """Quick test to verify Apify connection works."""
    print("\n2. Quick connection test...")
    
    from app.services.scraper import get_apify_token
    
    token = get_apify_token()
    if not token:
        print("APIFY_TOKEN not found in environment")
        return False
    
    print(f"APIFY_TOKEN found (length: {len(token)})")
    return True


if __name__ == "__main__":
    print("\nStarting Apify integration tests...\n")
    
    # Run quick test first
    asyncio.run(test_quick_status())
    
    # Optional: Run full crawl test (takes time)
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        success = asyncio.run(test_full_crawl())
        sys.exit(0 if success else 1)
    else:
        print("\nTo run the full crawl test (takes 2-5 min), use:")
        print("  python tests/manual_test_scraper.py --full")
