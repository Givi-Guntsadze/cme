"""
Bulk import script for CME sources.

Usage:
    python scripts/import_sources.py sources.txt
    python scripts/import_sources.py --url https://example.com/cme

Format of sources.txt:
One URL per line. Lines starting with # are ignored.
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import get_session
from app.models import ScrapeSource
from app.services.scheduler import refresh_sources
from sqlmodel import select

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_url(session, url: str) -> bool:
    """Import a single URL into ScrapeSource."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url.strip())
        domain = parsed.netloc or parsed.path.split("/")[0]
        if not domain:
            logger.warning(f"Skipping invalid URL: {url}")
            return False

        # Use full URL as the base for crawling
        base_url = url.strip()
        
        # Check duplicate
        existing = session.exec(
            select(ScrapeSource).where(ScrapeSource.domain == domain)
        ).first()
        
        if existing:
            logger.info(f"Skipping existing source: {domain}")
            return False
            
        new_source = ScrapeSource(domain=domain, url=base_url, enabled=True)
        session.add(new_source)
        logger.info(f"Added source: {domain}")
        return True
        
    except Exception as e:
        logger.error(f"Error importing {url}: {e}")
        return False


async def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    imported_count = 0
    
    with get_session() as session:
        # Handle file input
        if len(sys.argv) == 2 and not sys.argv[1].startswith("--"):
            filepath = Path(sys.argv[1])
            if not filepath.exists():
                print(f"File not found: {filepath}")
                return
                
            print(f"Reading sources from {filepath}...")
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    url = line.strip()
                    if url and not url.startswith("#"):
                        if import_url(session, url):
                            imported_count += 1
                            
        # Handle single URL flag
        elif len(sys.argv) == 3 and sys.argv[1] == "--url":
            if import_url(session, sys.argv[2]):
                imported_count += 1

        session.commit()
    
    print(f"\nImported {imported_count} new sources.")
    
    if imported_count > 0 or len(sys.argv) > 1:
        response = input("\nDo you want to run the scraper now? (y/n): ")
        if response.lower().startswith("y"):
            print("Starting scrape job (this may take a few minutes)...")
            await refresh_sources()
            print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
