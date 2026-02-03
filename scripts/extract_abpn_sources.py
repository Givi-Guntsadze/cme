"""
Script to extract ABPN Approved CC Activities URLs from saved HTML file.
Creates a source list for subsequent Apify crawling.

Run with: python scripts/extract_abpn_sources.py
"""
import json
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_abpn_sources(html_path: str) -> list[dict]:
    """
    Parse ABPN HTML and extract source URLs.
    
    Returns list of dicts with: organization, product_name, url
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    
    soup = BeautifulSoup(html, "html.parser")
    
    sources = []
    seen_urls = set()  # Dedupe
    
    # Find all rows in the table
    rows = soup.find_all("tr")
    
    for row in rows:
        org_cell = row.find("td", {"id": "td1"})
        product_cell = row.find("td", {"id": "td8"})
        
        if not org_cell or not product_cell:
            continue
        
        organization = org_cell.get_text(strip=True)
        
        # Get the link inside product cell
        link = product_cell.find("a", class_="link")
        if not link:
            continue
        
        product_name = link.get_text(strip=True)
        url = link.get("href", "").strip()
        
        # Skip if no valid URL
        if not url or not url.startswith("http"):
            continue
        
        # Skip duplicates
        if url in seen_urls:
            continue
        seen_urls.add(url)
        
        sources.append({
            "organization": organization,
            "product_name": product_name,
            "url": url,
        })
    
    return sources


def main():
    html_path = Path(__file__).parent.parent / "abpn_raw.html"
    
    if not html_path.exists():
        html_path = Path(__file__).parent.parent / "data" / "abpn_raw.html"
    
    if not html_path.exists():
        print(f"ERROR: Could not find abpn_raw.html")
        return
    
    print(f"Reading: {html_path}")
    
    sources = extract_abpn_sources(str(html_path))
    
    print(f"Extracted {len(sources)} unique ABPN-approved activity URLs")
    
    # Group by domain for analysis
    domains = {}
    for src in sources:
        domain = urlparse(src["url"]).netloc
        domains[domain] = domains.get(domain, 0) + 1
    
    print(f"\nURLs by domain (top 15):")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:15]:
        print(f"  {count:3d} - {domain}")
    
    # Save as JSON for further processing
    output_json = Path(__file__).parent.parent / "data" / "abpn_sources.json"
    output_json.parent.mkdir(exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)
    print(f"\nSaved to: {output_json}")
    
    # Also save just the URLs for easy feeding to crawler
    output_txt = Path(__file__).parent.parent / "data" / "abpn_urls.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        for src in sources:
            f.write(src["url"] + "\n")
    print(f"URL list: {output_txt}")
    
    print(f"\nNext step: Run crawler on these {len(sources)} URLs")
    print("  python scripts/crawl_all_sources.py  (after updating sources.txt)")


if __name__ == "__main__":
    main()
