"""
Script to extract ABPN Approved CC Activities from saved HTML file.
Uses BeautifulSoup to parse the structured table data.

Run with: python scripts/ingest_abpn_html.py
"""
import sys
from pathlib import Path
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.ingest import _insert_items
from app.models import Activity


def extract_abpn_activities(html_path: str) -> list[dict]:
    """
    Parse ABPN HTML and extract activities.
    
    Expected structure:
    <td id="td1" class="bodySM">Organization Name</td>
    <td id="td8" class="bodySM">
        <a class="link" href="URL">Product Name (modality)</a>
    </td>
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    
    soup = BeautifulSoup(html, "html.parser")
    
    activities = []
    
    # Find all rows in the table
    # Each row should have organization in td1 and product in td8
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
        url = link.get("href", "")
        
        if not product_name or len(product_name) < 5:
            continue
        
        # Parse modality from product name (often in parentheses)
        modality = "online"  # default
        if "(live" in product_name.lower():
            modality = "live"
        elif "(hybrid" in product_name.lower():
            modality = "hybrid"
        elif "(print" in product_name.lower():
            modality = "online"  # print materials are usually online access
        
        activity = {
            "title": product_name[:500],
            "provider": organization,
            "credits": 0.0,  # ABPN doesn't list credits in this view
            "cost_usd": 0.0,
            "modality": modality,
            "url": url[:2000] if url else None,
            "summary": f"ABPN-approved CC activity from {organization}",
            "source": "abpn",  # Special source tag for ABPN
            "open_to_public": True,
        }
        activities.append(activity)
    
    return activities


def main():
    html_path = Path(__file__).parent.parent / "data" / "abpn_raw.html"
    
    if not html_path.exists():
        # Also check root directory
        html_path = Path(__file__).parent.parent / "abpn_raw.html"
    
    if not html_path.exists():
        print(f"ERROR: Could not find abpn_raw.html")
        print(f"Checked: {html_path}")
        print("Please save the ABPN HTML file to 'data/abpn_raw.html' or project root")
        return
    
    print(f"Reading: {html_path}")
    
    activities = extract_abpn_activities(str(html_path))
    
    print(f"Extracted {len(activities)} ABPN-approved activities")
    
    if activities:
        # Show sample
        print("\nSample activities:")
        for act in activities[:5]:
            print(f"  - {act['title'][:60]}... ({act['provider']})")
        
        # Insert into database
        print(f"\nInserting into database...")
        inserted_count, stats = _insert_items(activities)
        print(f"New items inserted: {inserted_count}")
        print(f"Stats: {stats}")
    else:
        print("No activities found. Check the HTML structure.")


if __name__ == "__main__":
    main()
