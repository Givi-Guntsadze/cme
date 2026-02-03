"""
Playwright script to scrape ABPN Approved CC Activities List.
Automates clicking through the dynamic UI to extract all activity URLs.

Run with: python scripts/scrape_abpn_playwright.py
Requires: pip install playwright && playwright install chromium
"""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright


async def scrape_abpn():
    """
    Navigate to ABPN page, click the button, check all boxes,
    and extract all activity rows from the table.
    """
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set True for headless
        page = await browser.new_page()
        
        print("Navigating to ABPN page...")
        await page.goto("https://abpn.org/maintain-certification/abpn-approved-products-list/")
        await page.wait_for_timeout(2000)
        
        # Click the "View ABPN Approved CC Activities List" button
        print("Clicking 'View ABPN Approved CC Activities List' button...")
        try:
            await page.click("text=View ABPN Approved CC Activities List")
            await page.wait_for_timeout(3000)
        except Exception as e:
            print(f"Could not find button: {e}")
            await browser.close()
            return
        
        # Wait for the modal/popup to appear
        print("Waiting for checkbox panel...")
        await page.wait_for_selector("input[type='checkbox']", timeout=10000)
        
        # Check "Applies to All Sub/Specialties" if available, otherwise check all boxes
        print("Selecting all subspecialties...")
        checkboxes = await page.query_selector_all("input[type='checkbox']")
        for checkbox in checkboxes:
            is_checked = await checkbox.is_checked()
            if not is_checked:
                await checkbox.click()
                await page.wait_for_timeout(200)
        
        # Wait for table to load
        print("Waiting for results table...")
        await page.wait_for_timeout(5000)
        
        # Extract all rows from the table
        print("Extracting activity data...")
        sources = []
        
        rows = await page.query_selector_all("tr")
        for row in rows:
            try:
                org_cell = await row.query_selector("td#td1")
                product_cell = await row.query_selector("td#td8")
                
                if not org_cell or not product_cell:
                    continue
                
                organization = await org_cell.inner_text()
                organization = organization.strip()
                
                link = await product_cell.query_selector("a.link")
                if not link:
                    continue
                
                product_name = await link.inner_text()
                url = await link.get_attribute("href")
                
                if url and url.startswith("http"):
                    sources.append({
                        "organization": organization,
                        "product_name": product_name.strip(),
                        "url": url.strip(),
                    })
            except Exception:
                continue
        
        await browser.close()
        
        # Deduplicate by URL
        seen = set()
        unique_sources = []
        for src in sources:
            if src["url"] not in seen:
                seen.add(src["url"])
                unique_sources.append(src)
        
        print(f"\nExtracted {len(unique_sources)} unique activities")
        
        # Save outputs
        json_path = output_dir / "abpn_sources.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unique_sources, f, indent=2)
        print(f"Saved: {json_path}")
        
        txt_path = output_dir / "abpn_urls.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for src in unique_sources:
                f.write(src["url"] + "\n")
        print(f"Saved: {txt_path}")
        
        return unique_sources


if __name__ == "__main__":
    asyncio.run(scrape_abpn())
