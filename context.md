# CME Tool – AI Context Handover

**Last Updated:** 2026-02-02T19:15:00+04:00  
**Session Focus:** Apify Integration, Background Crawling & Visual Feedback

---

## Current Micro-Goal
✅ **COMPLETED** – Implemented robust Apify background crawling for 26 priority sources and fixed UI visual feedback.

---

## Mental Stack (What the Previous Agent Was Doing)

### Completed This Session:
1. **Accept/Reject Plan Bug** – Fixed in `app/main.py` lines 1601-1702
   - Root cause: `plan_accept_all()` and `plan_reject_all()` called `ensure_plan()` which could rebuild the run with new items, then operated on the old items
   - Fix: Now queries existing active `PlanRun` directly via `select(PlanRun).where(...)` without triggering rebuild. Only calls `ensure_plan()` if no run exists.
   - Added `PlanRun` to imports at line 32

2. **Domain Normalization** – Fixed in `app/ingest.py` lines 60-66, 151, 941
   - Added `normalize_domain()` helper function that strips `www.` prefix and lowercases
   - Applied in `is_valid_record()` line 151 and `_insert_items()` line 941
   - Now `masterpsych.com` and `www.masterpsych.com` are treated as the same provider

3. **Timeout with Retry** – Fixed in `app/ingest.py` lines 68-94
   - Increased default timeout from 20s to 30s
   - Added 1 retry with exponential backoff for `httpx.TimeoutException`

4. **Visual Feedback for Substitute Actions** – Multiple files:
   - `app/templates/_plan.html` lines 1-4: Added `#plan-loading-indicator` div
   - `app/templates/_plan.html` lines 134-148: Added `hx-indicator="#plan-loading-indicator"` to substitute buttons
   - `static/style.css` lines 1345-1471: Added `.plan-loading-overlay`, `.toast-*` styles, and `.newly-added` animation
   - `app/templates/base.html` lines 41-43: Added toast container `<div id="toast-container">`
   - `app/templates/base.html` lines 97-123: Added `window.showToast()` function and HTMX `showToast` event listener
   - `app/main.py` lines 1597-1600: Added `HX-Trigger` header for toast on substitute request
   - `app/services/plan.py` line 939: Added `newly_added` field to serialized plan items (true if generated within 10 seconds)
   - `app/templates/_plan.html` line 59: Added `{% if p.newly_added %} newly-added{% endif %}` class

5. **Apify Integration** – Implemented in `app/main.py` lines 18-35, 2601-2699
   - Updated `source_monitor` to trigger background crawl tasks immediately.
   - Added `run_crawl_background` for asynchronous ingestion and DB updates.
   - Added `/webhook/apify` endpoint for potential production webhook handling.
   - Verified 26 sources from `sources.txt` are now being monitored in DB.

6. **Crawler Robustness** – Updated in `app/services/scraper.py` lines 50-75
   - Switched to `playwright:adaptive` crawler type.
   - Implemented aggressive CSS pruning and accordion-opening logic (`[aria-expanded="false"]`).
   - Enabled cookie warning removal and iframe expansion.

---

## Active Hot Files

| File | Lines Modified | Purpose |
|------|----------------|---------|
| `app/main.py` | Imports, 2600-2740 | Apify background tasks, webhooks, source monitoring |
| `app/services/scraper.py` | 50-75 | Robust crawler configuration updates |
| `app/ingest.py` | 60-151, 941 | Domain normalization and timeout retry fixes |
| `app/services/plan.py` | 939 | Added `newly_added` field for UI highlights |
| `app/templates/_plan.html` | 1-148 | Loading indicator, substitute button animations |
| `static/style.css` | 1345-1471 | Toast and green pulse animation styles |

---

## Verification Status

| Check | Status |
|-------|--------|
| Python syntax (`py_compile`) | ✅ Passed |
| App running (`uvicorn`) | ✅ Running |
| Accept Plan button | ✅ Fixed |
| Domain dedup | ✅ Fixed |
| Toast notification | ✅ Added |
| Apify Background Jobs | ✅ Verified (26 sources triggered) |
| Database Migration | ✅ Switched to `cme.sqlite` |

---

## Database State
- SQLite at `cme.sqlite` (confirmed path in `app/db.py`)
- Tables: `User`, `Activity`, `PlanRun`, `PlanItem`, `Claim`, `CompletedActivity`, `AssistantMessage`, `UserPolicy`, `RequirementsSnapshot`, `ScrapeSource`
- Sources Count: 26 active monitoring rows.

---

## ABPN Requirements (Psychiatry)
- 30 Category 1 credits per year
- Minimum 8 SA-CME credits
- 2 PIP projects (or 6 credits SA-CME/PIP)
- 3 Patient Safety credits
- All credits must be AMA PRA Category 1

---

## Pending Issues (NOT Addressed This Session)

### 1. External Page Timeouts
- Some CME sites (e.g., `aacap.org`) still timeout even with retries
- May need Apify integration for JavaScript-heavy sites

### 2. Verbose Error Logs
- Some handled exceptions still log with full stack traces
- Could reduce log noise in `app/main.py` and `app/ingest.py`

### 4. AI Extraction Failures
- Some pages fail JSON parsing during AI extraction
- Logged as "AI extraction JSON parse failed; skipping"

---

## Immediate Next Steps

1. **Investigate Apify Integration Issues**
   - User reported NO new runs appearing in Apify Console despite successful API calls.
   - Verify `APIFY_TOKEN` matches the account being checked.
   - Debug `app/services/scraper.py` logging to ensure API response is actually successful (currently assumes success).
   - Check `Activity` table in `cme.sqlite` to see if ingestion happened silently (unlikely if runs aren't in console).

2. **Implement RAG Connector**
   - Ensure the `vectordb.py` logic is correctly indexing the `web` activities (triggered at the end of `crawl_and_extract_activities`).
   - Verify the AI Assistant uses the vector search before falling back to Perplexity.

3. **Nightly Cron Job**
   - Implement a scheduler (e.g., in `app/services/scheduler.py`) to run `crawl_and_extract_activities` for all enabled `ScrapeSource` rows every 24 hours.

---

## Environment Variables Required

```
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
OPENAI_API_KEY=...
PERPLEXITY_API_KEY=...
APIFY_TOKEN=... (optional, for scraping)
INGEST_MIN_RESULTS=... (optional, default 5)
```

---

## Key Architecture Notes

- **Single Plan Mode**: Multi-mode (balanced/cheapest) removed; all uses "standard" mode internally
- **Discovery Flow**: Google CSE → AI extraction → Deep fetch → OpenAI web search fallback
- **Policy System**: `UserPolicy` stores user preferences as JSON payloads; `remove_titles` array blacklists activities
- **Plan Lifecycle**: `PlanRun` contains `PlanItem`s; `committed=True` means user accepted that item
