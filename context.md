# 🧠 AI Context Handover
**Last Updated:** 2026-02-02 16:36 (UTC+4)
**Status:** Just Fixed Critical Bug - Plan Generation Working Again

## 🎯 Current Micro-Goal
Fixed the "Discover activities" button not populating the Recommended Plan. The planner was crashing due to missing variable declarations.

## 🏗 The "Mental Stack" (LIFO)

### 1. JUST FIXED: Missing Variable Declarations in `planner.py`
- **File:** `app/planner.py`, function `build_plan()` (starts line 467)
- **Root Cause:** After a previous refactor to remove multi-plan modes (Balanced/Cheapest/Default → single plan), 4 variables were accidentally deleted or never existed:
  - `pricing_cache` (line 551)
  - `provider_counts` (line 552)
  - `modality_counts` (line 553)
  - `topic_counts` (line 554)
- **Error:** `NameError: name 'pricing_cache' is not defined` at line 587
- **Result:** Plan generation crashed silently, creating PlanRun with 0 PlanItems
- **Fix Applied:** Added these 4 variable initializations (commit `97b91f7`, pushed to GitHub)
- **Verification:** Direct test of `build_plan()` now returns 7 activities, 19.5 credits

### 2. KNOWN ISSUE: External Page Fetch Timeouts
- **Symptom:** `httpx.ConnectTimeout` errors when fetching `aacap.org` and other external CME sites
- **File:** `app/ingest.py`, function `fetch_page()` (line 60)
- **Current timeout:** 20 seconds (hardcoded)
- **Impact:** Some CME activities fail to be enriched during discovery
- **NOT YET ADDRESSED** - user asked to focus on the breaking bug first

### 3. PENDING: Apify Actor Integration
- User mentioned wanting to set up an Apify actor to scrape CME websites and build a knowledge base
- **NOT YET STARTED**

### 4. Active Plan Mode = Single Plan Only
- Multi-plan modes (Balanced/Cheapest/Default) were removed in a previous session
- App now operates with a single unified plan
- Related prior fix in `app/services/plan.py`: Added `open_to_public` check to skip profile matching for public activities (lines 759-767)

## 🚧 Active "Hot" Files
- `app/planner.py` - **JUST FIXED** - plan generation logic, scoring algorithms
- `app/services/plan.py` - PlanManager class, `_rebuild_run()` calls `build_plan()`
- `app/ingest.py` - Discovery/ingestion flow, Google CSE, Perplexity, OpenAI web_search
- `app/main.py` - FastAPI routes including `/ingest` endpoint

## � Database State
- **Activities in DB:** 36
- **Latest PlanRun:** id=131, status=active (should now have PlanItems after refreshing app)
- **User:** Single psychiatrist user with ABPN requirements

## 📝 ABPN Requirements (from `app/config/abpn_psychiatry_requirements.json`)
- 90 CME credits per 3-year cycle
- **16 SA-CME credits minimum** (Self-Assessment CME)
- **1 PIP project** (Performance Improvement in Practice)
- **Patient Safety activity required** (one-time, check portal)
- All SA-CME/PIP/Patient Safety must be ABPN-approved

## 💡 Key Architectural Decisions (Do Not Revert)
- **Single Plan Mode:** No more Balanced/Cheapest/Default selection
- **Discovery Flow:** Google CSE (primary) → AI extraction → Deep fetch → OpenAI web_search fallback
- **Policy System:** Still in place but simplified for single mode

## 📋 Immediate Next Steps (The To-Do List)
- [x] Fix `NameError: pricing_cache` in `build_plan()` - **DONE**
- [ ] Restart server and verify plan populates in UI (user to confirm)
- [ ] Address timeout issues in `fetch_page()` - increase timeout or add retries
- [ ] Investigate Apify actor integration for CME website scraping (user's next intention)
- [ ] Clean up verbose error logs for handled exceptions

## 🐛 Known Bugs / Blockers
1. **External site timeouts:** Some CME provider sites (e.g., aacap.org) timeout during discovery
2. **AI extraction failures:** Some pages fail JSON parse (logged as "AI extraction JSON parse failed; skipping")

## � How to Test Plan Generation
```powershell
cd c:\Users\user\Documents\CME\cme-tool
python -c "from app.planner import build_plan; from app.db import get_session; from sqlmodel import select; from app.models import User; s = get_session().__enter__(); u = s.exec(select(User)).first(); result = build_plan(u, s, mode='balanced'); print(f'Recommended: {len(result[0])}, Credits: {result[1]:.1f}')"
```
Expected output: `Recommended: 7, Credits: 19.5` (or similar)

## 🔗 Key Endpoints
- `/` - Main dashboard with CME plan
- `/ingest` - Triggers discovery (GET or POST)
- `/reenrich` - Re-enriches existing activities with Perplexity
- `/fragment/plan` - HTMX fragment for plan rendering
