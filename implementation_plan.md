# Implementation Plan: Chat Fixes & Advanced Scraping Architecture

## Goal
Improve the user interface for better chat accessibility, fix chat functionality, and implement a robust scraping pipeline using Apify with RAG capabilities.

## Prerequisites & Accounts
- **Apify**: You already have `APIFY_TOKEN` in `.env`. We use the public actor `apify/website-content-crawler`, so **no manual actor creation is required**.
- **Vector DB**: We use **ChromaDB** which runs locally on your machine. **No account required**.
- **Embeddings**: We use **HuggingFace** models (`sentence-transformers`) which run locally. **No account required**.

## 1. UI & Chat Improvements (Completed)
### Problem
- Chat is currently at the bottom of the page, requiring scrolling away from the plan.
- User reported messages not appearing/sending when trying to update activities.

### Proposed Changes
- **Layout Redesign**: Move Chat to a sticky sidebar (right side) or a floating drawer that stays visible while scrolling the plan.
- **Chat Debugging**:
  - Investigate HTMX `hx-post="/chat/send"` behavior.
  - Ensure the message list updates immediately upon sending.
  - Verify the activity update parser (`_extract_activity_update`) catches user intent correctly.

## 2. Apify Scraping Integration
### Architecture
- **Scraper**: Use Apify's **Website Content Crawler** (or similar Actor) to robustly scrape CME provider sites.
- **Triggering**: Call Apify API from Python (`app/services/scraper.py`).
- **Data Structure**: Standardize scraped data into our `Activity` model format.

## 3. Data Storage & Retrieval (RAG)
### Strategy
- **Vector Store**: Use **PGVector** (if moving to Postgres) or **ChromaDB/FAISS** (local) to store embeddings of scraped course descriptions/eligibility criteria.
- **RAG Flow**:
  1. User asks "Find me cheap opioid courses".
  2. Embed query -> Search Vector Store.
  3. Retrieve top matches -> Feed to LLM to structure the response.
- **Why RAG?**: Better than simple SQL LIKE searches for concepts like "CME for leadership" or answering complex eligibility questions.

## 4. Scheduled Updates
### Implementation
- **Scheduler**: Use `APScheduler` (Advanced Python Scheduler) running in the background.
- **Job**: `refresh_sources()` runs weekly (or user-defined interval).
- **Process**:
  1. Fetch all `Source` URLs.
  2. Trigger Apify run.
  3. Update DB with new/changed activities.
  4. Archive expired activities.

## 5. Dynamic Source Management
### Workflow
- **Discovery**: When `ingest.py` finds a high-quality "Web" result (Google Search), offer a "Monitor this site" action.
- **Action**: Clicking "Monitor" adds the domain to a `ScrapeSource` table.
- **Integration**: The Scheduled Update job includes these new sources in the next Apify run.

## Proposed File Structure Changes
```text
app/
├── services/
│   ├── scraper.py       # Apify integration
│   ├── scheduler.py     # APScheduler setup
│   └── vectordb.py      # RAG/Embedding logic
├── models.py            # Add Source/ScrapeSource models
└── main.py              # Mount scheduler on startup
```

## Next Steps
1. **Fix UI/Chat first** (High Priority - UX blocker).
2. **Set up Apify account/monitoring**.
3. **Build the Scraper Service**.
4. **Implement RAG/Search**.
