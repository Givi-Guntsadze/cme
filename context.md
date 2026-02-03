# Project Context: ABPN-Focused CME Concierge
The user is building an AI agent to help Psychiatrists manage their ABPN Continuing Certification (CC) requirements. The immediate focus has been on **ingesting authoritative ABPN-approved activities** and making them searchable via an AI chat.

## Current State (Session Checkpoint)
We have a fully functional pipeline for crawling, ingesting, and searching ABPN activities.
- **Data Source**: 154 unique ABPN activity URLs extracted.
- **Crawling**: Smart-batch crawling (max 3 concurrent) implemented. First 20 sites have been crawled.
- **Ingestion**: 
    - `ingest_apify_runs.py` automatically tags activities from ABPN URLs with `source="abpn"`.
    - **69 activities** are currently in the database (49 properly tagged as ABPN).
- **Search & UI**: 
    - **Catalog Page**: `http://localhost:8000/catalog-page` allows browsing/filtering.
    - **Vector Search**: All SQLite activities have been indexed into ChromaDB.
    - **Chat**: The AI concierge can successfully find and search these activities (verified with "drug use/opioid" query).

## Mental Stack (Where we left off)
1.  **Pipeline verified**: The entire loop from "Raw HTML" -> "Crawler" -> "Sqlite" -> "Vector DB" -> "Chat" works.
2.  **Indexing Requirement**: Note that currently, SQLite -> Vector DB sync is manual. You must run `python scripts/index_activities_vectordb.py` after ingesting new runs to make them show up in chat.
3.  **Partial Data**: We still only have the first 20/154 ABPN sites crawled.

## Key Decisions
- **Source Tagging**: We modified `ingest_apify_runs.py` to check the URL against `data/abpn_urls.txt`. If it matches (exact or domain), the activity gets `source="abpn"`. This is critical for trust.
- **Vector Sync**: The chat tool (`search_activities`) queries ChromaDB, not SQLite. Therefore, keeping ChromaDB in sync is essential.

## Immediate Next Steps (To-Do List)
1.  **Run Full Crawl**: 
    - We stopped at 20 URLs for testing.
    - ACTION: Run `python scripts/crawl_all_sources.py` to process the remaining 134 ABPN URLs.
2.  **Ingest & Index**:
    - After the crawl finishes (check Apify console or `ingest_apify_runs.py --list`):
    - Run `python scripts/ingest_apify_runs.py [NEW_RUN_IDS]`
    - Run `python scripts/index_activities_vectordb.py` (Crucial step!)
3.  **Refine Ranking**:
    - Build logic to prioritize `source="abpn"` activities in the `planner.py` algorithm so they appear in top recommendations, not just chat search.

## Relevant Commands
- **Run Batch Crawler**: `python scripts/crawl_all_sources.py`
- **List Successful Runs**: `python scripts/ingest_apify_runs.py --list`
- **Ingest Specific Runs**: `python scripts/ingest_apify_runs.py RUN_ID1 RUN_ID2`
- **Sync Vector DB**: `python scripts/index_activities_vectordb.py` (Run this after ingestion!)
- **Browse Catalog**: `http://localhost:8000/catalog-page`
