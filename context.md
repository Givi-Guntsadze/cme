# ABPN Data Ingestion & Stabilization (Current Phase)

## ✅ Completed Tasks (Confirmed)
- **Batch 1-4 Complete**: Source URLs at indices 0-80 have been successfully crawled, ingested, and indexed.
- **Current Database Count**: **450 Activities** are live and searchable.
- **UI Fixes Deployed**: 
  - **Catalog**: Implemented pagination (50 items/page), Previous/Next controls, and preserved filter state.
  - **Navigation**: Added "📚 Browse Catalog" button to the main plan page.
  - **Plan Logic**: Fixed "Reject" button behavior to simply remove the item without triggering a full plan rebuild (prevents unwanted substitutes).
  - **Styling**: Added dark mode support for pagination controls.

## 🧠 Mental Stack (Resume Point)
- **Context**: We are incrementally processing the `data/abpn_urls.txt` list in batches of 20 to manage API limits and stability.
- **Last Completed Action**: Batch 4 (URLs 60-80) finished ingestion and indexing.
- **Next Logical Batch**: Batch 5 (URLs 80-100).
- **Total Progress**: ~52% complete (80 out of ~155 URLs).

## ⏭️ Next Steps (Execute Immediately)
1. **Start Batch 5 Crawl (URLs 80-100)**:
   - Run command: `python scripts/crawl_all_sources.py --start 80 --end 100`
   - *Note: This will trigger ~20 runs on Apify. Wait for them to complete.*

2. **Ingest Batch 5 Results**:
   - List the successful runs: `python scripts/ingest_apify_runs.py --list --limit 20`
   - Run the ingestion command with the new Run IDs.

3. **Update Vector Index**:
   - Run command: `python scripts/index_activities_vectordb.py`

4. **Verify Application**:
   - Start server: `uvicorn app.main:app --reload`
   - Visit `http://localhost:8000/catalog-page` to verify the new activity count and pagination.

## ⚠️ Critical Operational Notes (Do Not Ignore)
- **Batch Size**: Stick to **20 URLs per batch**. Increasing this risks timeouts or hitting `MAX_CONCURRENT_RUNS` limits on Apify.
- **Database Concurrency**: While `uvicorn` (reader) and ingestion scripts (writer) can technically run together, it is safer to stop `uvicorn` during heavy ingestion to prevent `database is locked` errors.
- **Deduplication**: The system automatically handles duplicate URLs. If a batch overlaps with previous ones, it will skip existing entries safely.
