# CME Discovery & Retrieval Notes

## ERD Overview
- **providers**: Tracks accredited organizations delivering CME. Columns include `provider_id` (UUID PK), legal name, DBA, `accreditation_number`, `contact_email`, `contact_phone`, `website_url`, address fields, `preferred_partner`, `created_at`, `updated_at`.
- **activities**: Core catalog items with `activity_id` (UUID PK), `provider_id` FK, canonical title, slug, description, modality enum (`online`, `live`, `hybrid`), format enum (conference, enduring, journal, webinar, workshop, PIP, etc.), release/expiration dates, `max_claimable_credits`, accreditation statement, `status`, `last_verified_at`, `data_confidence`, `revision_of_activity_id`.
- **activity_topics**: Many-to-many bridge with `topic_id`, `activity_id`, taxonomy term, `source` (provider vs internal).
- **activity_credit_types**: Credit breakdown per activity with `credit_row_id`, `activity_id`, `credit_type` enum (AMA PRA Cat 1, ABPN SA-CME, PIP, patient safety, ethics, etc.), `credit_quantity`, `credit_expiration`.
- **activity_sessions**: Schedules for live/hybrid events (`session_id`, `activity_id`, start/end datetimes, registration deadline, `time_zone`, venue info, recurrence pattern, capacity, waitlist status, cancellation policy URL).
- **activity_delivery_links**: Online access metadata (`delivery_id`, `activity_id`, access URL, platform name, instructions, availability window, `self_paced`).
- **activity_pricing_tiers**: Detailed pricing (`pricing_id`, `activity_id`, tier name, `price_amount`, currency, price type enum, eligibility notes, discount window, taxes/fees, refund policy, `last_checked_at`).
- **activity_eligibility_requirements**: Participation constraints (`eligibility_id`, `activity_id`, requirement type enum, value, notes, `verified_at`).
- **activity_commitment**: Time expectations (`commitment_id`, `activity_id`, `seat_time_hours`, `total_time_hours`, completion window, cohort size, pacing enum).
- **activity_documents**: Related artifacts (`document_id`, `activity_id`, document type, file URL, checksum, `ingest_source_id`).
- **requirement_mappings**: Links activities to regulatory rules (`map_id`, `activity_id`, requirement code such as `abpn.sa_cme`, coverage min/max, provenance, confidence).
- **ingest_sources**: Provenance for scraped data (`ingest_source_id`, source URL, crawl job ID, scraped timestamp, checksum, HTTP status, robots mode, parser version, manual review status, reviewer ID).
- **quality_signals**: Post-ingest quality metrics (`quality_id`, `activity_id`, metric type, metric value, sample size, `collected_at`).
- **provider_relationships** (optional): Partnership metadata for business logic.

### Relationships
- One `provider` has many `activities`.
- Each `activity` has many sessions, pricing tiers, eligibility rules, credits, delivery links, commitments, documents, requirement mappings, quality signals.
- `activity_topics` enables many-to-many between activities and shared taxonomy terms.
- `ingest_sources` relate to both raw documents and activity records for provenance tracking.

## Crawl & Enrichment Strategy
- **In-house pipeline**: Enables custom parsers, schema evolution, detailed QA, compliance handling, but requires heavier upfront engineering for scheduling, dedupe, monitoring, and legal safeguards.
- **Managed actors (e.g., Apify)**: Speed up coverage with hosted crawling, proxy rotation, scheduling UI. Trade-offs include cost, vendor lock-in, and less flexibility for rapid extractor changes. Still need internal normalization/enrichment stages.
- **Hybrid approach**: Own the normalization pipeline while selectively using managed crawlers for complex domains. Maintain a consistent enrichment/checksum workflow irrespective of source.

## Retrieval Layering
- **Primary catalog API**: Structured filters via SQL/ORM (modality, date range, credits, price) plus semantic embeddings over descriptions/objectives for vector search. Store embeddings per activity version.
- **Planner integration**: Filter candidates using structured queries, re-rank with semantic similarity, and feed top results into the scoring engine that balances requirement coverage, cost, preferences.
- **Chat grounding (RAG)**: Build context packets combining activity snippets, requirement mappings, and provenance data. Responses cite verification timestamps and unstructured notes.
- **Web search fallback**: Keep Perplexity/Google/OpenAI path behind a flag. Trigger when catalog answers low-confidence or user requests unseen activities. Any new leads get queued for ingestion so the catalog stays current.

## Prototype Crawl Snapshot
- `app/crawler/prototype_insight.py` parses the sample provider catalog in `app/crawler/samples/insight_psych.html`, producing schema-ready payloads (activities, pricing tiers, credit rows, eligibility, commitments).
- Normalization functions cover date expansion, modality detection, pricing tier splitting (including member vs general pricing), time-on-task parsing, and ABPN requirement mapping heuristics.
- Field gap detection currently flags missing `eligibility`, `location`, or `seat_time` coverage to focus human QA/LM enrichment where the HTML lacks explicit metadata.
- Output structure aligns with the DBML model so future ETL jobs can write directly into relational tables + downstream vector embeddings.
- `--sync-db` option writes the normalized payload into the relational catalog tables, triggers embedding generation, and mirrors the results into the legacy `activity` table via `app/catalog/bridge.py` so the existing planner can consume catalog-backed content immediately.

## Implementation Artifacts
- `design/catalog_schema.dbml` → materialized in `app/catalog/models.py`; tables auto-create on boot via `create_db_and_tables()`.
- FastAPI router (`app/catalog/router.py`) exposes `/catalog/activities`, detail lookups, and search endpoints backed by `app/catalog/service.py`.
- Embedding pipeline in `app/catalog/embeddings.py` prefers OpenAI embeddings when an API key is present, otherwise falls back to deterministic hash vectors.
- Planner sync (`app/planner.py`) now invokes `sync_catalog_to_activity_table` each plan build so catalog updates flow into existing recommendation logic.
