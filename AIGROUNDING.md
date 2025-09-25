# AI Grounding Snapshot

_Last updated: 2025-09-25_

## Project Context
- Proof-of-concept assistant for ABPN-certified psychiatrists to plan, discover, and track CME/MOC credits.
- Single-user FastAPI web app with SQLite persistence and Jinja/HTMX UI.
- Chat-first workflow that keeps planning logic server-side but surfaces plans, compliance, and activity log in the browser.

## Core Capabilities
- **Profile & Preferences**: Collects target credits, budget, days off, city, live-event preferences, affiliations, memberships, and training level during setup or via chat-driven PATCH directives.
- **Activity Catalog**: Starts with seeded activities and can ingest new ones via Google Programmable Search; falls back to OpenAI-powered page parsing when search results are sparse.
- **Eligibility Awareness**: Normalizes institution, group, and membership constraints on activities, tracking whether each item is publicly available or needs profile data to confirm eligibility.
- **Pricing Intelligence**: Stores tiered pricing (member, early-career, late/on-site) with registration deadlines and hybrid attendance, selecting the best-fit cost based on the user's memberships and career stage while flagging missing data or expiring rates.
- **Plan Generation**: Offers `cheapest` and `variety` modes; both respect remaining credits, budget heuristics, modality preferences, completed activity history, and optional user policy payloads (e.g., avoid topics, enforce diversity caps).
- **Assistant Chat Loop**: Stores structured user/assistant messages, applies PATCH and POLICY directives, explains plan rationale, and turns natural language updates into claims or adjustments.
- **Progress Tracking**: Parses user messages into Claim records, marks specific catalog items complete, recalculates remaining credits, and summarizes compliance against ABPN psychiatry requirements.
- **Requirements Validation**: Loads versioned rules (cycle years, annual minimums, safety/ethics thresholds) and renders pass/warn/fail checks plus source links.
- **Operations & Debug**: Provides ingest debug hook, requirements sync endpoint, policy clear/reset flows, and simple health check.

## Key Technologies & Integrations
- **Backend**: FastAPI + SQLModel on SQLite with lightweight migration logic.
- **Frontend**: Jinja templates, HTMX partials, and basic styling in `static/style.css`.
- **External Services**: OpenAI (assistant responses, fallback parsing, optional message parsing) and Google Custom Search (primary discovery pipeline).

## Known Constraints / Next Focus Areas
- Variety planner still allows certain subscriptions (e.g., AJP) to dominate unless tuned; additional diversity heuristics may be needed.
- POLICY entries accumulate until cleared; consider automatic expirations beyond current 24h default or visible management UI.
- Pricing ingestion now enforces explicit pricing/eligibility pulls; still relies on external pages exposing tiers before deep-fetch budget is exhausted.
- Eligibility checks remain coarse when profile data is missing; future work should prompt for affiliations/memberships to avoid "uncertain" status.
- Requirement validation currently only tracks annual minimum for the current year and combines safety/ethics into a single bucket.

## File Guide
- `app/main.py`: FastAPI app, HTMX endpoints, chat loop, ingest orchestration, and requirements view logic.
- `app/planner.py`: Plan assembly heuristics, eligibility helpers, and policy-aware scoring.
- `app/ingest.py`: Google CSE search, page fetching, AI extraction, and database insertion with eligibility parsing.
- `app/parser.py`: Regex + optional OpenAI parsing of user chat messages into credits/topics/dates.
- `app/requirements.py`: Loads requirement config, validates claims, and snapshots rules.
- `app/models.py`: SQLModel definitions for users, activities, claims, assistant messages, requirements snapshots, policies, and completion log.
- `static/style.css` & `app/templates/`: UI layout, chat window, plan presentation, and requirements dashboard.

## Data & Configuration Notes
- Requirements config lives at `app/config/abpn_psychiatry_requirements.json` and drives the compliance summary.
- SQLite database `cme.sqlite` is created at runtime; `app/db.py` handles migrations and purging stale seed data.
- Activity rows persist tiered `pricing_options` JSON and a `hybrid_available` flag to contextualize costs in the planner.
- Environment variables: `OPENAI_API_KEY`, `OPENAI_ASSISTANT_MODEL`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` (already configured in the deployment shell).

## Session History
- Initial grounding created; update this log with key changes or milestones at the end of each session.
- Added pricing tiers, membership/stage preferences, and hybrid-aware planning/visualization (2025-09-25).
- Fixed ingest counters/UI feedback, mapped membership aliases for pricing, and restored Discover button status updates (2025-09-25).
- Hardened ingest to require real pricing, force deep-fetch when snippets lack price/eligibility, and removed generic "check eligibility" badges for public items (2025-09-25).
- Enabled dynamic plan refresh after discovery and exposed curated source labels (2025-09-25).
- Initial version with core planning, chat loop, profile/preferences, activity ingest, claim parsing, and requirements validation (2025-09-20).
- Updated the abpn_psychiatry_requirements.json to be more detailed (2025-09-25).
