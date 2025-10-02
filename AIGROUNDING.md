# AI Grounding Snapshot

_Last updated: 2025-09-27_

## Project Context
- Proof-of-concept assistant for ABPN-certified psychiatrists to plan, discover, and track CME/MOC credits.
- Single-user FastAPI web app with SQLite persistence and Jinja/HTMX UI.
- Chat-first workflow that keeps planning logic server-side but surfaces plans, compliance, and activity log in the browser.

## Core Capabilities
- **Profile & Preferences**: Collects target credits, budget, days off, city, live-event preferences, affiliations, memberships, and training level during setup or via chat-driven PATCH directives.
- **Activity Catalog**: Starts with seeded activities and can ingest new ones via Google Programmable Search; falls back to OpenAI-powered page parsing when search results are sparse.
- **Eligibility Awareness**: Normalizes institution, group, and membership constraints on activities, tracking whether each item is publicly available or needs profile data to confirm eligibility.
- **Pricing Intelligence**: Stores tiered pricing (member, early-career, late/on-site) with registration deadlines and hybrid attendance, selecting the best-fit cost based on the user's memberships and career stage while flagging missing data or expiring rates.
- **Plan Generation**: Uses persistent plan runs (`PlanRun`/`PlanItem`) so the assistant can explain and reuse plans across messages; delivers a single balanced plan that honors budget, modality, profile, and policy constraints.
- **Requirement-Aware Planning**: Loads ABPN requirements through the knowledge-base registry and prioritizes gaps (patient safety, SA-CME, PIP) when scoring activities; highlights requirement tags and focus banners in the UI.
- **Assistant Chat Loop**: Stores structured user/assistant messages, applies PATCH and POLICY directives, explains plan rationale, and turns natural language updates into claims or adjustments.
- **Discovery & Ingestion**: Onboards new users by triggering Google CSE + OpenAI extraction in the background when the catalog is empty; parses pricing tiers, eligibility, and requirement cues per activity before inserting.
- **Progress Tracking**: Parses user messages into Claim records, marks specific catalog items complete, recalculates remaining credits, and summarizes compliance against ABPN psychiatry requirements using the same rule source as the planner.
- **Requirements Validation**: Loads versioned rules via the knowledge base, rendering pass/warn/fail checks plus source links; auto-tagged activities map to requirement badges in the plan.
- **Operations & Debug**: Provides ingest debug hook, requirements sync endpoint, policy clear/reset flows, and simple health check.

## Key Technologies & Integrations
- **Backend**: FastAPI + SQLModel on SQLite, now with persistent plan run tables and knowledge-base registry; auto-ingestion uses asyncio-aware helpers.
- **Frontend**: Jinja templates, HTMX partials, and basic styling in `static/style.css`.
- **External Services**: OpenAI (assistant responses, fallback parsing, optional message parsing) and Google Custom Search (primary discovery pipeline).

## Known Constraints / Next Focus Areas
- Auto-ingestion currently focuses on ABPN psychiatry; expand the knowledge base and ingest query builder for other specialties (family medicine, surgery) before launching more broadly.
- Policy entries still expire after 24h; consider user-visible management or longer-lived preferences.
- Pricing ingestion depends on page structure; additional heuristics or provider-specific adapters could tighten member rate detection.
- Eligibility checks remain coarse when profile data is missing; prompt for affiliations/memberships to reduce "uncertain" statuses.
- Extend requirement tagging beyond psychiatry once new specialty knowledge bases are added.

## File Guide
- `app/main.py`: FastAPI app, HTMX endpoints, chat loop, ingest orchestration, and requirements view logic; now delegates plan state to `PlanManager` and auto-triggers discovery for empty catalogs.
- `app/planner.py`: Plan assembly heuristics, eligibility helpers, requirement-aware scoring, and policy integration; exposes `requirements_gap_summary` for other services.
- `app/ingest.py`: Google CSE search, deep fetch, OpenAI extraction, eligibility parsing, and requirement tagging prior to insertion.
- `app/parser.py`: Regex + optional OpenAI parsing of user chat messages into credits/topics/dates with guardrails against misreading years.
- `app/requirements.py`: Loads requirement payloads through the knowledge-base registry, normalizes rules, validates claims, and exposes requirement tagging helpers.
- `app/services/plan.py`: Persistent plan run manager handling caching, policy application, auto-ingestion, and serialization for UI/chat consumers.
- `app/knowledge/`: Knowledge-base registry plus ABPN psychiatry loader used by requirements/planner.
- `app/models.py`: SQLModel definitions for users, activities, claims, assistant messages, requirements snapshots, policies, and completion log.
- `static/style.css` & `app/templates/`: UI layout, chat window, plan presentation, and requirements dashboard.

## Data & Configuration Notes
- Requirements configs live under `app/config/` and are exposed via knowledge-base classes (e.g., ABPN psychiatry) for reuse across services.
- SQLite database `cme.sqlite` is created at runtime; `app/db.py` now migrates plan run tables and JSON columns required by the plan cache.
- Activity rows persist pricing tiers, eligibility attributes, and requirement tags that inform planner/validation logic.
- Environment variables: `OPENAI_API_KEY`, `OPENAI_ASSISTANT_MODEL`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` (already configured in the deployment shell).
- Plan cache repopulates automatically after asynchronous discovery; welcome assistant message is inserted on first login so the chat column always has context.

## Session History
- Initial grounding created; update this log with key changes or milestones at the end of each session.
- Added pricing tiers, membership/stage preferences, and hybrid-aware planning/visualization (2025-09-25).
- Fixed ingest counters/UI feedback, mapped membership aliases for pricing, and restored Discover button status updates (2025-09-25).
- Hardened ingest to require real pricing, force deep-fetch when snippets lack price/eligibility, and removed generic "check eligibility" badges for public items (2025-09-25).
- Enabled dynamic plan refresh after discovery and exposed curated source labels (2025-09-25).
- Initial version with core planning, chat loop, profile/preferences, activity ingest, claim parsing, and requirements validation (2025-09-20).
- Updated the abpn_psychiatry_requirements.json to be more detailed (2025-09-25).
- Planner and compliance summary now leverage enriched ABPN requirement data; requirement gaps influence scoring/UI, ingest persists requirement tags, and legacy activities are backfilled (2025-09-26).
- Introduced persistent plan runs, knowledge-base registry, and safer logging/parser fixes to stabilize chat-driven planning (2025-09-27).
- Disabled default seed catalog; first-run plans now auto-ingest real activities when empty (2025-09-27).
- Simplified to a single balanced plan mode; first dashboard load now shows a welcome message while background discovery populates activities (2025-09-27).
