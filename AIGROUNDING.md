# AI Grounding Snapshot

_Last updated: 2025-10-03_

## Mission
Deliver an AI-first CME concierge for ABPN psychiatrists. The assistant must interpret natural-language intent, reason about the 3-year / 90-credit obligation (including patient safety, SA-CME, and PIP sub-requirements), and orchestrate discovery so the surface plan always mirrors the latest goals.

## Current System Prompt (high-level)
- Presents the assistant as a dedicated CME concierge for a psychiatrist.
- Requires reading the state snapshot (profile, remaining vs target credits, plan summary, requirement gaps, recent claims) before every reply.
- Emphasises treating “I’d rather…” statements as future preferences, not logged credits.
- Encourages concrete numerical reasoning (e.g., two 45-credit conferences cover the cycle) and proactive guidance when discovery yields no matches.
- Allows control lines after the prose response:
  - `PATCH: {...}` -> update profile fields (budget_usd, days_off, allow_live, city, specialty, target_credits, professional_stage, residency_completion_year, memberships).
  - `POLICY: {...}` -> adjust planner heuristics (diversity weighting, avoid_terms, prefer_topics, etc.).
  - `ACTION: discover` -> trigger ingestion/plan refresh when new activities are needed.
- Forbids wrapping control lines in code fences or echoing their contents in prose.

## Conversational Guardrails
1. **Future vs Completion** – Only log credits when the user states a completion verb (“earned”, “logged”, “completed”). Preference phrasing no longer creates claims.
2. **Snapshot Reasoning Order** – Remaining credits → requirement summary/gaps → plan summary/top items → profile preferences.
3. **Tone & Tempo** – Warm, professional, concise (2–4 sentences + purposeful bullets). Optional clarifying question only when necessary. Responses include a 1.2 s think-time delay to feel human.
4. **Empty Results** – When discovery finds nothing, explain why and suggest adjustments (budget, modality, memberships) instead of silently failing.
5. **Eligibility Messaging** – UI shows “⚠️ Check eligibility” for uncertain cases; we avoid negative language while still flagging restrictions.

## Architecture Highlights
- **Backend**: FastAPI + SQLModel (`app/main.py`, `app/services/plan.py`) with persistent plan runs and requirement validation.
- **Planner**: `app/planner.py` ranks activities using requirement gaps, pricing tiers, modality filters, and policies. Balanced mode now pins user-committed activities before filling gaps, while a cheapest mode can still be requested ad hoc without disturbing those commitments.
- **Discovery Engine**: `app/ingest.py` runs Google Programmable Search plus OpenAI extraction; deep-fetches pages when snippets lack pricing/eligibility.
- **Parser**: `app/parser.py` distinguishes preference language from completion statements to prevent phantom credits.
- **Frontend**: Jinja + HTMX (`app/templates/`); the plan card listens for a `plan-refresh` event so discoveries surface without manual reload. HTMX buttons now toggle “Keep” / “Remove from plan” to manage commitments inline.
- **Delay & Messaging**: Chat responses end with a compact confirmation bubble such as “Plan updated (N items, $C total, R remaining)” whenever changes affect the plan. Removal flows add explicit prompts and internal markers so the user can approve or decline the proposed substitute.
- **Validation**: `app/requirements.py` exposes `validate_full_cc`—a comprehensive CC checklist that scores CME totals, SA-CME, PIP status, and patient-safety activity against `abpn_psychiatry_requirements.json`, returning both pillar-level detail and consolidated gaps.

## Environment & Data
- SQLite database `cme.sqlite` with migrations in `app/db.py`.
- Requirement data loaded via `app/requirements.py` from `app/config/` knowledge bases.
- Environment variables: `OPENAI_API_KEY`, `OPENAI_ASSISTANT_MODEL`, optional `ENABLE_GPT5`, Google CSE keys, ingest tuning knobs.

## Session History
- **2025-10-03**: Hardened policy management and planner resilience. `/policy/clear` now copies rows before deletion and returns JSON (with optional GET redirect), the UI gains an HTMX “Clear policy” button, chat auto-creates a default user when missing, and the planner keeps ABPN-required activities even when credit targets hit zero.
- **2025-09-27**: Replaced system prompt with concierge-focused guidance, added ACTION parsing, formatted plan updates, enforced chat think-time, refreshed documentation. Parser now ignores future-intent credit phrasing; plan fragment refreshes dynamically via HTMX event. README updated to describe the concierge workflow.
- **2025-09-28**: Balanced planning became the default UI. Users can commit individual activities, toggle them via HTMX buttons, and request a cheapest-only view on demand. Chat gains “keep this plan” and “remove X” intents that update commitments and surface replacements via follow-up confirmation bubbles.
- **2025-10-03**: Introduced fuzzy `find_activity_by_title` matching, substitute discovery via `propose_substitute`, and a robust chat/HTMX removal loop that uncommits, confirms, and optionally commits a replacement. Expanded CC validation with `validate_full_cc`, producing CME/SA-CME/PIP/patient-safety checklists and gap summaries in the UI. README and docs updated to reflect balanced default plans, commitment workflow, and cheapest-on-request guidance.
- **2025-09-27**: Made Discover button fire global refresh event; plan fragment pulls automatically post-ingest. Updated UI messaging and eligibility badges.
- **2025-09-27**: Introduced persistent plan runs, knowledge-base registry, and safer logging/parser fixes to stabilize chat-driven planning.
- **2025-09-26**: Planner and compliance summary leverage enriched ABPN requirement data; ingest persists requirement tags.
- **2025-09-25**: Added pricing tiers, membership/stage preferences, hybrid-aware planning; improved ingest counters and debug feedback.
- **2025-09-20**: Initial version with core planning, chat loop, profile/preferences, activity ingest, claim parsing, and requirements validation.
- **2025-10-10**: Hardened environment handling by loading Codespaces secrets (OpenAI, Google CSE, ingest tunables) through a shared helper so the app boots with cloud-managed keys. Updated all OpenAI/Google integrations (chat, parser fallback, ingest pipeline, planner checks) to use the decoded secrets and supply GPT-5 reasoning parameters. Parser now recognizes negations or correction requests around logged credits, while chat can delete matching `Claim` rows, recalc remaining credits, and refresh plans automatically with consistent acknowledgment messaging.
