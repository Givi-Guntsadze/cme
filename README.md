# CME AI Concierge

## Product Vision
AI-first assistant that helps ABPN psychiatrists satisfy CME/MOC obligations without spreadsheets. The app combines a conversational copilot, dynamic planning engine, and real-time discovery so a physician can describe preferences (“I'd rather handle this with one 45-credit conference”) and immediately see a refreshed plan that honors ABPN rules.

Key capabilities:
- Understand natural-language goals and convert them into profile updates, requirement reasoning, and discovery commands.
- Always reflect the current plan in the UI; discover/refresh results stream back into the plan card without manual reloads.
- Track eligibility nuances (memberships, institutional exclusivity) while keeping the user-facing chat optimistic—uncertain items are surfaced as “check eligibility” instead of hard denials.
- Maintain persistent plans, claims, and requirement validation against the ABPN psychiatry rule set.
- Present a balanced plan by default, blending cost efficiency, requirement coverage, and topic/provider diversity while preserving any activities you’ve committed to.

See `AIGROUNDING.md` for the current grounding notes used by the assistant.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the service (default port 8000):
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port ${APP_PORT:-8000}
```

Stop prior instances when needed:
```bash
lsof -t -i TCP:${APP_PORT:-8000} | xargs -r kill
pkill -f "uvicorn .*app.main:app" || true
```

## Using the App
1. **Profile Setup** – Provide baseline preferences (target credits, budget, days off, memberships). The plan card will show an empty state until discovery runs.
2. **Conversational Planning** – Use the chat column to express goals or constraints. The assistant reasons over your 3-year obligation, remaining credits, plan summary, and requirement gaps before answering. Future-oriented statements (e.g., “I’d rather do one 45-credit conference”) are treated as preferences, not completed CME.
3. **Discovery on Demand** – Click **Discover** or let the assistant respond with `ACTION: discover` to trigger real-time ingestion. New activities appear in the plan instantly via HTMX refresh; no browser reload required.
4. **Lock What You’ll Do** – Each activity in the balanced plan has a **Keep** button. Once committed, it stays fixed; use **Remove from plan** to release it. The planner fills remaining gaps around your committed picks.
5. **Cheapest View on Request** – Ask the assistant for a “cheapest plan” (or hit `/fragment/plan?mode=cheapest`) to see cost-first recommendations; committed activities remain in place.
6. **Eligibility & Pricing** – Activities display badges for modality, membership pricing, and “⚠️ Check eligibility” when additional profile data is required. Member-only rates and hybrid options are calculated from the stored pricing snapshot.
7. **Compliance View** – The requirements panel compares logged claims against ABPN psychiatry rules, highlighting remaining totals and flagged sub-requirements (SA-CME, patient safety, PIP).

## AI Loop
- System prompt positions the assistant as a CME concierge that reasons about multi-year credit requirements and high-yield conferences.
- Snapshot passed to the model includes profile settings, plan summary, and requirement status so responses stay grounded.
- The assistant can emit control lines:
  - `PATCH: {...}` to update profile fields.
  - `POLICY: {...}` for planner heuristics.
  - `ACTION: discover` to launch ingestion and rebuild the plan.
- Preference intent (e.g., “one 45-credit conference”) is treated as planning guidance—credits are only logged when completion verbs are present.
- Chat responses are intentionally paced (1.2 s think time) before rendering, so the interaction feels like working with a human assistant.

## Development Notes
- **Backend**: FastAPI + SQLModel (`app/main.py`, `app/services/plan.py`). Plan state persists in `PlanRun` tables.
- **Planner**: `app/planner.py` scores activities using requirement gaps, pricing, and policy tweaks.
- **Discovery**: `app/ingest.py` orchestrates Google Programmable Search + OpenAI extraction with deep-fetch fallback.
- **Chat Parsing**: `app/parser.py` distinguishes between completed-credits statements and preference language.
- **Templates**: Jinja + HTMX in `app/templates/` with styling in `static/style.css`. The plan card listens for a global `plan-refresh` event to reload fragments automatically.
- **Database**: SQLite (`cme.sqlite`). See `app/db.py` for migrations.

## Testing
```bash
pytest
```
Tests cover the planner manager, parser guardrails, and requirement usage pipeline.

## Handy Git Commands
```bash
git status               # review working tree and staged changes
git add <path>           # stage updates (use a path or . for everything)
git commit -am "msg"     # stage modified tracked files and commit without opening an editor
git commit -m "msg"      # commit staged changes without the commit_editmsg prompt
```

## Environment Variables
- `OPENAI_API_KEY` (required) – assistant responses and ingest extraction
- `OPENAI_ASSISTANT_MODEL` (optional, defaults to `gpt-4o-mini`)
- `GOOGLE_API_KEY` / `GOOGLE_CSE_ID` – enable discovery pipeline
- `INGEST_MIN_RESULTS` / `INGEST_MAX_DEEPFETCH` – tune discovery volume

## Roadmap Hints
- Expand requirement knowledge bases beyond ABPN psychiatry.
- Teach the planner to balance multiple conference blocks per cycle automatically.
- Add richer telemetry on discovery results (conversion rate, eligibility blockers).
