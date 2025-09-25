# CME/MOC POC (ABPN Psychiatry)

## Overview
For the most current context and AI session notes, see `AIGROUNDING.md`.

Proof-of-concept tool for psychiatrists to plan and track CME/MOC requirements:
- Input: target credits, budget, days off, modality (online/live).
- Output: optimized plan from a seeded catalog of CME activities.
- Doctor messages progress (e.g., “Got 1 credit on Doximity”).
- Remaining credits and plan update dynamically.

ABPN portal remains the official record. This tool provides planning + organization + easy logging.

---

## Requirements
- Python 3.11+
- Git
- Chrome browser (for UI)

---

## Setup
```bash
# Clone/init repo
mkdir cme-poc && cd cme-poc && git init

# Virtual env
python -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### Start/Restart the app
Run one of the following from the repo root after activating the venv.

Foreground (shows logs):
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port ${APP_PORT:-8000}
```

Background (same terminal free):
```bash
nohup .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port ${APP_PORT:-8000} > /tmp/cme-uvicorn.log 2>&1 & disown
```

Stop any previous server (free port 8000):
```bash
lsof -t -i TCP:${APP_PORT:-8000} | xargs -r kill; pkill -f "uvicorn .*app.main:app" || true
```
