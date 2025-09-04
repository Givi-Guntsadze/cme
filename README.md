# CME/MOC POC (ABPN Psychiatry)

## Overview
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
