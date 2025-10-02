from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import KnowledgeBase
from .registry import register

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "abpn_psychiatry_requirements.json"
)


class ABPNPsychiatryKnowledgeBase(KnowledgeBase):
    board = "ABPN"
    specialty = "Psychiatry"

    def load_requirements(self) -> Dict[str, Any]:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Requirements payload must be a JSON object")
        return data


register(ABPNPsychiatryKnowledgeBase())
