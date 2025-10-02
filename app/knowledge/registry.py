from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .base import KnowledgeBase

_REGISTRY: Dict[Tuple[str, str], KnowledgeBase] = {}


def register(base: KnowledgeBase) -> None:
    key = (base.board.lower(), base.specialty.lower())
    _REGISTRY[key] = base


def get(board: str, specialty: str) -> Optional[KnowledgeBase]:
    key = (board.lower(), specialty.lower())
    return _REGISTRY.get(key)


def all_bases() -> List[KnowledgeBase]:
    return list(_REGISTRY.values())
