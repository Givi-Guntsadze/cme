from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class KnowledgeBase(ABC):
    """Abstract base for specialty-specific knowledge sources."""

    board: str
    specialty: str

    @abstractmethod
    def load_requirements(self) -> Dict[str, Any]:
        """Return canonical requirement payload for the specialty."""

    def metadata(self) -> Dict[str, Any]:  # pragma: no cover - simple default
        return {"board": self.board, "specialty": self.specialty}
