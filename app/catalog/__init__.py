"""
Catalog data models, services, and router for the CME activity database.
"""

from . import models  # noqa: F401  (ensures SQLModel metadata registers)

__all__ = ["models"]
