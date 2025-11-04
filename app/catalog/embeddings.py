from __future__ import annotations

import hashlib
import struct
from typing import Iterable, List, Optional

from sqlmodel import Session, select

from .models import CatalogActivityEmbedding
from ..env import get_secret


def _hash_fallback(text: str, dim: int = 64) -> List[float]:
    """Generate a deterministic pseudo-embedding using SHA256 chunks."""
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    floats: List[float] = []
    while len(floats) < dim:
        digest = hashlib.sha256(digest + text.encode("utf-8", errors="ignore")).digest()
        floats.extend(
            struct.unpack("!8f", digest[:32])
        )  # 8 floats per 32 bytes big-endian
    return floats[:dim]


def generate_embedding(text: str, model: Optional[str] = None) -> List[float]:
    model_name = model or "text-embedding-3-small"
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return _hash_fallback(text)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=model_name, input=text)
        return list(resp.data[0].embedding)
    except Exception:
        return _hash_fallback(text)


def upsert_activity_embedding(
    session: Session,
    activity_id: str,
    content_chunks: Iterable[str],
    *,
    model: str = "text-embedding-3-small",
) -> CatalogActivityEmbedding:
    content = "\n".join(chunk for chunk in content_chunks if chunk)
    if not content.strip():
        content = "empty activity metadata"
    embedding = generate_embedding(content, model=model)
    row = session.exec(
        select(CatalogActivityEmbedding)
        .where(
            CatalogActivityEmbedding.activity_id == activity_id,
            CatalogActivityEmbedding.model == model,
        )
        .limit(1)
    ).first()
    if row:
        row.embedding = embedding
        row.updated_at = row.updated_at or row.created_at
        session.add(row)
    else:
        row = CatalogActivityEmbedding(
            activity_id=activity_id,
            model=model,
            embedding=embedding,
        )
        session.add(row)
    return row
