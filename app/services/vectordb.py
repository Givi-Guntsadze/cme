"""
Vector database service for RAG-based activity search.

Uses ChromaDB for local vector storage and sentence-transformers for embeddings.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy-load heavy dependencies to speed up app startup
_chroma_client = None
_collection = None
_embedding_model = None

CHROMA_PERSIST_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"
COLLECTION_NAME = "cme_activities"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_embedding_model():
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _embedding_model


def _get_collection():
    """Lazy-load ChromaDB client and collection."""
    global _chroma_client, _collection
    if _collection is None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise

        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing ChromaDB at: %s", CHROMA_PERSIST_DIR)
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text string."""
    model = _get_embedding_model()
    return model.encode(text, convert_to_numpy=True).tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts (batched for efficiency)."""
    model = _get_embedding_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


def add_activities(activities: List[dict]) -> int:
    """
    Add or update activities in the vector store.
    
    Each activity dict should have:
      - id: unique identifier (string or int)
      - title: activity title
      - description: optional full description
      - provider: optional provider name
      - credits: optional credit count
      - cost: optional cost
      - tags: optional list of tags
    
    Returns the number of activities added/updated.
    """
    if not activities:
        return 0
    
    collection = _get_collection()
    
    ids = []
    documents = []
    metadatas = []
    
    for act in activities:
        act_id = str(act.get("id", ""))
        if not act_id:
            continue
            
        title = act.get("title", "")
        description = act.get("description", "")
        provider = act.get("provider", "")
        tags = act.get("tags", [])
        
        # Create searchable document from title + description + tags
        doc_parts = [title]
        if description:
            doc_parts.append(description)
        if provider:
            doc_parts.append(f"Provider: {provider}")
        if tags:
            doc_parts.append(f"Tags: {', '.join(tags)}")
        
        document = " | ".join(doc_parts)
        
        ids.append(act_id)
        documents.append(document)
        metadatas.append({
            "title": title,
            "provider": provider or "",
            "credits": float(act.get("credits") or 0),
            "cost": float(act.get("cost") or 0),
            "tags": ",".join(tags) if tags else "",
        })
    
    if not ids:
        return 0
    
    # Generate embeddings
    embeddings = embed_texts(documents)
    
    # Upsert into collection
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    
    logger.info("Added/updated %d activities in vector store", len(ids))
    return len(ids)


def search(
    query: str,
    n_results: int = 5,
    filter_tags: Optional[List[str]] = None,
) -> List[dict]:
    """
    Search for activities matching the query.
    
    Returns a list of dicts with keys:
      - id: activity ID
      - title: activity title
      - provider: provider name
      - credits: credit count
      - cost: cost
      - tags: list of tags
      - score: similarity score (0-1, higher is better)
    """
    collection = _get_collection()
    
    # Generate query embedding
    query_embedding = embed_text(query)
    
    # Build where clause for filtering
    where = None
    if filter_tags:
        # ChromaDB uses $contains for string matching
        where = {"tags": {"$contains": filter_tags[0]}}  # Simple single-tag filter
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    
    # Transform results into a cleaner format
    output = []
    if results and results.get("ids") and results["ids"][0]:
        for i, id_ in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 1.0
            
            # Convert distance to similarity score (cosine distance -> similarity)
            score = 1 - distance
            
            output.append({
                "id": id_,
                "title": metadata.get("title", ""),
                "provider": metadata.get("provider", ""),
                "credits": metadata.get("credits", 0),
                "cost": metadata.get("cost", 0),
                "tags": metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                "score": round(score, 3),
            })
    
    return output


def count() -> int:
    """Return the number of activities in the vector store."""
    collection = _get_collection()
    return collection.count()


def clear() -> None:
    """Clear all activities from the vector store."""
    global _collection
    if _chroma_client:
        try:
            _chroma_client.delete_collection(COLLECTION_NAME)
            _collection = None
            logger.info("Cleared vector store collection: %s", COLLECTION_NAME)
        except Exception as e:
            logger.warning("Could not clear collection: %s", e)
