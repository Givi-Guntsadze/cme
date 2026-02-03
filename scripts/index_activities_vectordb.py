"""
Script to index all existing SQLite activities into the vector database.
Run this after ingesting activities to enable AI chat search.

Run with: python scripts/index_activities_vectordb.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlmodel import Session, select, create_engine
from app.models import Activity
from app.services import vectordb


def main():
    print("Indexing all activities into vector database...")
    
    engine = create_engine("sqlite:///./cme.sqlite")
    
    with Session(engine) as session:
        activities = session.exec(select(Activity)).all()
        print(f"Found {len(activities)} activities in SQLite")
        
        if not activities:
            print("No activities to index.")
            return
        
        # Convert Activity models to dicts for vectordb
        activity_dicts = []
        for act in activities:
            activity_dicts.append({
                "id": act.id,
                "title": act.title or "",
                "description": act.summary or "",
                "provider": act.provider or "",
                "credits": act.credits or 0,
                "cost": act.cost_usd or 0,
                "source": act.source or "web",
                "tags": [act.source or "web", act.modality or "online"],
            })
        
        # Add to vector store
        indexed = vectordb.add_activities(activity_dicts)
        print(f"Indexed {indexed} activities into vector database")
        
        # Verify
        count = vectordb.count()
        print(f"Vector database now contains {count} activities")
        
        # Test search
        print("\nTesting search for 'opioid'...")
        results = vectordb.search("opioid", n_results=5)
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  - {r['title'][:60]}... (score: {r['score']})")


if __name__ == "__main__":
    main()
