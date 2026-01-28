
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vectordb():
    print("Testing VectorDB initialization...")
    try:
        from app.services import vectordb
        print("Import successful.")
        
        print("Initializing embedding model...")
        model = vectordb._get_embedding_model()
        print(f"Model initialized: {type(model)}")
        
        print("Initializing ChromaDB collection...")
        collection = vectordb._get_collection()
        print(f"Collection initialized: {collection.name}")
        
        print("Running a test search...")
        results = vectordb.search("patient safety", n_results=1)
        print(f"Search returned {len(results)} results.")
        
    except Exception as e:
        logger.exception("VectorDB test failed!")
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    test_vectordb()
