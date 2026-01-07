"""Knowledge ingestion script for populating ChromaDB."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.document_processor import DocumentProcessor
from rag.vector_store import VectorStore


def ingest_knowledge_base():
    """Process and ingest all knowledge base documents into ChromaDB."""
    print("=" * 60)
    print("EdTech Knowledge Base Ingestion")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing document processor...")
    processor = DocumentProcessor()
    
    print("\n2. Initializing vector store...")
    vector_store = VectorStore()
    
    # Check current state
    stats = vector_store.get_collection_stats()
    print(f"\n   Current collection: {stats['document_count']} documents")
    
    # Clear existing data
    if stats['document_count'] > 0:
        print("\n3. Clearing existing collection...")
        vector_store.clear_collection()
        print("   Collection cleared.")
    
    # Process documents
    print("\n4. Processing knowledge base documents...")
    chunks = processor.process_all_documents()
    
    if not chunks:
        print("\n   ERROR: No documents found in knowledge base!")
        print("   Make sure the knowledge_base/ directory contains .md files.")
        return False
    
    # Add to vector store
    print("\n5. Adding documents to vector store...")
    vector_store.add_documents(chunks)
    
    # Verify
    final_stats = vector_store.get_collection_stats()
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Documents ingested: {final_stats['document_count']}")
    print(f"Persist directory: {final_stats['persist_directory']}")
    
    # Test query
    print("\n" + "-" * 60)
    print("Testing retrieval...")
    test_query = "How is progress tracked?"
    results = vector_store.query(test_query, k=2)
    
    if results:
        print(f"✓ Query '{test_query}' returned {len(results)} results")
        print(f"  Top result topic: {results[0]['metadata'].get('topic', 'N/A')}")
    else:
        print("✗ Test query returned no results")
    
    return True


if __name__ == "__main__":
    success = ingest_knowledge_base()
    sys.exit(0 if success else 1)
