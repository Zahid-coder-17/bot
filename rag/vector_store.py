"""ChromaDB vector store for storing and retrieving knowledge embeddings."""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K_RESULTS
from .embeddings import EmbeddingGenerator


class VectorStore:
    """ChromaDB-based vector store for semantic search."""
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        """Initialize the vector store.
        
        Args:
            persist_dir: Directory to persist ChromaDB data.
        """
        self.persist_dir = persist_dir
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "EdTech knowledge base embeddings"}
        )
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with id, content, and metadata.
        """
        if not chunks:
            print("No chunks to add.")
            return
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            ids.append(chunk["id"])
            documents.append(chunk["content"])
            metadatas.append(chunk["metadata"])
            
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(chunk["content"])
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"Warning: Failed to generate embedding for chunk {chunk['id']}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 768)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks")
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        print(f"Added {len(chunks)} chunks to vector store.")
    
    def query(
        self, 
        query_text: str, 
        k: int = TOP_K_RESULTS,
        filter_topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store for relevant documents.
        
        Args:
            query_text: Query text.
            k: Number of results to return.
            filter_topic: Optional topic filter.
            
        Returns:
            List of relevant document chunks with metadata.
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query_text)
        
        if not query_embedding:
            print("Failed to generate query embedding.")
            return []
        
        # Build where filter if topic specified
        where_filter = None
        if filter_topic:
            where_filter = {"topic": filter_topic}
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        count = self.collection.count()
        return {
            "collection_name": COLLECTION_NAME,
            "document_count": count,
            "persist_directory": self.persist_dir
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "EdTech knowledge base embeddings"}
        )
        print("Collection cleared.")


if __name__ == "__main__":
    # Test vector store
    store = VectorStore()
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    # Test query
    results = store.query("How do I track my progress?", k=3)
    print(f"\nFound {len(results)} results")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Topic: {result['metadata'].get('topic', 'N/A')}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Content: {result['content'][:150]}...")
