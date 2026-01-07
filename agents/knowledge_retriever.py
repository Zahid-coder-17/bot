"""Knowledge retrieval agent for fetching relevant context from ChromaDB."""

from typing import List, Dict, Any, Optional

from rag.vector_store import VectorStore
from config import TOP_K_RESULTS


class KnowledgeRetriever:
    """Retrieve relevant knowledge from the vector store."""
    
    def __init__(self):
        """Initialize the knowledge retriever."""
        self.vector_store = VectorStore()
    
    def retrieve(
        self, 
        query: str, 
        k: int = TOP_K_RESULTS,
        topic_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge chunks for a query.
        
        Args:
            query: User query text.
            k: Number of results to retrieve.
            topic_filter: Optional topic to filter by.
            
        Returns:
            List of relevant knowledge chunks.
        """
        results = self.vector_store.query(
            query_text=query,
            k=k,
            filter_topic=topic_filter
        )
        
        return results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved results as context for the LLM.
        
        Args:
            results: List of retrieved knowledge chunks.
            
        Returns:
            Formatted context string.
        """
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        
        for i, result in enumerate(results, 1):
            topic = result.get("metadata", {}).get("topic", "General")
            source = result.get("metadata", {}).get("source", "Unknown")
            content = result.get("content", "")
            
            context_parts.append(
                f"[Source {i}: {source} - {topic}]\n{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_context_for_query(self, query: str) -> str:
        """Get formatted context for a query (convenience method).
        
        Args:
            query: User query text.
            
        Returns:
            Formatted context string ready for LLM prompt.
        """
        results = self.retrieve(query)
        return self.format_context(results)


if __name__ == "__main__":
    # Test the retriever
    retriever = KnowledgeRetriever()
    
    test_query = "How do I earn a certificate?"
    context = retriever.get_context_for_query(test_query)
    
    print(f"Query: {test_query}")
    print(f"\nRetrieved Context:\n{context[:500]}...")
