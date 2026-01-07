"""Embedding generation using Google Gemini API."""

from typing import List
import google.generativeai as genai

from config import GOOGLE_API_KEY, EMBEDDING_MODEL


class EmbeddingGenerator:
    """Generate embeddings using Google Gemini API."""
    
    def __init__(self):
        """Initialize the embedding generator."""
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
        self.model = EMBEDDING_MODEL
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query (optimized for retrieval).
        
        Args:
            query: Query text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings


if __name__ == "__main__":
    # Test embedding generation
    generator = EmbeddingGenerator()
    
    test_text = "How is course progress tracked on the platform?"
    embedding = generator.generate_query_embedding(test_text)
    
    if embedding:
        print(f"Generated embedding with {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
    else:
        print("Failed to generate embedding. Check API key.")
