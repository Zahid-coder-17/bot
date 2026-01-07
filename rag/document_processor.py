"""Document processor for chunking knowledge base documents."""

import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Process and chunk documents for the RAG pipeline."""
    
    def __init__(self, knowledge_base_dir: str = "./knowledge_base"):
        """Initialize the document processor.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base documents.
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _get_topic_from_filename(self, filename: str) -> str:
        """Extract topic category from filename.
        
        Args:
            filename: Name of the document file.
            
        Returns:
            Topic category string.
        """
        topic_mapping = {
            "course_structure": "Course Structure",
            "assessment_policy": "Assessment Policy",
            "progress_tracking": "Progress Tracking",
            "certification": "Certification Process",
            "platform_faq": "Platform Navigation"
        }
        
        base_name = Path(filename).stem
        return topic_mapping.get(base_name, "General")
    
    def load_document(self, file_path: Path) -> str:
        """Load a single document.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            Document content as string.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def chunk_document(self, content: str, source: str, topic: str) -> List[Dict[str, Any]]:
        """Split document into chunks with metadata.
        
        Args:
            content: Document content.
            source: Source document name.
            topic: Topic category.
            
        Returns:
            List of chunk dictionaries with content and metadata.
        """
        chunks = self.text_splitter.split_text(content)
        
        return [
            {
                "id": f"{source}_{i}",
                "content": chunk,
                "metadata": {
                    "source": source,
                    "topic": topic,
                    "chunk_index": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents in the knowledge base directory.
        
        Returns:
            List of all chunk dictionaries.
        """
        all_chunks = []
        
        if not self.knowledge_base_dir.exists():
            print(f"Knowledge base directory not found: {self.knowledge_base_dir}")
            return all_chunks
        
        # Process all markdown files
        for file_path in self.knowledge_base_dir.glob("*.md"):
            print(f"Processing: {file_path.name}")
            
            content = self.load_document(file_path)
            topic = self._get_topic_from_filename(file_path.name)
            chunks = self.chunk_document(content, file_path.name, topic)
            
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    chunks = processor.process_all_documents()
    
    if chunks:
        print("\nSample chunk:")
        print(f"ID: {chunks[0]['id']}")
        print(f"Topic: {chunks[0]['metadata']['topic']}")
        print(f"Content preview: {chunks[0]['content'][:200]}...")
