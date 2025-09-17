"""Simple vector database implementation for Streamlit Cloud fallback."""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleDocument:
    """Simple Document class compatible with Langchain Document."""

    def __init__(self, page_content: str, metadata: Optional[Dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class SimpleVectorDB:
    """
    Simple vector database implementation that works reliably on Streamlit Cloud.
    Uses numpy for similarity search - no complex database dependencies.
    """

    def __init__(self, persist_directory: str, embedding_function):
        self.persist_directory = Path(persist_directory)
        self.embedding_function = embedding_function
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add texts with their embeddings to the database."""
        if not texts:
            return

        # Generate embeddings
        new_embeddings = self.embedding_function.embed_documents(texts)

        # Store data
        self.documents.extend(texts)
        self.embeddings.extend(new_embeddings)

        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{}] * len(texts))

        logger.info(f"Added {len(texts)} documents to simple vector database")

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Any]:
        """Search for similar documents using cosine similarity.

        Returns list of Document-like objects for Chroma compatibility.
        """
        if not self.documents:
            logger.warning("No documents in database")
            return []

        try:
            # Get query embedding
            query_embedding = self.embedding_function.embed_query(query)
            query_vec = np.array(query_embedding)

            # Calculate cosine similarities
            embeddings_matrix = np.array(self.embeddings)

            # Normalize vectors
            query_norm = query_vec / np.linalg.norm(query_vec)
            embeddings_norm = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)

            # Calculate cosine similarity
            similarities = np.dot(embeddings_norm, query_norm)

            # Get top k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            # Create Document-like results for compatibility
            results = []
            for idx in top_k_indices:
                # Create a simple Document-like object
                doc = SimpleDocument(
                    page_content=self.documents[idx],
                    metadata=self.metadatas[idx]
                )
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def save(self):
        """Save the database to disk."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadatas": self.metadatas
        }

        save_path = self.persist_directory / "simple_vectordb.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        # Also save metadata
        metadata_path = self.persist_directory / "simple_vectordb_meta.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "num_documents": len(self.documents),
                "created_with": "SimpleVectorDB"
            }, f)

        logger.info(f"Saved {len(self.documents)} documents to {self.persist_directory}")

    def load(self) -> bool:
        """Load the database from disk."""
        save_path = self.persist_directory / "simple_vectordb.pkl"

        if not save_path.exists():
            logger.warning(f"No saved database found at {save_path}")
            return False

        try:
            with open(save_path, "rb") as f:
                data = pickle.load(f)

            self.documents = data["documents"]
            self.embeddings = data["embeddings"]
            self.metadatas = data["metadatas"]

            logger.info(f"Loaded {len(self.documents)} documents from {self.persist_directory}")
            return True

        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False

    def count(self) -> int:
        """Get the number of documents in the database."""
        return len(self.documents)

    def delete_collection(self):
        """Clear the database (for compatibility)."""
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        logger.info("Collection cleared")

    def get(self) -> Dict[str, Any]:
        """Get collection info (for compatibility)."""
        return {
            "ids": list(range(len(self.documents))),
            "documents": self.documents,
            "metadatas": self.metadatas
        }

    @property
    def _collection(self):
        """Provide collection-like interface for compatibility."""
        class CollectionProxy:
            def __init__(self, parent):
                self.parent = parent

            def count(self):
                return self.parent.count()

            def get(self):
                return self.parent.get()

        return CollectionProxy(self)


def create_simple_vectordb(persist_directory: str, embedding_function, documents: List[Any]) -> SimpleVectorDB:
    """Create and populate a simple vector database with rate limiting."""
    import time

    db = SimpleVectorDB(persist_directory, embedding_function)

    # Process documents in smaller batches to avoid rate limits
    batch_size = 20  # Reduced from 100 to avoid rate limits
    retry_delay = 2  # Initial delay in seconds
    max_retries = 5

    for i in range(0, len(documents), batch_size):
        batch = documents[i:min(i+batch_size, len(documents))]

        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        # Retry logic for rate limits
        for attempt in range(max_retries):
            try:
                db.add_texts(texts, metadatas)
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                time.sleep(1)  # Add delay between batches to avoid rate limits
                break
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                else:
                    logger.error(f"Error processing batch: {e}")
                    raise

    # Save to disk
    db.save()

    return db