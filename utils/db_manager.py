"""
Database connection manager for Chroma vector database.
"""

import os
import sys

# SQLite3 fix for Streamlit Cloud (must be before any Chroma imports)
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
except ImportError:
    pass

import threading
from contextlib import contextmanager
from typing import Optional
from langchain_community.vectorstores import Chroma
from .simple_chroma import get_simple_chroma

class ChromaManager:
    """
    Singleton manager for Chroma database connections.
    Ensures proper connection handling and cleanup.
    """
    _instance = None
    _lock = threading.Lock()
    _connections = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ChromaManager, cls).__new__(cls)
        return cls._instance
    
    @contextmanager
    def get_db(self, persist_directory: str, embedding_function, mode: str = "current") -> Chroma:
        """
        Get a Chroma database connection with proper cleanup.
        
        Args:
            persist_directory: Directory where Chroma stores its data
            embedding_function: Function to use for embeddings
            mode: "current" for persistent DB, "new" for in-memory on cloud
            
        Yields:
            Chroma: A Chroma database connection
        """
        thread_id = threading.get_ident()
        
        try:
            # Create new connection if needed
            # Always use persistent database for both modes to ensure data is saved
            if False:  # Disabled in-memory mode to fix empty responses
                # (keeping code for reference but disabled)
                pass
            else:
                # Normal caching for persistent databases
                if thread_id not in self._connections:
                    # Use the simplified Chroma initialization
                    self._connections[thread_id] = get_simple_chroma(
                        persist_directory=persist_directory,
                        embedding_function=embedding_function,
                        collection_name="langchain",
                        mode=mode
                    )
                
                # Yield the connection
                yield self._connections[thread_id]
            
        finally:
            # Clean up connection
            if thread_id in self._connections:
                # Community Chroma vectorstore does not expose a stable close API
                # Just drop the reference and let GC handle it
                del self._connections[thread_id]

# Global instance
db_manager = ChromaManager()