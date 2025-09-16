"""
Simplified Chroma initialization for Streamlit Cloud.
This module provides the simplest possible Chroma setup to avoid migration issues.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# SQLite3 fix for Streamlit Cloud
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
except ImportError:
    pass

try:
    # Preferred modern wrapper compatible with chromadb 1.x new clients
    from langchain_chroma import Chroma
    CHROMA_FROM_NEW_PACKAGE = True
except ImportError:
    # Fallback to community package if wrapper unavailable
    from langchain_community.vectorstores import Chroma
    CHROMA_FROM_NEW_PACKAGE = False

logger = logging.getLogger(__name__)

def get_simple_chroma(persist_directory: str, embedding_function, collection_name: str = "langchain", mode: str = "current"):
    """
    Get a Chroma vector store using the simplest possible approach.
    This avoids all migration issues by using direct initialization.

    Args:
        persist_directory: Directory for persistence
        embedding_function: Embedding function to use
        collection_name: Name of the collection
        mode: "current" for readonly persistent DB, "new" for writable temp DB

    Returns:
        Chroma: Langchain Chroma wrapper
    """
    persist_path = Path(persist_directory)

    # Check if we have a pre-built vector database (from GitHub)
    is_prebuilt = False
    if persist_path.exists():
        # Check for our specific pre-built database markers
        creation_summary = persist_path / "creation_summary.json"
        chroma_sqlite = persist_path / "chroma.sqlite3"

        if creation_summary.exists() and chroma_sqlite.exists():
            is_prebuilt = True
            logger.info(f"Found pre-built vector database at {persist_directory}")

    # For pre-built databases, never clear them
    if is_prebuilt:
        try:
            # Initialize with existing data
            vectordb = Chroma(
                persist_directory=str(persist_path),
                embedding_function=embedding_function,
                collection_name=collection_name
            )

            # Test if it works
            try:
                count = vectordb._collection.count()
                logger.info(f"Pre-built Chroma database loaded successfully with {count} documents")
                return vectordb
            except Exception as e:
                logger.warning(f"Pre-built database test failed: {e}, but continuing anyway")
                # Even if count fails, return the vectordb as it might still work for queries
                return vectordb

        except Exception as e:
            logger.error(f"Failed to load pre-built database: {e}")
            # Don't fall back to clearing - this is our production data
            raise RuntimeError(f"Cannot load pre-built vector database: {e}")

    # For non-prebuilt databases (user uploads), handle normally
    # For Streamlit Cloud, clear old format databases
    if os.environ.get("STREAMLIT_CLOUD") == "true" and not is_prebuilt:
        # Check if there's an old-format database
        old_db_files = [
            persist_path / "chroma.sqlite3",
            persist_path / "chroma-collections.parquet",
            persist_path / "chroma-embeddings.parquet"
        ]

        # If any old format files exist, clear the directory
        if any(f.exists() for f in old_db_files):
            logger.info(f"Clearing old format database at {persist_directory}")
            if persist_path.exists():
                shutil.rmtree(persist_path)

    # Ensure the directory exists
    persist_path.mkdir(parents=True, exist_ok=True)

    try:
        # Try the simplest initialization first
        vectordb = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embedding_function,
            collection_name=collection_name
        )

        # Test if it works by trying to get collection info
        try:
            _ = vectordb._collection.count()
            logger.info(f"Chroma initialized successfully at {persist_directory}")
            return vectordb
        except Exception as e:
            # If there's an error, it might be old format data
            logger.warning(f"Chroma test failed: {e}")

            # On Streamlit Cloud, just clear and recreate (only for non-prebuilt)
            if os.environ.get("STREAMLIT_CLOUD") == "true" and not is_prebuilt:
                logger.info("Clearing and recreating database on Streamlit Cloud")
                if persist_path.exists():
                    shutil.rmtree(persist_path)
                persist_path.mkdir(parents=True, exist_ok=True)

                # Try again with fresh directory
                vectordb = Chroma(
                    persist_directory=str(persist_path),
                    embedding_function=embedding_function,
                    collection_name=collection_name
                )
                logger.info("Fresh Chroma database created")
                return vectordb
            else:
                # For local or prebuilt, raise the error
                raise

    except Exception as e:
        logger.error(f"Failed to initialize Chroma: {e}")

        # Last resort - in-memory database (only for non-prebuilt)
        if not is_prebuilt:
            logger.warning("Falling back to in-memory Chroma database")
            return Chroma(
                embedding_function=embedding_function,
                collection_name=collection_name
            )
        else:
            # For prebuilt, we must not fall back to in-memory
            raise RuntimeError(f"Cannot initialize vector database: {e}")