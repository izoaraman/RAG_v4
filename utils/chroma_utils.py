"""
Centralized Chroma utilities for handling database initialization.
Provides compatibility for both old and new Chroma versions.
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

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

def get_chroma_client(persist_directory: str, reset_if_needed: bool = False):
    """
    Get a Chroma client with proper error handling for migration issues.
    
    Args:
        persist_directory: Directory for persistence
        reset_if_needed: Whether to reset the database if migration issues occur
        
    Returns:
        chromadb.PersistentClient: Configured Chroma client
    """
    persist_path = Path(persist_directory)
    
    # Check if this is a fresh deployment (no existing data)
    is_fresh = not persist_path.exists() or not any(persist_path.iterdir())
    
    if is_fresh:
        logger.info(f"Creating fresh Chroma database at {persist_directory}")
        persist_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to create/load with new API
        client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Test the client by trying to list collections
        try:
            client.list_collections()
            logger.info("Chroma client initialized successfully")
        except Exception as e:
            if "migration" in str(e).lower() or "deprecated" in str(e).lower():
                logger.warning(f"Migration issue detected: {e}")
                if reset_if_needed:
                    logger.info("Resetting Chroma database due to migration issues")
                    return reset_chroma_database(persist_directory)
            raise
            
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Chroma client: {e}")
        
        # If we encounter migration issues and reset is allowed
        if reset_if_needed and ("migration" in str(e).lower() or "deprecated" in str(e).lower()):
            return reset_chroma_database(persist_directory)
        
        # For Streamlit Cloud, always try to start fresh if there are issues
        if "STREAMLIT_SHARING_MODE" in os.environ:
            logger.info("Running on Streamlit Cloud - creating fresh database")
            return reset_chroma_database(persist_directory)
            
        raise

def reset_chroma_database(persist_directory: str):
    """
    Reset the Chroma database by removing old data and creating fresh.
    
    Args:
        persist_directory: Directory to reset
        
    Returns:
        chromadb.PersistentClient: Fresh Chroma client
    """
    persist_path = Path(persist_directory)
    
    # Backup old data if it exists
    if persist_path.exists() and any(persist_path.iterdir()):
        backup_path = persist_path.parent / f"{persist_path.name}_backup"
        if backup_path.exists():
            shutil.rmtree(backup_path)
        logger.info(f"Backing up old database to {backup_path}")
        shutil.move(str(persist_path), str(backup_path))
    
    # Create fresh directory
    persist_path.mkdir(parents=True, exist_ok=True)
    
    # Create new client
    client = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
    )
    
    logger.info("Created fresh Chroma database")
    return client

def get_langchain_chroma(persist_directory: str, embedding_function, collection_name: str = "langchain", reset_if_needed: bool = False):
    """
    Get a Langchain-compatible Chroma vector store.
    
    Args:
        persist_directory: Directory for persistence
        embedding_function: Embedding function to use
        collection_name: Name of the collection
        reset_if_needed: Whether to reset if migration issues occur
        
    Returns:
        Chroma: Langchain Chroma wrapper
    """
    try:
        # First, try the simple approach (for new deployments)
        if not Path(persist_directory).exists() or not any(Path(persist_directory).iterdir()):
            logger.info("Creating new Chroma vector store")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function,
                collection_name=collection_name
            )
        
        # For existing data, use the client approach
        client = get_chroma_client(persist_directory, reset_if_needed=reset_if_needed)
        
        # Get or create the collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            # Collection doesn't exist, will be created by Langchain wrapper
            logger.info(f"Collection {collection_name} will be created")
        
        return Chroma(
            client=client,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
        
    except Exception as e:
        logger.error(f"Failed to create Langchain Chroma: {e}")
        
        # On Streamlit Cloud, always try simple initialization as fallback
        if "STREAMLIT_SHARING_MODE" in os.environ:
            logger.info("Streamlit Cloud detected - using simple initialization")
            
            # Clear the directory and start fresh
            persist_path = Path(persist_directory)
            if persist_path.exists():
                shutil.rmtree(persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function,
                collection_name=collection_name
            )
        
        raise

def is_running_on_streamlit_cloud():
    """Check if the app is running on Streamlit Cloud."""
    return "STREAMLIT_SHARING_MODE" in os.environ or "STREAMLIT_RUNTIME_ENV" in os.environ