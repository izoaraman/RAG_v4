import os
import sys

# SQLite3 fix for Streamlit Cloud (must be before any Chroma imports)
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
except ImportError:
    pass

# Only use our helper; avoid importing Chroma directly here to prevent client init at import time
from .simple_chroma import get_simple_chroma

def list_all_source_documents(persist_directory: str, embedding_function) -> list:
    """
    Retrieves a unique list of all source document file names from the vector database.
    """
    # Initialize the vector database using simplified approach
    vectordb = get_simple_chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name="langchain"
    )
    
    # Retrieve metadata for all documents in the collection
    collection_data = vectordb._collection.get(include=["metadatas"])
    
    unique_sources = set()
    metadatas = collection_data.get("metadatas", [])
    for meta in metadatas:
        if isinstance(meta, dict) and "source" in meta:
            # Normalize the file name if needed
            filename = os.path.basename(meta["source"]).replace("\\", "/")
            unique_sources.add(filename)
    
    return sorted(list(unique_sources))
