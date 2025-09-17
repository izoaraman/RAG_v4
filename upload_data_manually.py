import os
import sys

# SQLite3 fix for Streamlit Cloud (must be before any Chroma imports)
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
except ImportError:
    pass

from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig
from utils.simple_chroma import get_simple_chroma
from utils.docx_loader_fallback import SimpleDocxLoader
import hashlib
import json
import time
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredEmailLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)

# Import multimodal indexing if available
try:
    from hybrid_rag.index_multimodal import MultimodalIndexer
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("Note: Multimodal indexing not available. Install required dependencies for multimodal support.")

CONFIG = LoadConfig()

def ensure_directory_access(directory: str):
    """Ensure the directory exists and has write access."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        print(f"Fixing permission issues for: {directory}")
        try:
            os.chmod(directory, 0o777)  # Grant full permissions
        except PermissionError:
            print(f"Permission denied: Unable to modify {directory}")

def calculate_file_hash(file_path):
    """Calculate the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_existing_file_hashes(persist_directory):
    """Retrieve existing file hashes from the metadata file."""
    metadata_file = os.path.join(persist_directory, "file_hashes.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def update_file_hashes(persist_directory, file_hashes):
    """Update the metadata file with new file hashes."""
    metadata_file = os.path.join(persist_directory, "file_hashes.json")
    with open(metadata_file, 'w') as f:
        json.dump(file_hashes, f)

def load_document_by_type(file_path):
    """Load a document based on its file extension.
    
    Returns:
        list: Document pages if successful
        None: If loading failed (missing dependencies or errors)
        []: Empty list for unsupported file types
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    
    try:
        if file_ext == '.pdf':
            return PyPDFLoader(file_path).load()
        elif file_ext in ['.doc', '.docx']:
            # Use fallback loader for .docx files to avoid segmentation fault
            if file_ext == '.docx':
                try:
                    return SimpleDocxLoader(file_path).load()
                except Exception as e:
                    print(f"ERROR: Failed to load {filename} with fallback loader: {e}")
                    return None
            else:
                # .doc files still need special handling
                print(f"WARNING: .doc files not supported. Please convert {filename} to .docx format.")
                return None
        elif file_ext in ['.eml', '.msg']:
            try:
                return UnstructuredEmailLoader(file_path).load()
            except (ImportError, ModuleNotFoundError):
                print(f"WARNING: Cannot load {filename} - 'unstructured' package not installed.")
                print(f"  To enable email support, install: pip install unstructured")
                return None
        elif file_ext == '.txt':
            return TextLoader(file_path, encoding='utf-8').load()
        elif file_ext == '.csv':
            return CSVLoader(file_path, encoding='utf-8').load()
        elif file_ext in ['.html', '.htm']:
            try:
                return UnstructuredHTMLLoader(file_path).load()
            except (ImportError, ModuleNotFoundError):
                print(f"WARNING: Cannot load {filename} - 'unstructured' package not installed.")
                print(f"  To enable HTML support, install: pip install unstructured")
                return None
        else:
            print(f"Unsupported file type: {file_ext}. Skipping {filename}")
            return []
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return None  # Return None to indicate error

def upload_data_for_mode(mode="current", clear_existing=False, enable_multimodal=False) -> None:
    """
    Process documents for the specified mode.
    
    Args:
        mode: "current" for main vector DB, "new" for new document mode
        clear_existing: If True, clear existing vector DB before processing (useful for New document mode)
        enable_multimodal: If True and available, use multimodal indexing for PDFs
    """
    
    # Handle multimodal indexing if enabled
    if enable_multimodal and MULTIMODAL_AVAILABLE:
        print("Multimodal indexing enabled for PDFs")
        multimodal_indexer = MultimodalIndexer(
            persist_directory="data/vectordb/multimodal/chroma/",
            collection_name="docs_mm",
            embedding_model="text-embedding-ada-002",
            enable_ocr=True,
            enable_table_detection=True,
            enable_file_hashing=True,
            use_azure_ocr=True
        )
        
        # Process PDFs with multimodal indexing
        if mode == "new":
            # Use temp directory on Streamlit Cloud
            if os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser"):
                temp_base = os.environ.get("TMPDIR", "/tmp")
                data_directory = os.path.join(temp_base, "new_uploads")
            else:
                data_directory = os.path.join(os.path.dirname(CONFIG.data_directory), "new_uploads")
        else:
            data_directory = CONFIG.data_directory
            
        pdf_files = [f for f in os.listdir(data_directory) 
                    if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(data_directory, f))]
        
        if pdf_files:
            print(f"Processing {len(pdf_files)} PDF(s) with multimodal indexing...")
            for pdf_file in pdf_files:
                pdf_path = os.path.join(data_directory, pdf_file)
                success = multimodal_indexer.process_document(pdf_path)
                if success:
                    print(f"  ✓ Processed successfully: {pdf_file}")
                else:
                    print(f"  ✗ Failed to process: {pdf_file}")
    elif enable_multimodal and not MULTIMODAL_AVAILABLE:
        print("WARNING: Multimodal indexing requested but dependencies not installed.")
        print("Install: pip install pdfplumber tabula-py pytesseract pillow")
        print("Falling back to standard processing...")
    
    # Select the appropriate directories based on mode
    if mode == "new":
        persist_dir = CONFIG.custom_persist_directory
        # Use separate directory for new document uploads
        if os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser"):
            temp_base = os.environ.get("TMPDIR", "/tmp")
            data_directory = os.path.join(temp_base, "new_uploads")
        else:
            data_directory = os.path.join(os.path.dirname(CONFIG.data_directory), "new_uploads")
        # For new document mode, optionally clear existing to ensure isolation
        if clear_existing and os.path.exists(persist_dir):
            import shutil
            print(f"Attempting to clear existing vector database at {persist_dir}")
            try:
                shutil.rmtree(persist_dir)
                print("Successfully cleared vector database")
            except PermissionError as e:
                print(f"Warning: Could not clear due to locked files. Will overwrite.")
                # Continue anyway - the new data will overwrite
    else:
        persist_dir = CONFIG.persist_directory
        data_directory = CONFIG.data_directory
    
    ensure_directory_access(persist_dir)
    ensure_directory_access(data_directory)

    # For in-memory mode on Streamlit Cloud, skip hash file operations
    if mode == "new" and (os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser")):
        existing_hashes = {}  # Always treat as fresh upload in-memory
        print("Using in-memory mode - treating all files as new for processing")
    else:
        existing_hashes = get_existing_file_hashes(persist_dir)
    
    print(f"Checking directory: {data_directory}")
    print(f"Using persist directory: {persist_dir}")
    print(f"Mode: {mode}")
    print(f"Found {len(existing_hashes)} previously processed files")
    
    # List all supported document files in the data directory.
    supported_extensions = ('.pdf', '.doc', '.docx', '.txt', '.csv', '.eml', '.msg', '.html', '.htm')
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"Directory does not exist: {data_directory}. Creating it...")
        os.makedirs(data_directory, exist_ok=True)
        print("No documents to process in newly created directory.")
        return
    
    data_files = [f for f in os.listdir(data_directory) 
                  if os.path.isfile(os.path.join(data_directory, f)) 
                  and f.lower().endswith(supported_extensions)]
    
    # For new document mode, if no files found, return early
    if mode == "new" and not data_files:
        print("No documents in new uploads directory. Nothing to process.")
        return
    
    new_files = []
    new_hashes = {}
    
    for filename in data_files:
        full_path = os.path.join(data_directory, filename)
        # Skip empty files
        if os.path.getsize(full_path) == 0:
            print(f"Skipping empty file: {filename}")
            continue
        file_hash = calculate_file_hash(full_path)
        # Check if this exact file (by hash) has been processed before
        if filename not in existing_hashes or existing_hashes[filename] != file_hash:
            new_files.append(full_path)
            new_hashes[filename] = file_hash

    if not new_files:
        print("No new documents to process. VectorDB is up-to-date.")
        print(f"Files in directory: {data_files}")
        print(f"Files in hash db: {list(existing_hashes.keys())}")
        return

    print(f"Found {len(new_files)} new document(s) to process:")
    for f in new_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load new documents using the shared loader function
    new_docs = []
    failed_files = []  # Track files that failed to load
    for file_path in new_files:
        loaded_docs = load_document_by_type(file_path)
        if loaded_docs is None:
            # Loading failed - don't mark as processed
            failed_files.append(os.path.basename(file_path))
            filename = os.path.basename(file_path)
            # Remove from new_hashes so it won't be marked as processed
            if filename in new_hashes:
                del new_hashes[filename]
        elif loaded_docs:  # Successfully loaded with content
            new_docs.extend(loaded_docs)
        # If loaded_docs is [], it's an unsupported type but still mark as processed
    
    if failed_files:
        print(f"\nFailed to load {len(failed_files)} file(s) due to errors or missing dependencies:")
        for f in failed_files:
            print(f"  - {f}")
        print("These files will NOT be marked as processed and can be retried.")
    
    print(f"Number of new pages: {len(new_docs)}")
    
    # Instantiate PrepareVectorDB for processing
    pvdb = PrepareVectorDB(
        data_directory=data_directory,  # Use the appropriate data directory
        persist_directory=persist_dir,
        embedding_model_engine=CONFIG.embedding_model_engine,
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap
    )
    
    # Chunk the new documents.
    # Calling the private method via name mangling.
    chunked_new_docs = pvdb._PrepareVectorDB__chunk_documents(new_docs)
    print(f"Number of new chunks: {len(chunked_new_docs)}")
    
    # Skip if no chunks were created (e.g., scanned PDFs with no text)
    if len(chunked_new_docs) == 0:
        print("WARNING: No text could be extracted from the document(s).")
        print("This may be due to:")
        print("  1. Scanned PDFs or image-based documents")
        print("  2. Missing dependencies (e.g., 'unstructured' package for .docx/.doc files)")
        print("  3. Corrupted or empty files")
        print("\nNOTE: Failed files are NOT marked as processed and can be retried after fixing the issue.")
        # Only update hashes for successfully loaded files
        # Failed files were already removed from new_hashes
        existing_hashes.update(new_hashes)
        update_file_hashes(persist_dir, existing_hashes)
        return
    
    # Use simplified Chroma initialization
    # For "new" mode on Streamlit Cloud, force in-memory to avoid readonly filesystem issues
    if mode == "new" and (os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser")):
        print("Using in-memory vector database for New document mode on Streamlit Cloud")
        from langchain_community.vectorstores import Chroma
        vectordb = Chroma(
            embedding_function=CONFIG.embedding_model,
            collection_name="langchain_new_docs"
        )
    else:
        vectordb = get_simple_chroma(
            persist_directory=persist_dir,
            embedding_function=CONFIG.embedding_model,
            collection_name="langchain",
            mode=mode  # Pass mode to handle in-memory for "new" on Streamlit Cloud
        )
    
    # Prepare lists of texts and metadata from the new chunks.
    new_texts = [doc.page_content for doc in chunked_new_docs]
    new_metadatas = [doc.metadata for doc in chunked_new_docs]
    
    # Process in batches to reduce rate limit issues.
    batch_size = 50  # Adjust this as needed.
    total_batches = (len(new_texts) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batch(es) of embeddings...")
    
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = new_texts[start:end]
        batch_metadatas = new_metadatas[start:end]
        
        max_retries = 5
        retries = 0
        success = False
        # Use exponential backoff for each batch.
        wait_time = 30  # start with 30 seconds
        while retries < max_retries and not success:
            try:
                vectordb.add_texts(batch_texts, metadatas=batch_metadatas)
                success = True
                print(f"Batch {batch_idx+1}/{total_batches} processed successfully.")
            except Exception as e:
                if "429" in str(e):
                    retries += 1
                    print(f"RateLimitError in batch {batch_idx+1} (attempt {retries}/{max_retries}): {e}")
                    print(f"Waiting for {wait_time} seconds before retrying batch {batch_idx+1}...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff
                else:
                    raise e
        if not success:
            raise Exception(f"Failed to process batch {batch_idx+1} after multiple retries due to rate limits.")

    # Update the file hashes metadata file with new hashes.
    # Skip hash updates for in-memory databases on Streamlit Cloud
    if not (mode == "new" and (os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser"))):
        existing_hashes.update(new_hashes)
        update_file_hashes(persist_dir, existing_hashes)
    else:
        print("Skipping hash file update for in-memory database on Streamlit Cloud")
    
    print(f"Updated VectorDB with {len(new_files)} new document(s).")
    print(f"VectorDB location: {persist_dir}")

def upload_data_manually(enable_multimodal=False) -> None:
    """
    Checks the data directory for new documents and generates embeddings only for them.
    It then updates the existing vector database accordingly.
    
    Args:
        enable_multimodal: If True and available, use multimodal indexing for PDFs
    """
    
    # Check for multimodal flag in config or environment
    if enable_multimodal or os.environ.get('HYBRID_RAG_MULTIMODAL_ENABLED', '').lower() == 'true':
        return upload_data_for_mode(mode="current", enable_multimodal=True)
    
    ensure_directory_access(CONFIG.persist_directory)
    existing_hashes = get_existing_file_hashes(CONFIG.persist_directory)
    
    data_directory = CONFIG.data_directory  # Use default data directory
    
    print(f"Checking directory: {data_directory}")
    print(f"Using persist directory: {CONFIG.persist_directory}")
    print(f"Found {len(existing_hashes)} previously processed files")
    
    # List all supported document files in the data directory.
    supported_extensions = ('.pdf', '.doc', '.docx', '.txt', '.csv', '.eml', '.msg', '.html', '.htm')
    data_files = [f for f in os.listdir(data_directory) 
                  if os.path.isfile(os.path.join(data_directory, f)) 
                  and f.lower().endswith(supported_extensions)]
    
    new_files = []
    new_hashes = {}
    
    for filename in data_files:
        full_path = os.path.join(data_directory, filename)
        # Skip empty files
        if os.path.getsize(full_path) == 0:
            print(f"Skipping empty file: {filename}")
            continue
        file_hash = calculate_file_hash(full_path)
        # Check if this exact file (by hash) has been processed before
        if filename not in existing_hashes or existing_hashes[filename] != file_hash:
            new_files.append(full_path)
            new_hashes[filename] = file_hash

    if not new_files:
        print("No new documents to process. VectorDB is up-to-date.")
        print(f"Files in directory: {data_files}")
        print(f"Files in hash db: {list(existing_hashes.keys())}")
        return

    print(f"Found {len(new_files)} new document(s) to process:")
    for f in new_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load new documents using the shared loader function
    new_docs = []
    failed_files = []  # Track files that failed to load
    for file_path in new_files:
        loaded_docs = load_document_by_type(file_path)
        if loaded_docs is None:
            # Loading failed - don't mark as processed
            failed_files.append(os.path.basename(file_path))
            filename = os.path.basename(file_path)
            # Remove from new_hashes so it won't be marked as processed
            if filename in new_hashes:
                del new_hashes[filename]
        elif loaded_docs:  # Successfully loaded with content
            new_docs.extend(loaded_docs)
        # If loaded_docs is [], it's an unsupported type but still mark as processed
    
    if failed_files:
        print(f"\nFailed to load {len(failed_files)} file(s) due to errors or missing dependencies:")
        for f in failed_files:
            print(f"  - {f}")
        print("These files will NOT be marked as processed and can be retried.")
    
    print(f"Number of new pages: {len(new_docs)}")
    
    # Instantiate PrepareVectorDB for processing
    pvdb = PrepareVectorDB(
        data_directory=CONFIG.data_directory,  # Not used here for loading new docs
        persist_directory=CONFIG.persist_directory,
        embedding_model_engine=CONFIG.embedding_model_engine,
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap
    )
    
    # Chunk the new documents.
    # Calling the private method via name mangling.
    chunked_new_docs = pvdb._PrepareVectorDB__chunk_documents(new_docs)
    print(f"Number of new chunks: {len(chunked_new_docs)}")
    
    # Skip if no chunks were created (e.g., scanned PDFs with no text)
    if len(chunked_new_docs) == 0:
        print("WARNING: No text could be extracted from the document(s).")
        print("This may be a scanned PDF or image-based document.")
        print("Skipping embedding for documents with no extractable text.")
        # Still update the hash to avoid reprocessing
        existing_hashes.update(new_hashes)
        update_file_hashes(CONFIG.persist_directory, existing_hashes)
        return
    
    # Use simplified Chroma initialization
    vectordb = get_simple_chroma(
        persist_directory=CONFIG.persist_directory,
        embedding_function=CONFIG.embedding_model,
        collection_name="langchain"
    )
    
    # Prepare lists of texts and metadata from the new chunks.
    new_texts = [doc.page_content for doc in chunked_new_docs]
    new_metadatas = [doc.metadata for doc in chunked_new_docs]
    
    # Process in batches to reduce rate limit issues.
    batch_size = 50  # Adjust this as needed.
    total_batches = (len(new_texts) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} batch(es) of embeddings...")
    
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_texts = new_texts[start:end]
        batch_metadatas = new_metadatas[start:end]
        
        max_retries = 5
        retries = 0
        success = False
        # Use exponential backoff for each batch.
        wait_time = 30  # start with 30 seconds
        while retries < max_retries and not success:
            try:
                vectordb.add_texts(batch_texts, metadatas=batch_metadatas)
                success = True
                print(f"Batch {batch_idx+1}/{total_batches} processed successfully.")
            except Exception as e:
                if "429" in str(e):
                    retries += 1
                    print(f"RateLimitError in batch {batch_idx+1} (attempt {retries}/{max_retries}): {e}")
                    print(f"Waiting for {wait_time} seconds before retrying batch {batch_idx+1}...")
                    time.sleep(wait_time)
                    wait_time *= 2  # Exponential backoff
                else:
                    raise e
        if not success:
            raise Exception(f"Failed to process batch {batch_idx+1} after multiple retries due to rate limits.")

    # Update the file hashes metadata file with new hashes.
    existing_hashes.update(new_hashes)
    update_file_hashes(CONFIG.persist_directory, existing_hashes)
    
    print(f"Updated VectorDB with {len(new_files)} new document(s).")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    enable_multimodal = False
    if len(sys.argv) > 1:
        if "--multimodal" in sys.argv or "-m" in sys.argv:
            enable_multimodal = True
            print("Multimodal indexing enabled via command line")
    
    # Check environment variable
    if os.environ.get('HYBRID_RAG_MULTIMODAL_ENABLED', '').lower() == 'true':
        enable_multimodal = True
        print("Multimodal indexing enabled via environment variable")
    
    upload_data_manually(enable_multimodal=enable_multimodal)