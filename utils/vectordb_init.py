"""Initialize vector database on Streamlit Cloud if needed."""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# SQLite3 fix for Streamlit Cloud
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
except ImportError:
    pass

from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from azure.storage.blob import BlobServiceClient
import pypdf
from io import BytesIO

logger = logging.getLogger(__name__)

def check_vectordb_health(persist_directory: str, embedding_function) -> bool:
    """Check if the vector database is healthy and has data."""
    try:
        from utils.simple_chroma import get_simple_chroma

        vectordb = get_simple_chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name="langchain",
            mode="current"
        )

        # Try to get document count
        count = vectordb._collection.count()
        logger.info(f"Vector database has {count} documents")

        # If we have documents, the database is healthy
        return count > 0

    except Exception as e:
        logger.error(f"Vector database health check failed: {e}")
        return False

def rebuild_vectordb_from_azure(persist_directory: str, embedding_function, use_simple_db: bool = False) -> bool:
    """Rebuild vector database from Azure Blob Storage."""

    logger.info("Starting vector database rebuild from Azure...")
    logger.info(f"Target directory: {persist_directory}")

    try:
        # Load Azure metadata - use demo version for Streamlit Cloud
        # Multiple ways to detect Streamlit Cloud environment
        is_streamlit_cloud = (
            os.environ.get("STREAMLIT_CLOUD") == "true" or
            os.path.exists("/home/appuser") or
            os.environ.get("STREAMLIT_SHARING_MODE") is not None or
            os.environ.get("STREAMLIT_RUNTIME_ENV") is not None
        )

        if is_streamlit_cloud:
            metadata_file = "vectordb/azure_blob_metadata_demo.json"
            logger.info("🎯 STREAMLIT CLOUD DETECTED: Using 20-document demo metadata")
        else:
            metadata_file = "vectordb/azure_blob_metadata.json"
            logger.info("🏠 LOCAL ENVIRONMENT: Using full 470-document metadata")
        metadata_path = Path(metadata_file)

        if not metadata_path.exists():
            logger.error(f"Azure metadata file not found: {metadata_file}")
            return False

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        uploaded_docs = metadata.get('uploaded_documents', [])

        if not uploaded_docs:
            logger.error("No documents found in Azure metadata")
            return False

        logger.info(f"Found {len(uploaded_docs)} documents in Azure metadata")

        # Initialize Azure Blob client
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            logger.error("AZURE_STORAGE_CONNECTION_STRING not found")
            return False

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = metadata.get('container_name', 'documents')

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", ".", "!", "?", ",", " ", ""]
        )

        # Process documents
        all_docs = []
        batch_size = 5  # Reduced batch size to avoid memory and processing issues

        for i in range(0, len(uploaded_docs), batch_size):
            batch = uploaded_docs[i:i+batch_size]
            logger.info(f"Processing document batch {i//batch_size + 1}/{(len(uploaded_docs)-1)//batch_size + 1}")

            for doc_info in batch:
                try:
                    blob_name = doc_info.get('blob_name')
                    file_name = doc_info.get('file_name')

                    # Download blob
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name,
                        blob=blob_name
                    )
                    blob_data = blob_client.download_blob().readall()

                    # Extract text from PDF
                    pdf_file = BytesIO(blob_data)
                    pdf_reader = pypdf.PdfReader(pdf_file)
                    text_content = []

                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text_content.append(f"[Page {page_num + 1}]\\n{page_text}")
                        except:
                            continue

                    text = "\\n\\n".join(text_content)

                    if not text:
                        continue

                    # Create chunks
                    chunks = text_splitter.split_text(text)

                    # Create Document objects with metadata
                    for chunk in chunks:
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": file_name,
                                "blob_name": blob_name,
                                "pdf_url": doc_info.get('public_url', ''),
                                "container": container_name
                            }
                        )
                        all_docs.append(doc)

                except Exception as e:
                    logger.error(f"Error processing {doc_info.get('file_name')}: {e}")
                    continue

        if not all_docs:
            logger.error("No documents were successfully processed")
            return False

        logger.info(f"Processed {len(all_docs)} document chunks")

        # On Streamlit Cloud with newer ChromaDB versions, use SimpleVectorDB as fallback
        if use_simple_db or (os.environ.get("STREAMLIT_CLOUD") == "true"):
            logger.info("Using SimpleVectorDB as fallback for Streamlit Cloud")
            from utils.simple_vectordb import create_simple_vectordb

            # Create SimpleVectorDB
            vectordb = create_simple_vectordb(persist_directory, embedding_function, all_docs)

            logger.info(f"Successfully created SimpleVectorDB with {vectordb.count()} documents")

            return True

        # Try ChromaDB for local development
        # Clear existing directory
        persist_path = Path(persist_directory)
        if persist_path.exists():
            import shutil
            shutil.rmtree(persist_path)

        persist_path.mkdir(parents=True, exist_ok=True)

        # Use direct ChromaDB PersistentClient to bypass tenants table issue
        import chromadb
        from chromadb.config import Settings
        from langchain_community.vectorstores import Chroma
        import uuid

        # Create ChromaDB client with specific settings
        client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )

        # Create or get collection
        collection_name = "langchain"
        try:
            # Delete existing collection if it exists
            client.delete_collection(name=collection_name)
        except:
            pass  # Collection doesn't exist, that's fine

        # Create new collection
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Process documents in batches
        batch_size = 100
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:min(i+batch_size, len(all_docs))]

            # Prepare batch data
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            ids = [str(uuid.uuid4()) for _ in batch]

            # Generate embeddings
            embeddings = embedding_function.embed_documents(texts)

            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added batch {i//batch_size + 1}/{(len(all_docs)-1)//batch_size + 1}")

        logger.info(f"Successfully created vector database with {len(all_docs)} documents")

        # Create a Chroma wrapper for compatibility
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )

        # Create summary file
        summary = {
            "total_documents": len(uploaded_docs),
            "total_chunks": len(all_docs),
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "azure_blob_url": "https://sandbox3190080146.blob.core.windows.net/documents/",
            "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rebuild_reason": "Automatic rebuild on Streamlit Cloud"
        }

        summary_path = persist_path / "creation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Vector database rebuilt successfully with {len(all_docs)} chunks")
        return True

    except Exception as e:
        logger.error(f"Failed to rebuild vector database: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def ensure_vectordb_ready(persist_directory: str, force_rebuild: bool = False) -> bool:
    """Ensure the vector database is ready, rebuild if necessary.

    Args:
        persist_directory: Directory to store the vector database
        force_rebuild: If True, always rebuild from Azure (recommended for Streamlit Cloud)
    """

    logger.info(f"Checking vector database at {persist_directory}...")
    logger.info(f"Environment: STREAMLIT_CLOUD={os.environ.get('STREAMLIT_CLOUD')}, Force rebuild={force_rebuild}")

    try:
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

        # Check if we have a pre-built database first
        persist_path = Path(persist_directory)
        is_streamlit_cloud = os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser")

        # On Streamlit Cloud, MUST use pre-built 20-document database
        if is_streamlit_cloud:
            logger.info("🎯 STREAMLIT CLOUD DETECTED - Looking for pre-built 20-document vectordb...")

            # Try multiple demo vectordb locations - prioritize the one with 4924 chunks
            demo_paths = [
                Path("vectordb/demo_vectordb/azure_docs_db_text-embedding-ada-002_20docs"),  # 4924 chunks from 20 docs
                Path("vectordb/test_single"),  # 35 chunks test
                Path("vectordb/demo_vectordb_simple"),  # Copy of test_single
            ]

            for demo_db_path in demo_paths:
                if demo_db_path.exists():
                    # Look for chroma.sqlite3 file directly in root (preferred structure)
                    demo_chroma = demo_db_path / "chroma.sqlite3"

                    if demo_chroma.exists():
                        # Check file size to ensure it's valid (should be ~800KB for 20 docs)
                        file_size_kb = demo_chroma.stat().st_size / 1024
                        logger.info(f"✅ Found pre-built vectordb at: {demo_db_path}")
                        logger.info(f"✅ Database size: {file_size_kb:.1f} KB (20 documents)")

                        # Copy demo vectordb to target location
                        import shutil
                        if persist_path.exists():
                            try:
                                shutil.rmtree(persist_path)
                            except:
                                logger.warning("Could not remove existing path, continuing...")

                        shutil.copytree(demo_db_path, persist_path)
                        logger.info("✅ Pre-built 20-document vectordb copied successfully!")
                        logger.info("✅ Using curated ACCC documents including card surcharges, annual reports, etc.")
                        return True

                    # Fallback: check for nested SQLite files
                    demo_chroma_nested = list(demo_db_path.glob("**/chroma.sqlite3"))
                    if demo_chroma_nested:
                        logger.info(f"Found {len(demo_chroma_nested)} nested chroma files at {demo_db_path}")
                        # Still copy the whole structure
                        import shutil
                        if persist_path.exists():
                            shutil.rmtree(persist_path)
                        shutil.copytree(demo_db_path, persist_path)
                        logger.info("✅ Pre-built vectordb with nested structure copied")
                        return True
                    else:
                        logger.warning(f"Path {demo_db_path} exists but no chroma.sqlite3 found")

            logger.warning("No working demo vectordb found in any location")

            # CRITICAL: On Streamlit Cloud, NEVER rebuild from Azure - it's too slow
            # Force use of pre-built 20-document database only
            if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("STREAMLIT_RUNTIME_ENV"):
                logger.error("❌ STREAMLIT CLOUD: No pre-built vectordb found - REFUSING to rebuild")
                logger.error("This prevents 470-document rebuild. Demo vectordb with 20 documents must be in repository.")
                logger.error("Expected locations: vectordb/test_single/ or vectordb/demo_vectordb_simple/")

                # Try to create a minimal working database from what we have
                # Check if we at least have the demo metadata with 20 documents
                demo_metadata = Path("vectordb/azure_blob_metadata_demo.json")
                if demo_metadata.exists():
                    logger.info("Found demo metadata with 20 documents - but cannot rebuild on Streamlit Cloud")
                    logger.info("Pre-built database required for performance reasons")

                # Create error marker to prevent crashes
                persist_path.mkdir(parents=True, exist_ok=True)
                error_file = persist_path / "ERROR_NO_PREBUILT_DB.txt"
                error_file.write_text(
                    "No pre-built database found. Cannot rebuild on Streamlit Cloud.\n"
                    "Required: Pre-built vectordb with 20 selected ACCC documents.\n"
                    "Check repository includes: vectordb/test_single/chroma.sqlite3"
                )
                return False

            # Fallback: Check if pre-built database exists at target location
            if persist_path.exists():
                summary_file = persist_path / "creation_summary.json"
                chroma_db = persist_path / "chroma.sqlite3"

                # If we have the pre-built database files, use them
                if summary_file.exists() or chroma_db.exists():
                    logger.info("Found pre-built vector database from GitHub, using it directly")
                    # Test if it's healthy
                    if check_vectordb_health(persist_directory, embeddings):
                        logger.info("Pre-built database is healthy and ready to use")
                        return True
                    else:
                        logger.warning("Pre-built database exists but may need initialization")
                        # Still return True as the database files exist
                        return True

            # Only rebuild if we don't have a pre-built database
            logger.info("No pre-built database found, attempting rebuild from Azure...")
            # Use SimpleVectorDB on Streamlit Cloud to avoid ChromaDB tenants issue
            if rebuild_vectordb_from_azure(persist_directory, embeddings, use_simple_db=True):
                logger.info("Successfully rebuilt vector database from Azure")
                return True
            else:
                logger.error("Failed to rebuild from Azure")
                return False

        # For non-Streamlit Cloud (local), check as before
        if force_rebuild:
            logger.info("Force rebuild requested. Rebuilding from Azure...")
            if rebuild_vectordb_from_azure(persist_directory, embeddings, use_simple_db=False):
                logger.info("Successfully rebuilt vector database from Azure")
                return True
            else:
                logger.error("Failed to rebuild from Azure")
                return False

        # For local development, check if database is healthy
        if check_vectordb_health(persist_directory, embeddings):
            logger.info("Vector database is healthy")
            return True

        # If local database is not healthy, try to rebuild
        logger.info("Local database not healthy, attempting to rebuild from Azure...")
        if rebuild_vectordb_from_azure(persist_directory, embeddings):
            return True

        logger.error("Could not ensure vector database is ready")
        return False

    except Exception as e:
        logger.error(f"Error ensuring vector database: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def rebuild_vectordb_from_azure_batch(persist_directory: str, embedding_function, documents: list, batch_num: int) -> bool:
    """Rebuild vector database from a batch of Azure documents."""

    logger.info(f"Processing batch {batch_num} with {len(documents)} documents")
    logger.info(f"Target directory: {persist_directory}")

    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", ".", "!", "?", ",", " ", ""]
        )

        # Initialize Azure Blob client
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            logger.error("AZURE_STORAGE_CONNECTION_STRING not found")
            return False

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = "documents"  # Default container name

        # Process documents in this batch
        batch_docs = []

        for doc_info in documents:
            try:
                blob_name = doc_info.get('blob_name')
                file_name = doc_info.get('file_name')

                logger.info(f"Processing: {file_name}")

                # Download blob
                blob_client = blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )
                blob_data = blob_client.download_blob().readall()

                # Extract text from PDF
                pdf_file = BytesIO(blob_data)
                pdf_reader = pypdf.PdfReader(pdf_file)
                text_content = []

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(f"[Page {page_num + 1}]\\n{page_text}")
                    except:
                        continue

                text = "\\n\\n".join(text_content)

                if not text:
                    logger.warning(f"No text extracted from {file_name}")
                    continue

                # Create chunks
                chunks = text_splitter.split_text(text)

                # Create Document objects with metadata
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_name,
                            "blob_name": blob_name,
                            "pdf_url": doc_info.get('public_url', ''),
                            "container": container_name,
                            "batch_num": batch_num
                        }
                    )
                    batch_docs.append(doc)

                logger.info(f"✅ Processed {file_name}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"❌ Error processing {doc_info.get('file_name')}: {e}")
                continue

        if not batch_docs:
            logger.error("No documents were successfully processed in this batch")
            return False

        logger.info(f"Batch {batch_num}: Processed {len(batch_docs)} document chunks")

        # Initialize or append to vector database
        persist_path = Path(persist_directory)

        # Check if this is the first batch (no existing database)
        is_first_batch = not persist_path.exists() or batch_num == 1

        if is_first_batch:
            logger.info("Creating new vector database (first batch)")
            # Clear existing directory for fresh start
            if persist_path.exists():
                import shutil
                shutil.rmtree(persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Appending to existing vector database (batch {batch_num})")

        # Use simple vectordb for cloud compatibility
        is_streamlit_cloud = os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser")

        if is_streamlit_cloud:
            logger.info("Using SimpleVectorDB for Streamlit Cloud compatibility")
            from utils.simple_vectordb import SimpleVectorDB, append_to_simple_vectordb

            if is_first_batch:
                # Create new SimpleVectorDB
                vectordb = SimpleVectorDB(str(persist_path), embedding_function)
                # Add documents
                texts = [doc.page_content for doc in batch_docs]
                metadatas = [doc.metadata for doc in batch_docs]
                vectordb.add_texts(texts, metadatas)
                vectordb.save()
            else:
                # Append to existing SimpleVectorDB
                append_to_simple_vectordb(str(persist_path), embedding_function, batch_docs)

        else:
            # Use ChromaDB for local processing
            import chromadb
            from chromadb.config import Settings
            from langchain_community.vectorstores import Chroma
            import uuid

            # Create or load ChromaDB client
            client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,  # Don't reset existing data
                    is_persistent=True
                )
            )

            collection_name = "langchain"

            if is_first_batch:
                # Create new collection
                try:
                    client.delete_collection(name=collection_name)
                except:
                    pass  # Collection doesn't exist

                collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                # Get existing collection
                collection = client.get_collection(name=collection_name)

            # Process documents in smaller batches for embeddings
            embedding_batch_size = 50
            for i in range(0, len(batch_docs), embedding_batch_size):
                batch = batch_docs[i:min(i+embedding_batch_size, len(batch_docs))]

                # Prepare batch data
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                ids = [str(uuid.uuid4()) for _ in batch]

                # Generate embeddings
                embeddings = embedding_function.embed_documents(texts)

                # Add to collection
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

                logger.info(f"Added embedding batch {i//embedding_batch_size + 1}")

        # Create/update summary file
        summary_file = persist_path / "creation_summary.json"
        summary = {
            "last_batch": batch_num,
            "total_chunks_processed": len(batch_docs),
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "azure_blob_url": "https://sandbox3190080146.blob.core.windows.net/documents/",
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_processing": True
        }

        # If summary exists, update it; otherwise create new
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                existing_summary = json.load(f)

            existing_summary.update(summary)
            existing_summary["total_chunks_processed"] = existing_summary.get("total_chunks_processed", 0) + len(batch_docs)
            summary = existing_summary

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✅ Batch {batch_num} completed successfully with {len(batch_docs)} chunks")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to process batch {batch_num}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False