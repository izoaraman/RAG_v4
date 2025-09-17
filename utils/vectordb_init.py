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

def rebuild_vectordb_from_azure(persist_directory: str, embedding_function) -> bool:
    """Rebuild vector database from Azure Blob Storage."""

    logger.info("Starting vector database rebuild from Azure...")
    logger.info(f"Target directory: {persist_directory}")

    try:
        # Load Azure metadata
        metadata_file = "vectordb/azure_blob_metadata.json"
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
        batch_size = 10  # Process in smaller batches

        for i in range(0, len(uploaded_docs), batch_size):
            batch = uploaded_docs[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(uploaded_docs)-1)//batch_size + 1}")

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

        # Create vector database - simpler approach
        # Clear existing directory
        persist_path = Path(persist_directory)
        if persist_path.exists():
            import shutil
            shutil.rmtree(persist_path)

        persist_path.mkdir(parents=True, exist_ok=True)

        # Use direct Chroma initialization (without the tenants table issue)
        from langchain_community.vectorstores import Chroma

        # Split documents into texts and metadatas
        texts = [doc.page_content for doc in all_docs]
        metadatas = [doc.metadata for doc in all_docs]

        # Create vector store using from_texts which avoids the tenants issue
        vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embedding_function,
            persist_directory=persist_directory,
            collection_name="langchain"
        )

        logger.info(f"Successfully created vector database with {len(all_docs)} documents")

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

        # On Streamlit Cloud, ALWAYS rebuild from Azure for reliability
        # This ensures we always have the latest documents from Azure
        is_streamlit_cloud = os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser")

        if is_streamlit_cloud or force_rebuild:
            logger.info("Streamlit Cloud detected or force rebuild requested. Rebuilding from Azure...")
            if rebuild_vectordb_from_azure(persist_directory, embeddings):
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