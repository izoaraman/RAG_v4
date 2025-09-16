#!/usr/bin/env python3
"""Create vector database from Azure Blob Storage documents with proper metadata."""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# SQLite3 fix for Windows/Linux compatibility
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
except ImportError:
    pass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import Document
from azure.storage.blob import BlobServiceClient
import pypdf
from io import BytesIO

# Configuration
AZURE_BLOB_URL = "https://sandbox3190080146.blob.core.windows.net/documents/"
VECTORDB_PATH = "vectordb/azure_docs_db"
METADATA_FILE = "vectordb/azure_blob_metadata.json"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BATCH_SIZE = 50  # Process documents in batches to avoid memory issues

def load_azure_metadata() -> Dict[str, Any]:
    """Load the Azure blob metadata from JSON file."""
    metadata_path = Path(METADATA_FILE)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def download_and_process_pdf(blob_service_client, container_name: str, blob_name: str) -> str:
    """Download PDF from Azure Blob and extract text."""
    try:
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )

        # Download blob to memory
        blob_data = blob_client.download_blob().readall()
        pdf_file = BytesIO(blob_data)

        # Extract text from PDF
        pdf_reader = pypdf.PdfReader(pdf_file)
        text_content = []

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"[Page {page_num + 1}]\n{page_text}")
            except Exception as e:
                print(f"    Warning: Could not extract page {page_num + 1}: {e}")
                continue

        return "\n\n".join(text_content)

    except Exception as e:
        print(f"    Error processing {blob_name}: {e}")
        return ""

def create_document_with_metadata(
    text: str,
    doc_info: Dict[str, Any],
    page_content: str = None
) -> Document:
    """Create a LangChain Document with proper Azure metadata."""

    # Extract key information
    file_name = doc_info.get('file_name', 'unknown.pdf')
    blob_name = doc_info.get('blob_name', file_name)
    public_url = doc_info.get('public_url', '')

    # Ensure we have the correct URL
    if not public_url:
        # URL encode the blob name for special characters
        safe_blob_name = blob_name.replace('%', '%25').replace('#', '%23').replace('?', '%3F')
        public_url = f"{AZURE_BLOB_URL}{safe_blob_name}"

    # Create metadata
    metadata = {
        "source": file_name,  # Clean filename for display
        "blob_name": blob_name,  # Actual blob name with timestamp
        "pdf_url": public_url,  # Full URL for View link
        "container": doc_info.get('container', 'documents'),
        "file_size": doc_info.get('file_size', 0),
        "upload_timestamp": doc_info.get('upload_timestamp', ''),
        "content_type": doc_info.get('content_type', 'application/pdf')
    }

    # Use provided page content or full text
    content = page_content if page_content else text[:CHUNK_SIZE]

    return Document(page_content=content, metadata=metadata)

def process_documents_batch(
    documents: List[Dict[str, Any]],
    blob_service_client,
    container_name: str,
    text_splitter
) -> List[Document]:
    """Process a batch of documents and return Document objects."""

    all_docs = []

    for i, doc_info in enumerate(documents, 1):
        blob_name = doc_info.get('blob_name')
        file_name = doc_info.get('file_name')

        print(f"  [{i}/{len(documents)}] Processing: {file_name}")

        # Download and extract text
        text = download_and_process_pdf(blob_service_client, container_name, blob_name)

        if not text:
            print(f"    Skipping: No text extracted")
            continue

        # Split text into chunks
        try:
            chunks = text_splitter.split_text(text)

            # Create Document objects with metadata
            for chunk in chunks:
                doc = create_document_with_metadata(text, doc_info, chunk)
                all_docs.append(doc)

            print(f"    Created {len(chunks)} chunks")

        except Exception as e:
            print(f"    Error chunking document: {e}")
            continue

    return all_docs

def main():
    """Main function to create vector database from Azure Blob documents."""

    print("=" * 60)
    print("Creating Vector Database from Azure Blob Documents")
    print("=" * 60)

    # Load metadata
    print("\n1. Loading Azure metadata...")
    metadata = load_azure_metadata()
    uploaded_docs = metadata.get('uploaded_documents', [])

    if not uploaded_docs:
        print("Error: No uploaded documents found in metadata")
        return False

    print(f"   Found {len(uploaded_docs)} documents in metadata")

    # Initialize Azure Blob client
    print("\n2. Initializing Azure Blob client...")
    try:
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment")

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = metadata.get('container_name', 'documents')
        print(f"   Connected to container: {container_name}")
    except Exception as e:
        print(f"Error: Could not connect to Azure Blob Storage: {e}")
        return False

    # Initialize embeddings
    print("\n3. Initializing Azure OpenAI embeddings...")
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        print("   Embeddings initialized")
    except Exception as e:
        print(f"Error: Could not initialize embeddings: {e}")
        return False

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    # Clear existing vector database
    vectordb_path = Path(VECTORDB_PATH)
    if vectordb_path.exists():
        print(f"\n4. Clearing existing vector database at {VECTORDB_PATH}...")
        import shutil
        shutil.rmtree(vectordb_path)

    # Process documents in batches
    print(f"\n5. Processing {len(uploaded_docs)} documents in batches of {BATCH_SIZE}...")

    vectorstore = None
    total_chunks = 0

    for batch_start in range(0, len(uploaded_docs), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(uploaded_docs))
        batch_docs = uploaded_docs[batch_start:batch_end]

        print(f"\nBatch {batch_start//BATCH_SIZE + 1}: Documents {batch_start+1}-{batch_end}")

        # Process batch
        documents = process_documents_batch(
            batch_docs,
            blob_service_client,
            container_name,
            text_splitter
        )

        if not documents:
            print("  No documents processed in this batch")
            continue

        # Add to vector store
        print(f"  Adding {len(documents)} chunks to vector store...")
        try:
            if vectorstore is None:
                # Create new vector store with first batch
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=VECTORDB_PATH
                )
            else:
                # Add to existing vector store
                vectorstore.add_documents(documents)

            total_chunks += len(documents)
            print(f"  Total chunks so far: {total_chunks}")

            # Persist after each batch
            vectorstore.persist()

        except Exception as e:
            print(f"  Error adding to vector store: {e}")
            continue

        # Brief pause between batches to avoid rate limits
        if batch_end < len(uploaded_docs):
            print("  Pausing before next batch...")
            time.sleep(2)

    # Final summary
    print("\n" + "=" * 60)
    print("VECTOR DATABASE CREATION COMPLETE")
    print("=" * 60)
    print(f"Total documents processed: {len(uploaded_docs)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Vector database location: {VECTORDB_PATH}")
    print(f"Metadata preserved for View links")

    # Save summary
    summary_path = Path(VECTORDB_PATH) / "creation_summary.json"
    summary = {
        "total_documents": len(uploaded_docs),
        "total_chunks": total_chunks,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "azure_blob_url": AZURE_BLOB_URL,
        "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("\nNext steps:")
    print("1. Test the vector database locally")
    print("2. Add vectordb folder to git and push to GitHub")
    print("3. The Streamlit Cloud app will use this pre-built database")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)