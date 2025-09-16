"""Enhanced document upload with Azure Blob Storage integration."""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# SQLite3 fix for Streamlit Cloud
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2
except ImportError:
    pass

from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig
from azure_blob_manager import AzureBlobManager
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)
from utils.docx_loader_fallback import SimpleDocxLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

CONFIG = LoadConfig()

class EnhancedDocumentProcessor:
    """Process documents with Azure Blob Storage integration for public URLs."""
    
    def __init__(
        self,
        data_directory: str,
        persist_directory: str,
        use_azure_blob: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the enhanced document processor.
        
        Args:
            data_directory: Directory containing documents to process
            persist_directory: Directory to persist the vector database
            use_azure_blob: Whether to upload documents to Azure Blob Storage
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.use_azure_blob = use_azure_blob
        
        # Initialize Azure Blob Manager if enabled
        self.blob_manager = None
        if use_azure_blob:
            try:
                self.blob_manager = AzureBlobManager()
                print("Azure Blob Storage initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Azure Blob Storage: {e}")
                print("Documents will be processed without public URLs")
                self.use_azure_blob = False
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings
        self.embeddings = self._init_embeddings()
        
        # Track uploaded documents
        self.uploaded_docs_metadata = []
    
    def _init_embeddings(self):
        """Initialize Azure OpenAI embeddings."""
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not azure_key or not azure_endpoint:
            raise ValueError("Azure OpenAI credentials required. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_key=azure_key
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on its file extension."""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                return PyPDFLoader(file_path).load()
            elif file_ext in ['.doc', '.docx']:
                if file_ext == '.docx':
                    return SimpleDocxLoader(file_path).load()
                else:
                    print(f"WARNING: .doc files not supported. Please convert to .docx format.")
                    return []
            elif file_ext == '.txt':
                return TextLoader(file_path, encoding='utf-8').load()
            elif file_ext == '.csv':
                return CSVLoader(file_path, encoding='utf-8').load()
            elif file_ext in ['.html', '.htm']:
                return UnstructuredHTMLLoader(file_path).load()
            else:
                print(f"Unsupported file type: {file_ext}")
                return []
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a single document: upload to Azure Blob and chunk it.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks with metadata including public URLs
        """
        # Load the document
        docs = self.load_document(file_path)
        if not docs:
            return []
        
        # Upload to Azure Blob Storage if enabled
        blob_info = None
        if self.use_azure_blob and self.blob_manager:
            try:
                print(f"Uploading {Path(file_path).name} to Azure Blob Storage...")
                blob_info = self.blob_manager.upload_document(file_path)
                print(f"  Uploaded successfully: {blob_info['blob_name']}")
                self.uploaded_docs_metadata.append(blob_info)
            except Exception as e:
                print(f"  Warning: Could not upload to Azure: {e}")
        
        # Chunk the documents
        chunks = self.text_splitter.split_documents(docs)
        
        # Enhance metadata with Azure Blob URLs
        if blob_info:
            for chunk in chunks:
                # Preserve existing metadata
                if not chunk.metadata:
                    chunk.metadata = {}
                
                # Add Azure Blob Storage URLs
                chunk.metadata['public_url'] = blob_info['public_url']
                chunk.metadata['sas_url'] = blob_info['sas_url']
                chunk.metadata['blob_name'] = blob_info['blob_name']
                chunk.metadata['full_path'] = file_path
                
                # Ensure source/filename is set
                if 'source' not in chunk.metadata:
                    chunk.metadata['source'] = Path(file_path).name
                if 'filename' not in chunk.metadata:
                    chunk.metadata['filename'] = Path(file_path).name
        
        return chunks
    
    def process_all_documents(self) -> List[Document]:
        """
        Process all documents in the data directory.
        
        Returns:
            List of all document chunks with metadata
        """
        all_chunks = []
        
        if isinstance(self.data_directory, list):
            # Process list of file paths
            for file_path in self.data_directory:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
        else:
            # Process directory
            data_path = Path(self.data_directory)
            supported_extensions = ('.pdf', '.doc', '.docx', '.txt', '.csv', '.html', '.htm')
            
            for file_path in data_path.glob('*'):
                if file_path.suffix.lower() in supported_extensions:
                    chunks = self.process_document(str(file_path))
                    all_chunks.extend(chunks)
        
        print(f"\nProcessed {len(self.uploaded_docs_metadata)} documents")
        print(f"Total chunks: {len(all_chunks)}")
        
        return all_chunks
    
    def create_vectordb(self, chunks: List[Document]) -> Chroma:
        """
        Create or update the vector database with processed chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Chroma vector database instance
        """
        print("\nCreating vector database...")
        
        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Create or update vector database
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist the database
        vectordb.persist()
        print(f"Vector database saved to {self.persist_directory}")
        
        # Save metadata for uploaded documents
        self.save_upload_metadata()
        
        return vectordb
    
    def save_upload_metadata(self):
        """Save metadata about uploaded documents for reference."""
        metadata_file = Path(self.persist_directory) / "azure_blob_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump({
                'uploaded_documents': self.uploaded_docs_metadata,
                'use_azure_blob': self.use_azure_blob,
                'container_name': self.blob_manager.container_name if self.blob_manager else None
            }, f, indent=2)
        
        print(f"Metadata saved to {metadata_file}")
    
    def run(self):
        """Run the complete document processing pipeline."""
        print("Starting enhanced document processing with Azure Blob Storage...")
        print("=" * 60)
        
        # Process all documents
        chunks = self.process_all_documents()
        
        if not chunks:
            print("No documents to process.")
            return
        
        # Create vector database
        vectordb = self.create_vectordb(chunks)
        
        print("\n" + "=" * 60)
        print("Document processing completed successfully!")
        
        if self.uploaded_docs_metadata:
            print(f"\nDocuments available at public URLs:")
            for doc in self.uploaded_docs_metadata:
                print(f"  - {doc['file_name']}: {doc['public_url']}")
        
        return vectordb


def main():
    """Main function to run document processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents with Azure Blob Storage integration")
    parser.add_argument("--data-dir", default="data", help="Directory containing documents")
    parser.add_argument("--vectordb-dir", default="vectordb", help="Directory for vector database")
    parser.add_argument("--no-azure", action="store_true", help="Disable Azure Blob Storage upload")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for text splitting")
    
    args = parser.parse_args()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create processor and run
    processor = EnhancedDocumentProcessor(
        data_directory=args.data_dir,
        persist_directory=args.vectordb_dir,
        use_azure_blob=not args.no_azure,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    processor.run()


if __name__ == "__main__":
    main()