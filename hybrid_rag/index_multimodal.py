"""
Multimodal Document Indexing for Hybrid RAG
Extracts and indexes text, tables, captions, and OCR from documents
"""

import os
import re
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Document processing
import pdfplumber
from tabula import read_pdf
import pytesseract
from PIL import Image
import io

# Import enhanced loader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.enhanced_document_loader import EnhancedDocumentLoader

# ML and embeddings
import numpy as np

# Vector store
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LangchainDocument
from langchain_openai import AzureOpenAIEmbeddings

# Text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalChunk:
    """Represents a chunk of multimodal content"""
    content: str
    modality: str  # 'text', 'table', 'table_image', 'caption', 'ocr'
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class MultimodalIndexer:
    """Handles extraction and indexing of multimodal document content"""

    def __init__(
        self,
        persist_directory: str = "data/vectordb/multimodal/chroma/",
        collection_name: str = "docs_mm",
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        ocr_threshold: int = 200,  # Min chars to skip OCR
        enable_ocr: bool = True,
        enable_table_detection: bool = True,
        enable_file_hashing: bool = True,
        use_azure_ocr: bool = True
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_threshold = ocr_threshold
        self.enable_ocr = enable_ocr
        self.enable_table_detection = enable_table_detection
        self.enable_file_hashing = enable_file_hashing
        self.use_azure_ocr = use_azure_ocr

        # Initialize Azure OpenAI embeddings
        logger.info(f"Loading Azure embedding model: {embedding_model}")
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", embedding_model),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Initialize enhanced document loader
        self.enhanced_loader = EnhancedDocumentLoader(
            enable_ocr=enable_ocr,
            enable_table_detection=enable_table_detection,
            enable_file_hashing=enable_file_hashing,
            hash_directory=persist_directory,
            use_azure_ocr=use_azure_ocr,
            ocr_threshold=ocr_threshold
        )

        # Ensure directories exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize or load vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        return self.enhanced_loader.calculate_file_hash(file_path)

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """Extract text content from PDF pages"""
        text_pages = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        text_pages.append((i + 1, text))
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")

        return text_pages

    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using tabula with fallback to pdfplumber"""
        tables = []

        # Try tabula first
        try:
            dfs = read_pdf(pdf_path, pages='all', multiple_tables=True, silent=True)
            for i, df in enumerate(dfs):
                if not df.empty:
                    # Convert dataframe to structured text
                    headers = list(df.columns)
                    table_text = f"Table Headers: {', '.join(map(str, headers))}\n"
                    table_text += df.to_string()

                    tables.append({
                        'table_index': i,
                        'headers': headers,
                        'content': table_text,
                        'rows': len(df)
                    })
        except Exception as e:
            logger.warning(f"Tabula failed, falling back to pdfplumber: {e}")

            # Fallback to pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_tables = page.extract_tables()
                        for table_idx, table in enumerate(page_tables):
                            if table and len(table) > 1:
                                headers = table[0] if table[0] else []
                                table_text = f"Table Headers: {', '.join(map(str, headers))}\n"

                                for row in table[1:]:
                                    if row:
                                        table_text += ' | '.join(map(str, row)) + '\n'

                                tables.append({
                                    'table_index': len(tables),
                                    'page': page_num + 1,
                                    'headers': headers,
                                    'content': table_text,
                                    'rows': len(table) - 1
                                })
            except Exception as e:
                logger.error(f"Error extracting tables with pdfplumber: {e}")

        return tables

    def extract_captions(self, text: str) -> List[str]:
        """Extract figure and table captions using regex"""
        captions = []

        # Common caption patterns
        patterns = [
            r'(Figure\s+\d+[:\.\s]+[^\n]+)',
            r'(Table\s+\d+[:\.\s]+[^\n]+)',
            r'(Fig\.\s+\d+[:\.\s]+[^\n]+)',
            r'(Chart\s+\d+[:\.\s]+[^\n]+)',
            r'(Exhibit\s+\d+[:\.\s]+[^\n]+)',
            r'(Diagram\s+\d+[:\.\s]+[^\n]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            captions.extend(matches)

        return captions

    def extract_ocr_from_pdf(self, pdf_path: str, page_num: int) -> Optional[str]:
        """Extract text from PDF page using OCR (for scanned pages)"""
        if not self.enable_ocr:
            return None

        try:
            # Convert PDF page to image using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    im = page.to_image(resolution=150)

                    # Convert to PIL Image
                    pil_image = im.original

                    # Perform OCR
                    ocr_text = pytesseract.image_to_string(pil_image)

                    # Clean up OCR text
                    ocr_text = ' '.join(ocr_text.split())

                    return ocr_text if ocr_text else None
        except Exception as e:
            logger.warning(f"OCR failed for page {page_num}: {e}")

        return None

    def process_pdf(self, pdf_path: str) -> List[MultimodalChunk]:
        """Process a PDF file and extract multimodal content using enhanced loader"""
        chunks = []
        doc_id = self.calculate_file_hash(pdf_path)
        filename = os.path.basename(pdf_path)

        logger.info(f"Processing PDF: {filename}")

        # Check if file already processed
        if self.enable_file_hashing and self.enhanced_loader.is_file_processed(pdf_path):
            logger.info(f"File already processed (hash exists): {filename}")
            return chunks

        # Use enhanced loader to extract documents
        doc_results = self.enhanced_loader.load_documents(
            [pdf_path],
            include_tables=self.enable_table_detection,
            include_ocr=self.enable_ocr
        )

        text_docs = doc_results.get('text', [])
        table_docs = doc_results.get('table', [])

        # Process text documents
        all_text = ""
        for doc in text_docs:
            all_text += doc.page_content + "\n"

            # Check modality from metadata
            modality = doc.metadata.get('modality', 'text')

            # Chunk large text documents
            if len(doc.page_content) > self.chunk_size:
                text_chunks = self.text_splitter.split_text(doc.page_content)
                for i, chunk_text in enumerate(text_chunks):
                    chunks.append(MultimodalChunk(
                        content=chunk_text,
                        modality=modality,
                        metadata={
                            **doc.metadata,
                            'doc_id': doc_id,
                            'source_path': pdf_path,
                            'filename': filename,
                            'chunk_index': i
                        }
                    ))
            else:
                chunks.append(MultimodalChunk(
                    content=doc.page_content,
                    modality=modality,
                    metadata={
                        **doc.metadata,
                        'doc_id': doc_id,
                        'source_path': pdf_path,
                        'filename': filename
                    }
                ))

        # Process table documents (already in structured JSON format)
        for doc in table_docs:
            # Tables are already properly formatted, just add them
            modality = doc.metadata.get('modality', 'table')
            chunks.append(MultimodalChunk(
                content=doc.page_content,
                modality=modality,
                metadata={
                    **doc.metadata,
                    'doc_id': doc_id,
                    'source_path': pdf_path,
                    'filename': filename
                }
            ))

        # Extract captions from the combined text
        if all_text:
            captions = self.extract_captions(all_text)
            for caption in captions:
                chunks.append(MultimodalChunk(
                    content=caption,
                    modality='caption',
                    metadata={
                        'doc_id': doc_id,
                        'source_path': pdf_path,
                        'filename': filename,
                        'extraction_method': 'regex'
                    }
                ))

        # Log extraction statistics
        logger.info(f"Extracted {len(chunks)} chunks from {filename}")
        logger.info(f"  - Text chunks: {sum(1 for c in chunks if c.modality == 'text')}")
        logger.info(f"  - Table chunks: {sum(1 for c in chunks if c.modality == 'table')}")
        logger.info(f"  - Table w/ images: {sum(1 for c in chunks if c.modality == 'table_image')}")
        logger.info(f"  - Caption chunks: {sum(1 for c in chunks if c.modality == 'caption')}")
        logger.info(f"  - OCR chunks: {sum(1 for c in chunks if c.modality == 'ocr')}")

        return chunks

    def embed_chunks(self, chunks: List[MultimodalChunk]) -> List[MultimodalChunk]:
        """Generate embeddings for chunks using Azure OpenAI"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        for chunk in chunks:
            # Generate embedding using Azure OpenAI
            embedding = self.embeddings.embed_query(chunk.content)
            chunk.embedding = np.array(embedding)

        return chunks

    def index_chunks(self, chunks: List[MultimodalChunk]):
        """Index chunks into Chroma vector store"""
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Indexing {len(chunks)} chunks to Chroma")

        # Convert to LangChain documents
        documents = []
        for chunk in chunks:
            doc = LangchainDocument(
                page_content=chunk.content,
                metadata=chunk.metadata
            )
            doc.metadata['modality'] = chunk.modality
            documents.append(doc)

        # Add to vector store in batches
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.vectorstore.add_documents(batch)

        # Persist the vector store
        self.vectorstore.persist()
        logger.info(f"Successfully indexed {len(chunks)} chunks")

    def process_document(self, file_path: str) -> bool:
        """Process a single document and index its multimodal content"""
        try:
            # Currently supports PDF, can be extended for other formats
            if file_path.lower().endswith('.pdf'):
                chunks = self.process_pdf(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return False

            # Generate embeddings
            chunks = self.embed_chunks(chunks)

            # Index chunks
            self.index_chunks(chunks)

            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False

    def process_directory(self, directory: str) -> Dict[str, bool]:
        """Process all PDFs in a directory"""
        results = {}

        pdf_files = [f for f in os.listdir(directory)
                    if f.lower().endswith('.pdf')]

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            file_path = os.path.join(directory, pdf_file)
            success = self.process_document(file_path)
            results[pdf_file] = success

        return results


def main():
    """Main function for testing multimodal indexing"""
    import sys

    # Configure indexer
    indexer = MultimodalIndexer(
        persist_directory="data/vectordb/multimodal/chroma/",
        collection_name="docs_mm",
        embedding_model="text-embedding-ada-002",
        enable_ocr=True
    )

    # Process a single file or directory
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            success = indexer.process_document(path)
            print(f"Processing {'successful' if success else 'failed'}")
        elif os.path.isdir(path):
            results = indexer.process_directory(path)
            print("\nProcessing Results:")
            for file, success in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {file}")
    else:
        print("Usage: python index_multimodal.py <pdf_file_or_directory>")


if __name__ == "__main__":
    main()