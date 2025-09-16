"""
Enhanced Document Loader with Advanced Table and Image Extraction
Provides improved table detection, image-in-table detection, and structured extraction
"""

import os
import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from pathlib import Path

# PDF processing
import fitz  # PyMuPDF
import tabula
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document

# Azure OCR support
try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    AZURE_OCR_AVAILABLE = True
except ImportError:
    AZURE_OCR_AVAILABLE = False
    logging.warning("Azure Computer Vision not available. OCR features limited.")

# Local OCR fallback
try:
    import pytesseract
    from PIL import Image
    LOCAL_OCR_AVAILABLE = True
except ImportError:
    LOCAL_OCR_AVAILABLE = False
    logging.warning("Pytesseract not available. OCR features disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDocumentLoader:
    """Advanced document loader with table and image extraction capabilities"""
    
    def __init__(
        self,
        enable_ocr: bool = True,
        enable_table_detection: bool = True,
        enable_file_hashing: bool = True,
        hash_directory: str = "data/vectordb/",
        use_azure_ocr: bool = True,
        ocr_threshold: int = 200
    ):
        """
        Initialize enhanced document loader
        
        Args:
            enable_ocr: Enable OCR for scanned documents
            enable_table_detection: Enable advanced table detection
            enable_file_hashing: Enable file deduplication via hashing
            hash_directory: Directory to store file hashes
            use_azure_ocr: Prefer Azure OCR over local OCR
            ocr_threshold: Minimum characters before triggering OCR
        """
        self.enable_ocr = enable_ocr
        self.enable_table_detection = enable_table_detection
        self.enable_file_hashing = enable_file_hashing
        self.hash_directory = hash_directory
        self.use_azure_ocr = use_azure_ocr and AZURE_OCR_AVAILABLE
        self.ocr_threshold = ocr_threshold
        
        # Ensure hash directory exists
        if enable_file_hashing:
            os.makedirs(hash_directory, exist_ok=True)
    
    # ========== File Hashing & Deduplication ==========
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_file_hashes(self) -> Dict[str, str]:
        """Load existing file hashes from metadata"""
        metadata_file = os.path.join(self.hash_directory, "file_hashes.json")
        if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted file_hashes.json, starting fresh")
        return {}
    
    def update_file_hashes(self, files: List[str]) -> Dict[str, str]:
        """Update file hashes metadata"""
        metadata_file = os.path.join(self.hash_directory, "file_hashes.json")
        existing_hashes = self.get_file_hashes()
        
        updated = {}
        for file_path in files:
            abs_path = os.path.abspath(file_path) if not os.path.isabs(file_path) else file_path
            file_hash = self.calculate_file_hash(abs_path)
            
            # Check if this hash already exists (duplicate file)
            if file_hash not in existing_hashes.values():
                existing_hashes[abs_path] = file_hash
                updated[abs_path] = file_hash
            else:
                logger.info(f"File already indexed: {os.path.basename(abs_path)}")
        
        # Save updated hashes
        with open(metadata_file, "w") as f:
            json.dump(existing_hashes, f, indent=2)
        
        return updated
    
    def is_file_processed(self, file_path: str) -> bool:
        """Check if file has already been processed"""
        if not self.enable_file_hashing:
            return False
        
        abs_path = os.path.abspath(file_path) if not os.path.isabs(file_path) else file_path
        file_hash = self.calculate_file_hash(abs_path)
        existing_hashes = self.get_file_hashes()
        
        return file_hash in existing_hashes.values()
    
    # ========== Table Detection & Extraction ==========
    
    def _check_overlap(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Check if two bounding boxes overlap"""
        # bbox format: (x0, y0, x1, y1)
        return not (
            bbox1[2] <= bbox2[0] or  # bbox1 right <= bbox2 left
            bbox1[0] >= bbox2[2] or  # bbox1 left >= bbox2 right
            bbox1[3] <= bbox2[1] or  # bbox1 bottom <= bbox2 top
            bbox1[1] >= bbox2[3]     # bbox1 top >= bbox2 bottom
        )
    
    def find_images_in_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Detect tables and identify which cells contain images
        
        Returns:
            List of table dictionaries with content and metadata
        """
        pdf_file = fitz.open(pdf_path)
        all_tables = []
        
        try:
            # Process each page
            for page_num in range(len(pdf_file)):
                # Get image bounding boxes on the page
                page = pdf_file.load_page(page_num)
                images = page.get_image_info()
                image_bboxes = [img["bbox"] for img in images]
                
                # Extract tables using tabula
                try:
                    tables = tabula.read_pdf(
                        pdf_path,
                        pages=page_num + 1,  # tabula uses 1-based page numbers
                        multiple_tables=True,
                        encoding="utf-8",
                        output_format="json",
                        silent=True
                    )
                except Exception as e:
                    logger.warning(f"Tabula failed for page {page_num + 1}: {e}")
                    tables = []
                
                if not tables:
                    continue
                
                # Process each table found
                for table_idx, table in enumerate(tables):
                    # Extract column positions
                    left_edges = []
                    for row in table.get("data", []):
                        for cell in row:
                            if cell.get("left", 0) > 0 and cell["left"] not in left_edges:
                                left_edges.append(cell["left"])
                    left_edges.sort()
                    
                    # Build structured table data
                    table_data = {
                        "page": page_num,
                        "page_label": page_num + 1,
                        "table_index": table_idx,
                        "total_pages": len(pdf_file),
                        "content": [],
                        "has_images": False
                    }
                    
                    # Process each row
                    for row in table.get("data", []):
                        row_entries = []
                        
                        # Align cells to column positions
                        for edge in left_edges:
                            cell_found = False
                            
                            for cell in row:
                                if cell.get("left") == edge:
                                    cell_text = cell.get("text", "").strip()
                                    
                                    # Check if cell is empty and overlaps with image
                                    if not cell_text:
                                        cell_bbox = (
                                            cell.get("left", 0),
                                            cell.get("top", 0),
                                            cell.get("left", 0) + cell.get("width", 0),
                                            cell.get("top", 0) + cell.get("height", 0)
                                        )
                                        
                                        # Check for image overlap
                                        image_found = False
                                        for img_bbox in image_bboxes:
                                            if self._check_overlap(cell_bbox, img_bbox):
                                                row_entries.append("[IMAGE]")
                                                table_data["has_images"] = True
                                                image_found = True
                                                break
                                        
                                        if not image_found:
                                            row_entries.append("")
                                    else:
                                        # Clean up text encoding
                                        clean_text = re.sub(r'\s+', ' ', cell_text)
                                        row_entries.append(clean_text)
                                    
                                    cell_found = True
                                    break
                                elif cell.get("left", 0) > edge:
                                    break
                            
                            if not cell_found:
                                row_entries.append(None)
                        
                        table_data["content"].append(row_entries)
                    
                    if table_data["content"]:
                        all_tables.append(table_data)
                        
        finally:
            pdf_file.close()
        
        return all_tables
    
    def table_to_structured_json(self, tables: List[Dict[str, Any]], source_path: str) -> List[Document]:
        """
        Convert table data to structured JSON format for better retrieval
        
        Returns:
            List of Document objects with table content as JSON
        """
        documents = []
        
        for table in tables:
            if not table.get("content") or len(table["content"]) == 0:
                continue
            
            # Treat first row as headers if available
            headers = table["content"][0] if len(table["content"]) > 0 else []
            
            # Convert to list of row dictionaries
            table_json = []
            for row_idx, row in enumerate(table["content"][1:] if headers else table["content"]):
                if not row:
                    continue
                
                row_dict = {}
                for col_idx, cell_value in enumerate(row):
                    # Use header as key if available, otherwise use column index
                    if headers and col_idx < len(headers) and headers[col_idx]:
                        key = str(headers[col_idx])
                    else:
                        key = f"Column_{col_idx + 1}"
                    
                    # Handle None values
                    value = cell_value if cell_value is not None else ""
                    row_dict[key] = value
                
                if row_dict:  # Only add non-empty rows
                    table_json.append(row_dict)
            
            # Create structured content
            if table_json:
                # Determine modality based on content
                modality = "table_image" if table.get("has_images") else "table"
                
                # Format content for better readability
                content_lines = [
                    f"Table from page {table['page_label']}:",
                    json.dumps(table_json, indent=2, ensure_ascii=False)
                ]
                
                # Add summary if table has many rows
                if len(table_json) > 10:
                    content_lines.insert(1, f"(Table contains {len(table_json)} rows)")
                
                doc = Document(
                    page_content="\n".join(content_lines),
                    metadata={
                        "source": source_path,
                        "modality": modality,
                        "page": table["page"],
                        "page_label": table["page_label"],
                        "table_index": table.get("table_index", 0),
                        "has_images": table.get("has_images", False),
                        "row_count": len(table_json),
                        "column_count": len(headers) if headers else len(table["content"][0]) if table["content"] else 0,
                        "producer": "Tabula and PyMuPDF"
                    }
                )
                documents.append(doc)
        
        return documents
    
    # ========== OCR Support ==========
    
    def perform_azure_ocr(self, pdf_path: str, page_num: Optional[int] = None) -> List[Document]:
        """
        Perform OCR using Azure Computer Vision
        
        Args:
            pdf_path: Path to PDF file
            page_num: Specific page to OCR (None for all pages)
        
        Returns:
            List of Document objects with OCR text
        """
        if not self.use_azure_ocr or not AZURE_OCR_AVAILABLE:
            return []
        
        endpoint = os.getenv("AZURE_OCR_ENDPOINT")
        api_key = os.getenv("AZURE_OCR_API_KEY")
        
        if not endpoint or not api_key:
            logger.warning("Azure OCR credentials not configured")
            return self.perform_local_ocr(pdf_path, page_num)
        
        try:
            client = ComputerVisionClient(
                endpoint, 
                CognitiveServicesCredentials(api_key)
            )
            
            documents = []
            
            with open(pdf_path, "rb") as file:
                # Submit for OCR
                read_response = client.read_in_stream(file, raw=True)
                operation_location = read_response.headers["Operation-Location"]
                operation_id = operation_location.split("/")[-1]
                
                # Poll for results
                while True:
                    result = client.get_read_result(operation_id)
                    if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                        break
                    time.sleep(1)
                
                # Extract text from results
                if result.status == OperationStatusCodes.succeeded:
                    for page_idx, page in enumerate(result.analyze_result.read_results):
                        # Skip if specific page requested and this isn't it
                        if page_num is not None and page_idx != page_num:
                            continue
                        
                        lines = [line.text for line in page.lines]
                        page_text = "\n".join(lines)
                        
                        if page_text.strip():
                            doc = Document(
                                page_content=page_text,
                                metadata={
                                    "source": pdf_path,
                                    "modality": "ocr",
                                    "ocr_engine": "azure",
                                    "page": page_idx,
                                    "page_label": page_idx + 1,
                                    "producer": "Azure Computer Vision"
                                }
                            )
                            documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Azure OCR failed: {e}")
            return self.perform_local_ocr(pdf_path, page_num)
    
    def perform_local_ocr(self, pdf_path: str, page_num: Optional[int] = None) -> List[Document]:
        """
        Perform OCR using local Tesseract
        
        Args:
            pdf_path: Path to PDF file
            page_num: Specific page to OCR (None for all pages)
        
        Returns:
            List of Document objects with OCR text
        """
        if not LOCAL_OCR_AVAILABLE:
            logger.warning("Local OCR not available")
            return []
        
        documents = []
        
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = [page_num] if page_num is not None else range(len(pdf.pages))
                
                for page_idx in pages_to_process:
                    if page_idx >= len(pdf.pages):
                        continue
                    
                    page = pdf.pages[page_idx]
                    
                    # Convert page to image
                    im = page.to_image(resolution=150)
                    pil_image = im.original
                    
                    # Perform OCR
                    ocr_text = pytesseract.image_to_string(pil_image)
                    
                    # Clean up text
                    ocr_text = ' '.join(ocr_text.split())
                    
                    if ocr_text.strip():
                        doc = Document(
                            page_content=ocr_text,
                            metadata={
                                "source": pdf_path,
                                "modality": "ocr",
                                "ocr_engine": "tesseract",
                                "page": page_idx,
                                "page_label": page_idx + 1,
                                "producer": "Tesseract OCR"
                            }
                        )
                        documents.append(doc)
        
        except Exception as e:
            logger.error(f"Local OCR failed: {e}")
        
        return documents
    
    # ========== Main Loading Functions ==========
    
    def load_pdf_with_tables(self, pdf_path: str) -> Tuple[List[Document], List[Document]]:
        """
        Load PDF with enhanced table extraction
        
        Returns:
            Tuple of (text_documents, table_documents)
        """
        text_docs = []
        table_docs = []
        
        # Check if file already processed
        if self.enable_file_hashing and self.is_file_processed(pdf_path):
            logger.info(f"File already processed: {pdf_path}")
            return text_docs, table_docs
        
        try:
            # Load text content using standard loader
            loader = PyPDFLoader(file_path=pdf_path, extract_images=False)
            text_docs = loader.load()
            
            # Check if OCR needed
            total_text_length = sum(len(doc.page_content) for doc in text_docs)
            
            if self.enable_ocr and total_text_length < self.ocr_threshold:
                logger.info(f"Low text content detected ({total_text_length} chars), performing OCR")
                ocr_docs = self.perform_azure_ocr(pdf_path) if self.use_azure_ocr else self.perform_local_ocr(pdf_path)
                text_docs.extend(ocr_docs)
            
            # Extract tables if enabled
            if self.enable_table_detection:
                logger.info(f"Extracting tables from {pdf_path}")
                raw_tables = self.find_images_in_tables(pdf_path)
                table_docs = self.table_to_structured_json(raw_tables, pdf_path)
                logger.info(f"Extracted {len(table_docs)} tables")
            
            # Update file hashes if enabled
            if self.enable_file_hashing:
                self.update_file_hashes([pdf_path])
        
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
        
        return text_docs, table_docs
    
    def load_documents(
        self,
        file_paths: List[str],
        include_tables: bool = True,
        include_ocr: bool = True
    ) -> Dict[str, List[Document]]:
        """
        Load multiple documents with enhanced extraction
        
        Args:
            file_paths: List of document paths to process
            include_tables: Include table extraction
            include_ocr: Include OCR for scanned pages
        
        Returns:
            Dictionary with 'text' and 'table' document lists
        """
        all_text_docs = []
        all_table_docs = []
        
        # Temporarily update settings
        original_ocr = self.enable_ocr
        original_tables = self.enable_table_detection
        
        self.enable_ocr = include_ocr
        self.enable_table_detection = include_tables
        
        try:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                # Only process PDFs for now
                if file_path.lower().endswith('.pdf'):
                    text_docs, table_docs = self.load_pdf_with_tables(file_path)
                    all_text_docs.extend(text_docs)
                    all_table_docs.extend(table_docs)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
        
        finally:
            # Restore original settings
            self.enable_ocr = original_ocr
            self.enable_table_detection = original_tables
        
        return {
            "text": all_text_docs,
            "table": all_table_docs
        }