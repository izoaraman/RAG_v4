#!/usr/bin/env python3
"""Upload all documents to Azure Blob Storage without creating embeddings."""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from azure_blob_manager import AzureBlobManager

def upload_all_documents(data_directory: str, output_metadata_file: str = "vectordb/azure_blob_metadata.json"):
    """
    Upload all documents in the data directory to Azure Blob Storage.

    Args:
        data_directory: Directory containing documents to upload
        output_metadata_file: Path to save the metadata JSON file
    """
    print("Starting bulk document upload to Azure Blob Storage...")
    print("=" * 60)

    # Initialize Azure Blob Manager
    try:
        blob_manager = AzureBlobManager()
        print("Azure Blob Storage initialized successfully")
    except Exception as e:
        print(f"Error: Could not initialize Azure Blob Storage: {e}")
        return False

    # Find all PDF documents
    data_path = Path(data_directory)
    pdf_files = list(data_path.glob('*.pdf'))

    print(f"Found {len(pdf_files)} PDF documents to upload")

    if not pdf_files:
        print("No PDF files found in the directory.")
        return False

    # Track uploaded documents
    uploaded_docs_metadata = []
    failed_uploads = []

    # Upload each document
    for i, file_path in enumerate(pdf_files, 1):
        try:
            print(f"[{i}/{len(pdf_files)}] Uploading {file_path.name}...")
            blob_info = blob_manager.upload_document(str(file_path))
            uploaded_docs_metadata.append(blob_info)
            print(f"  SUCCESS: {blob_info['blob_name']}")
        except Exception as e:
            error_info = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'error': str(e)
            }
            failed_uploads.append(error_info)
            print(f"  FAILED: {e}")

        # Progress indicator
        if i % 50 == 0:
            print(f"\nProgress: {i}/{len(pdf_files)} documents processed")
            print(f"Successful: {len(uploaded_docs_metadata)}, Failed: {len(failed_uploads)}")
            print("-" * 40)

    # Save metadata
    metadata_path = Path(output_metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        'uploaded_documents': uploaded_docs_metadata,
        'failed_uploads': failed_uploads,
        'total_attempted': len(pdf_files),
        'successful_uploads': len(uploaded_docs_metadata),
        'failed_uploads_count': len(failed_uploads),
        'use_azure_blob': True,
        'container_name': blob_manager.container_name,
        'upload_timestamp': str(Path().cwd()),
        'data_directory': data_directory
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"Total documents processed: {len(pdf_files)}")
    print(f"Successfully uploaded: {len(uploaded_docs_metadata)}")
    print(f"Failed uploads: {len(failed_uploads)}")
    print(f"Success rate: {(len(uploaded_docs_metadata) / len(pdf_files) * 100):.1f}%")
    print(f"Metadata saved to: {metadata_path}")

    if failed_uploads:
        print("\nFailed uploads:")
        for fail in failed_uploads:
            print(f"  - {fail['file_name']}: {fail['error']}")

    if uploaded_docs_metadata:
        print(f"\nSample uploaded URLs:")
        for doc in uploaded_docs_metadata[:3]:
            print(f"  - {doc['file_name']}: {doc['public_url']}")
        if len(uploaded_docs_metadata) > 3:
            print(f"  ... and {len(uploaded_docs_metadata) - 3} more")

    return len(failed_uploads) == 0  # Return True if all uploads succeeded

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Upload all documents to Azure Blob Storage")
    parser.add_argument("--data-dir", default="data/docs", help="Directory containing documents")
    parser.add_argument("--metadata-file", default="vectordb/azure_blob_metadata.json", help="Output metadata file")

    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return False

    # Start upload process
    success = upload_all_documents(args.data_dir, args.metadata_file)

    if success:
        print("\nAll documents uploaded successfully!")
        return True
    else:
        print("\nSome uploads failed. Check the summary above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)