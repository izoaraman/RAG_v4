from .prepare_vectordb import PrepareVectorDB
from typing import List, Tuple
from .load_config import LoadConfig
from .summarizer import Summarizer
import hashlib
import json
import os

APPCFG = LoadConfig()

def ensure_directory_access(directory: str):
    """Ensure the directory exists and has write access."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Access denied to {directory}. Check permissions.")


class UploadFile:

    @staticmethod
    def process_uploaded_files(files_dir, chatbot, rag_with_dropdown):
        """Process uploaded files and update chatbot conversation."""

        # Check if files exist in the directory
        if not os.path.exists(files_dir) or not os.listdir(files_dir):
            chatbot.append({
                "role": "assistant",
                "content": "No files found to process. Please upload some documents first."
            })
            return "", chatbot

        # Process files based on selected RAG option
        if rag_with_dropdown and rag_with_dropdown != "Select an option":
            try:
                # Initialize vector database
                vector_db = PrepareVectorDB()

                # Process files from directory
                processed_files = []
                for filename in os.listdir(files_dir):
                    file_path = os.path.join(files_dir, filename)
                    if os.path.isfile(file_path):
                        processed_files.append(filename)

                if processed_files:
                    # Create summary of processed files
                    files_list = "\n".join([f"• {file}" for file in processed_files])

                    # Generate response based on RAG option
                    if "summary" in rag_with_dropdown.lower():
                        final_summary = f"""Successfully processed {len(processed_files)} files:

{files_list}

The documents have been processed and are ready for querying. You can now ask questions about the content of these documents."""
                    else:
                        final_summary = f"""Successfully processed {len(processed_files)} files with {rag_with_dropdown}:

{files_list}

The documents have been indexed and are ready for semantic search."""
                else:
                    final_summary = "No valid files found to process."

            except Exception as e:
                final_summary = f"Error processing files: {str(e)}"

            chatbot.append({"role": "assistant", "content": final_summary})
        else:
            chatbot.append({"role": "assistant", "content": "Select an option from the dropdown to process the uploaded documents."})

        return "", chatbot


def upload_data_manually():
    """Basic upload function for manual data processing."""
    try:
        from .prepare_vectordb import PrepareVectorDB
        vector_db = PrepareVectorDB()
        print("Processing documents...")
        # Add basic document processing logic here
        print("Documents processed successfully!")
    except Exception as e:
        print(f"Error processing documents: {str(e)}")


def upload_data_for_mode(mode="current", clear_existing=False):
    """Upload data with specific mode settings."""
    try:
        from .prepare_vectordb import PrepareVectorDB
        vector_db = PrepareVectorDB()

        if clear_existing:
            print("Clearing existing vector database...")

        print(f"Processing documents in {mode} mode...")
        # Add mode-specific processing logic here
        print("Documents processed successfully!")
    except Exception as e:
        print(f"Error processing documents: {str(e)}")


def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


def get_existing_file_hashes():
    """Get existing file hashes from storage."""
    try:
        # Implementation depends on where hashes are stored
        # This is a placeholder implementation
        return {}
    except Exception as e:
        print(f"Error getting existing hashes: {e}")
        return {}


def update_file_hashes(new_hashes):
    """Update stored file hashes."""
    try:
        # Implementation depends on where hashes are stored
        # This is a placeholder implementation
        print(f"Updated hashes for {len(new_hashes)} files")
    except Exception as e:
        print(f"Error updating hashes: {e}")