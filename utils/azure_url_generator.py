"""Generate Azure Blob Storage URLs for documents."""

import os
from typing import Optional
from urllib.parse import quote

def get_azure_blob_url(filename: str, container: str = "documents") -> str:
    """
    Generate Azure Blob Storage URL for a document.
    
    Args:
        filename: Name of the file
        container: Azure container name (default: "documents")
    
    Returns:
        Azure Blob Storage URL for the document
    """
    # Get Azure storage account name from connection string
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    
    # Extract account name from connection string
    account_name = "sandbox3190080146"  # Default from your connection string
    if conn_str:
        for part in conn_str.split(";"):
            if "AccountName=" in part:
                account_name = part.split("=")[1]
                break
    
    # Clean up filename - remove any timestamp prefixes or hash suffixes
    clean_filename = filename
    
    # Remove timestamp prefix (e.g., "20250908_022717_05ab142c_")
    import re
    timestamp_pattern = r'^\d{8}_\d{6}_[a-f0-9]{8}_'
    clean_filename = re.sub(timestamp_pattern, '', clean_filename)
    
    # URL encode the filename to handle spaces and special characters
    encoded_filename = quote(clean_filename)
    
    # Generate Azure Blob Storage URL
    base_url = f"https://{account_name}.blob.core.windows.net"
    blob_url = f"{base_url}/{container}/{encoded_filename}"
    
    return blob_url

def format_source_with_azure_url(source_text: str) -> str:
    """
    Replace example.com URLs with actual Azure Blob Storage URLs in source citations.
    
    Args:
        source_text: Text containing source citations with example.com URLs
    
    Returns:
        Text with Azure Blob Storage URLs
    """
    import re
    
    # Pattern to match View Document links
    pattern = r'\[View Document\]\(http://example\.com/data/docs/([^)]+)\)'
    
    def replace_url(match):
        filename = match.group(1)
        azure_url = get_azure_blob_url(filename)
        return f'[View Document]({azure_url})'
    
    # Replace all example.com URLs with Azure URLs
    updated_text = re.sub(pattern, replace_url, source_text)
    
    return updated_text