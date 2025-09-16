"""Generate Azure Blob Storage SAS URLs for documents."""

import os
from datetime import datetime, timedelta
from urllib.parse import quote
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

def get_azure_blob_sas_url(filename: str, container: str = "documents", expiry_hours: int = 24) -> str:
    """
    Generate Azure Blob Storage SAS URL for a document.
    
    Args:
        filename: Name of the file (may include timestamp prefix)
        container: Azure container name (default: "documents")
        expiry_hours: Hours until the SAS token expires (default: 24)
    
    Returns:
        Azure Blob Storage SAS URL for the document
    """
    try:
        # Get connection string
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        if not conn_str:
            # Fallback to public URL if no connection string
            return get_public_url_encoded(filename, container)
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        
        # Get account name and key from connection string
        account_name = None
        account_key = None
        for part in conn_str.split(";"):
            if "AccountName=" in part:
                account_name = part.split("=")[1]
            elif "AccountKey=" in part:
                account_key = part.split("=")[1]
        
        if not account_name or not account_key:
            return get_public_url_encoded(filename, container)
        
        # Clean up filename - remove timestamp prefix if instructed
        # But use the full filename for actual blob reference
        blob_name = filename
        
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
        )
        
        # Construct the full URL with SAS token
        # Properly encode the blob name for URL
        encoded_blob_name = quote(blob_name, safe='/')
        blob_url = f"https://{account_name}.blob.core.windows.net/{container}/{encoded_blob_name}?{sas_token}"
        
        return blob_url
        
    except Exception as e:
        print(f"Error generating SAS URL: {e}")
        # Fallback to encoded public URL
        return get_public_url_encoded(filename, container)

def get_public_url_encoded(filename: str, container: str = "documents") -> str:
    """
    Generate properly encoded public URL (fallback when SAS not available).
    
    Args:
        filename: Name of the file
        container: Azure container name
    
    Returns:
        Properly encoded Azure Blob Storage URL
    """
    # Get Azure storage account name
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    account_name = "sandbox3190080146"  # Default
    
    if conn_str:
        for part in conn_str.split(";"):
            if "AccountName=" in part:
                account_name = part.split("=")[1]
                break
    
    # Properly encode the filename to handle spaces and special characters
    encoded_filename = quote(filename, safe='/')
    
    # Generate Azure Blob Storage URL
    blob_url = f"https://{account_name}.blob.core.windows.net/{container}/{encoded_filename}"
    
    return blob_url

def clean_document_name(filename: str) -> str:
    """
    Remove timestamp and hash prefixes from document names.
    
    Args:
        filename: Original filename with possible prefix
    
    Returns:
        Clean filename without prefix
    """
    import re
    
    # Remove timestamp prefix pattern: YYYYMMDD_HHMMSS_HASH_
    timestamp_pattern = r'^\d{8}_\d{6}_[a-f0-9]{8}_'
    clean_name = re.sub(timestamp_pattern, '', filename)
    
    return clean_name