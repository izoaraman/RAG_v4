"""Azure Blob Storage manager for document uploads and public URL generation."""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from azure.storage.blob import BlobServiceClient, BlobClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
import hashlib
from pathlib import Path

class AzureBlobManager:
    """Manages document uploads to Azure Blob Storage and generates public URLs."""
    
    def __init__(self, connection_string: Optional[str] = None, container_name: str = "documents"):
        """
        Initialize Azure Blob Storage manager.
        
        Args:
            connection_string: Azure Storage connection string (uses env var if not provided)
            container_name: Name of the container to use for documents
        """
        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("Azure Storage connection string not provided. Set AZURE_STORAGE_CONNECTION_STRING env var.")
        
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        
        # Create container if it doesn't exist
        self._ensure_container_exists()
    
    def _ensure_container_exists(self):
        """Create container if it doesn't exist."""
        try:
            # Try with public access first (if allowed)
            try:
                container_client = self.blob_service_client.create_container(
                    self.container_name,
                    public_access="blob"  # Allow public read access to blobs
                )
                print(f"Created container with public access: {self.container_name}")
            except:
                # If public access not allowed, create without it
                container_client = self.blob_service_client.create_container(
                    self.container_name
                )
                print(f"Created container (private): {self.container_name}")
        except ResourceExistsError:
            print(f"Container already exists: {self.container_name}")
    
    def upload_document(self, file_path: str, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Upload a document to Azure Blob Storage.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata to attach to the blob
            
        Returns:
            Dict containing blob_name, public_url, and sas_url
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate unique blob name using hash and timestamp
        file_hash = self._calculate_file_hash(str(file_path))
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        blob_name = f"{timestamp}_{file_hash[:8]}_{file_path.name}"
        
        # Upload to blob storage
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        with open(file_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                metadata=metadata or {}
            )
        
        # Generate URLs
        public_url = blob_client.url
        sas_url = self.generate_sas_url(blob_name, expiry_days=365)
        
        return {
            "blob_name": blob_name,
            "public_url": public_url,
            "sas_url": sas_url,
            "file_name": file_path.name,
            "file_path": str(file_path)
        }
    
    def generate_sas_url(self, blob_name: str, expiry_days: int = 365) -> str:
        """
        Generate a SAS URL for a blob with read permissions.
        
        Args:
            blob_name: Name of the blob in the container
            expiry_days: Number of days until the SAS token expires
            
        Returns:
            SAS URL for the blob
        """
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        sas_token = generate_blob_sas(
            account_name=blob_client.account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self._get_account_key(),
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(days=expiry_days)
        )
        
        return f"{blob_client.url}?{sas_token}"
    
    def _get_account_key(self) -> str:
        """Extract account key from connection string."""
        parts = self.connection_string.split(';')
        for part in parts:
            if part.startswith('AccountKey='):
                return part.split('=', 1)[1]
        raise ValueError("Account key not found in connection string")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def download_blob(self, blob_name: str, download_path: str) -> str:
        """
        Download a blob from Azure Storage.
        
        Args:
            blob_name: Name of the blob to download
            download_path: Local path to save the file
            
        Returns:
            Path to the downloaded file
        """
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        with open(download_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        return download_path
    
    def list_blobs(self) -> list:
        """
        List all blobs in the container.
        
        Returns:
            List of blob names
        """
        container_client = self.blob_service_client.get_container_client(self.container_name)
        return [blob.name for blob in container_client.list_blobs()]
    
    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from the container.
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.delete_blob()
            return True
        except ResourceNotFoundError:
            return False