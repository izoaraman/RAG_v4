"""Centralized secrets management for Streamlit deployment."""

import os
import streamlit as st

def get_secret(key: str, default=None):
    """
    Get secret from Streamlit secrets or environment variables.
    Priority: Streamlit secrets > Environment variables > Default
    """
    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    
    # Fall back to environment variable
    env_value = os.environ.get(key)
    if env_value:
        return env_value
    
    # Return default if provided
    return default

def get_openai_config():
    """Get OpenAI configuration from secrets."""
    return {
        "api_key": get_secret("OPENAI_API_KEY"),
        "azure_api_key": get_secret("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": get_secret("AZURE_OPENAI_ENDPOINT"),
        "azure_deployment": get_secret("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "azure_api_version": get_secret("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    }

def get_auth_config():
    """Get authentication configuration from secrets."""
    return {
        "username": get_secret("AUTH_USERNAME", "test_user"),
        "password": get_secret("AUTH_PASSWORD", "sdau2025")
    }

def set_environment_from_secrets():
    """Set environment variables from Streamlit secrets for compatibility."""
    secrets_to_env = [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_API_VERSION",
        "LOCAL_EMBEDDINGS_MODEL"
    ]
    
    for key in secrets_to_env:
        value = get_secret(key)
        if value and not os.environ.get(key):
            os.environ[key] = value