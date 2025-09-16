#!/usr/bin/env python3
"""
Azure App Service startup script - Most reliable method
This ensures Streamlit runs on Azure by handling all edge cases
"""
import subprocess
import sys
import os
import time

print("=== RAG v3 Azure Startup Script ===", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Working directory: {os.getcwd()}", flush=True)

# Ensure we're in the right directory
app_dir = "/home/site/wwwroot"
if os.path.exists(app_dir):
    os.chdir(app_dir)
    print(f"Changed to {app_dir}", flush=True)

# Install critical dependencies first
critical_packages = [
    "streamlit==1.28.2",
    "openai==1.10.0"
]

print("Installing critical packages...", flush=True)
for package in critical_packages:
    print(f"Installing {package}...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                   check=False, capture_output=False)

# Try to install from requirements.txt if it exists
req_file = "requirements.txt"
if os.path.exists(req_file):
    print(f"Installing from {req_file}...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], 
                   check=False, capture_output=False, timeout=300)
else:
    print(f"Warning: {req_file} not found", flush=True)

# Verify streamlit is installed
try:
    import streamlit
    print(f"✓ Streamlit {streamlit.__version__} installed", flush=True)
except ImportError:
    print("ERROR: Streamlit not installed, attempting recovery...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "streamlit==1.28.2"], 
                   check=True)

# Start Streamlit - FORCE PORT 8000
port = "8000"  # Force port 8000, ignore environment variable
print(f"Starting Streamlit on port {port} (forced to 8000)...", flush=True)

# Ensure streamlit_app.py exists
if not os.path.exists("streamlit_app.py"):
    print("ERROR: streamlit_app.py not found!", flush=True)
    print("Files in current directory:", flush=True)
    for f in os.listdir("."):
        print(f"  - {f}", flush=True)
    sys.exit(1)

# Run Streamlit with proper arguments
cmd = [
    sys.executable, "-m", "streamlit", "run",
    "streamlit_app.py",
    "--server.port", port,
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
]

print(f"Executing: {' '.join(cmd)}", flush=True)
subprocess.run(cmd)