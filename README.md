# RAG\_v4

Retrieval-Augmented Generation (RAG) system for a knowledge management chatbot to search and answer questions in documents with Azure integration.

## What’s new vs RAG\_v3

- Two deployment paths: Azure Web App or Hybrid (Streamlit Cloud + Azure Blob)
- Ingestion writes a per‑file `original_url` into every chunk’s metadata so “View” opens the real Blob file

## Folder Structure

```
RAG_v4/
├── streamlit_app.py                 # Main Streamlit app (renders Sources with real blob URLs)
├── azure_blob_manager.py            # Blob helpers (upload, URL/SAS)
├── create_vectordb_from_azure.py    # Builds/refreshes vector DB from Azure Blob
├── upload_data_with_azure.py        # Upload local docs to Blob, then index
├── upload_to_azure_only.py          # Upload local docs to Blob (no indexing)
├── upload_data_manually.py          # Simple/local uploader
├── appsettings.json                 # App/service config for Azure
├── configs/
│   └── app_config.yml
├── hybrid_rag/                      # Hybrid retrieval (dense + BM25, reranker)
│   ├── hybrid_retrieval.py          # Dense+BM25 fusion retriever
│   ├── dense_retriever.py           # Embedding-based retrieval helpers
│   ├── bm25_index.py                # Keyword/BM25 index builder
│   ├── reranker.py                  # Cross-encoder / fusion reranker
│   └── eval.py                      # Retrieval diagnostics (optional)
├── utils/                           # Shared utilities
├── data/
│   └── cloud_test_docs/             # Sample docs for testing
├── vectordb/                        # Local Chroma (or chosen store)
├── .deployment/                     # Azure deployment assets
├── requirements.txt
├── .env.example
└── README.md
```

## Quickstart

```
# 1) Clone
git clone https://github.com/izoaraman/RAG_v4.git
cd RAG_v4

# 2) Create env & install (choose one requirements set)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt    

# 3) Configure
cp .env.example .env
# Fill in keys, storage connection string, container name

# 4) Run the app
streamlit run streamlit_app.py
```

## Minimal Configuration (env)

Use either OpenAI or Azure OpenAI, plus Azure Blob:

```
# OpenAI (choose this OR the Azure block)
OPENAI_API_KEY=...

# Azure OpenAI (alternative)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT_NAME=...
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=...

# Azure Blob Storage (required)
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_BLOB_CONTAINER=documents
# If using public container links
BLOB_PUBLIC_READ=true

```



## Run Locally

```
streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Deploy – Option A: Azure Web App

1. Create App Service (Linux) with a Python stack compatible with `requirements.txt`.
2. In Configuration → Application settings, add the env vars from **Minimal Configuration**.
3. Startup command example:
   ```
   python -m pip install -r requirements.txt && streamlit run app/streamlit_app.py --server.port 8000 --server.address 0.0.0.0
   ```
4. Confirm the app renders Sources with working “View document” links.

## Deploy – Option B: Hybrid (Streamlit Cloud + Azure Blob)

1. Connect this repo in Streamlit Cloud.
2. Add the same variables in Secrets.
3. Run `ingest.py` (locally or once in Cloud) so the vector DB contains `original_url`.
4. Launch the app and verify “View document” opens the Blob file directly.
