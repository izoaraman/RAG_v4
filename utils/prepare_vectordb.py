from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredEmailLoader,
    TextLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)
try:
    from utils.docx_loader_fallback import SimpleDocxLoader
except ImportError:
    from docx_loader_fallback import SimpleDocxLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
import time
import openai
import logging

# Import hybrid RAG components
try:
    from hybrid_rag.index_multimodal import MultimodalIndexer
    from hybrid_rag.index_graph import GraphIndexer
    HYBRID_RAG_AVAILABLE = True
    logging.info("Hybrid RAG components loaded successfully")
except ImportError as e:
    HYBRID_RAG_AVAILABLE = False
    logging.warning(f"Hybrid RAG components not available: {e}")

# Graceful import for langchain_openai
try:
    from langchain_openai import AzureOpenAIEmbeddings
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: langchain_openai not available. Using fallback embedding.")
    AzureOpenAIEmbeddings = None
    LANGCHAIN_OPENAI_AVAILABLE = False

def ensure_directory_access(directory: str):
    """Ensure the directory exists and has write access."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Access denied to {directory}. Check permissions.")


class PrepareVectorDB:
    """
    A class for preparing and saving a VectorDB including loading documents, chunking them, and creating a VectorDB
    """

    def __init__(
            self,
            data_directory: str,
            persist_directory: str,
            embedding_model_engine: str,
            chunk_size: int,
            chunk_overlap: int
    ) -> None:

        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        """Other options: CharacterTextSplitter, TokenTextSplitter, etc."""
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        # Always use Azure embeddings
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_key and azure_endpoint and LANGCHAIN_OPENAI_AVAILABLE:
            self.embedding = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                azure_endpoint=(azure_endpoint or "").rstrip("/"),
                api_key=azure_key
            )
            print("Using Azure OpenAI embeddings")
        else:
            if not LANGCHAIN_OPENAI_AVAILABLE:
                raise RuntimeError(
                    "langchain_openai not installed. Please install with: pip install langchain-openai"
                )
            else:
                raise RuntimeError(
                    "Azure OpenAI credentials are required. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
                )

    def __load_document_by_type(self, file_path: str):
        """Load a document based on its file extension."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                return PyPDFLoader(file_path).load()
            elif file_ext in ['.doc', '.docx']:
                if file_ext == '.docx':
                    return SimpleDocxLoader(file_path).load()
                else:
                    print(f"WARNING: .doc files not supported. Please convert to .docx format.")
                    return []
            elif file_ext in ['.eml', '.msg']:
                return UnstructuredEmailLoader(file_path).load()
            elif file_ext == '.txt':
                return TextLoader(file_path, encoding='utf-8').load()
            elif file_ext == '.csv':
                return CSVLoader(file_path, encoding='utf-8').load()
            elif file_ext in ['.html', '.htm']:
                return UnstructuredHTMLLoader(file_path).load()
            else:
                print(f"Unsupported file type: {file_ext}. Skipping {file_path}")
                return []
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return []

    def __load_all_documents(self) -> List:

        doc_counter = 0
        if isinstance(self.data_directory, list):
            print("Loading the uploaded documents...")
            docs = []
            for doc_path in self.data_directory:
                loaded_docs = self.__load_document_by_type(doc_path)
                if loaded_docs:
                    docs.extend(loaded_docs)
                    doc_counter += 1
            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")
        else:
            print("Loading documents manually...")
            document_list = os.listdir(self.data_directory)
            docs = []
            supported_extensions = ('.pdf', '.doc', '.docx', '.txt', '.csv', '.eml', '.msg', '.html', '.htm')
            for doc_name in document_list:
                if doc_name.lower().endswith(supported_extensions):
                    file_path = os.path.join(self.data_directory, doc_name)
                    loaded_docs = self.__load_document_by_type(file_path)
                    if loaded_docs:
                        docs.extend(loaded_docs)
                        doc_counter += 1
            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")

        return docs

    def __chunk_documents(self, docs: List) -> List:

        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def prepare_and_save_vectordb(self, enable_hybrid_rag=True):
        ensure_directory_access(self.persist_directory)
        docs = self.__load_all_documents()
        chunked_documents = self.__chunk_documents(docs)
        print("Preparing vectordb...")

        # Run hybrid RAG indexing if available and enabled
        if enable_hybrid_rag and HYBRID_RAG_AVAILABLE:
            print("Running hybrid RAG indexing...")
            try:
                # Load config for hybrid RAG settings
                import yaml
                config_path = "configs/app_config.yml"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        hybrid_config = config.get('hybrid_rag', {})

                        # Run multimodal indexing if enabled
                        if hybrid_config.get('multimodal', {}).get('enabled', False):
                            print("Running multimodal indexing...")
                            mm_indexer = MultimodalIndexer(
                                persist_directory=hybrid_config['multimodal'].get('persist_directory', 'data/vectordb/multimodal/chroma/'),
                                collection_name=hybrid_config['multimodal'].get('collection_name', 'docs_mm')
                            )
                            # Process documents for multimodal indexing
                            if isinstance(self.data_directory, list):
                                for doc_path in self.data_directory:
                                    if doc_path.lower().endswith('.pdf'):
                                        mm_indexer.process_document(doc_path)
                            else:
                                pdf_files = [f for f in os.listdir(self.data_directory)
                                           if f.lower().endswith('.pdf')]
                                for pdf_file in pdf_files:
                                    file_path = os.path.join(self.data_directory, pdf_file)
                                    mm_indexer.process_document(file_path)
                            print("Multimodal indexing completed")

                        # Run graph indexing if enabled
                        if hybrid_config.get('graph', {}).get('enabled', False):
                            print("Running knowledge graph indexing...")
                            graph_indexer = GraphIndexer(
                                store_path=hybrid_config['graph'].get('store_path', 'data/kg/'),
                                use_azure_openai=hybrid_config['graph'].get('use_azure_openai', True)
                            )
                            # Process documents for graph indexing
                            for doc in docs:
                                if hasattr(doc, 'page_content'):
                                    doc_id = doc.metadata.get('source', 'unknown')
                                    graph_indexer.process_document(doc.page_content, doc_id)

                            # Build and save graph
                            graph_indexer.build_graph()
                            graph_indexer.detect_communities()
                            if hybrid_config['graph'].get('use_azure_openai', True):
                                graph_indexer.summarize_communities_azure()
                            else:
                                graph_indexer.summarize_communities_fallback()
                            graph_indexer.save_graph(hybrid_config['graph'].get('graph_name', 'knowledge_graph'))
                            print("Knowledge graph indexing completed")

            except Exception as e:
                print(f"Warning: Hybrid RAG indexing encountered an error: {e}")
                print("Continuing with standard vector indexing...")

        max_retries = 5
        retries = 0
        vectordb = None

        while retries < max_retries:
            try:
                vectordb = Chroma.from_documents(
                    documents=chunked_documents,
                    embedding=self.embedding,
                    persist_directory=self.persist_directory
                )
                break  # Success, exit the retry loop
            except Exception as e:
                # Check if the error message indicates a rate limit error (429)
                error_text = str(e)
                if "dimension" in error_text and "expecting embedding" in error_text:
                    # Dimension mismatch; reset directory and retry once
                    try:
                        import shutil
                        print("Embedding dimension mismatch detected. Resetting persist directory and retrying...")
                        if os.path.exists(self.persist_directory):
                            shutil.rmtree(self.persist_directory)
                        os.makedirs(self.persist_directory, exist_ok=True)
                        retries += 1
                        continue
                    except Exception as reset_err:
                        raise Exception(f"Failed to reset Chroma directory after dim mismatch: {reset_err}") from e
                # Detect Chroma schema mismatch (e.g., "no such table: tenants") and reset the directory once
                if "no such table: tenants" in error_text or "schema" in error_text.lower():
                    try:
                        import shutil
                        print("Chroma schema mismatch detected. Resetting persist directory and retrying...")
                        if os.path.exists(self.persist_directory):
                            shutil.rmtree(self.persist_directory)
                        os.makedirs(self.persist_directory, exist_ok=True)
                        retries += 1
                        continue
                    except Exception as reset_err:
                        raise Exception(f"Failed to reset Chroma directory: {reset_err}") from e
                if "401" in error_text or "Access denied" in error_text or "invalid subscription key" in error_text:
                    # Credentials invalid - no fallback
                    raise Exception(
                        "Azure OpenAI authentication failed. Please check your AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
                    ) from e
                if "429" in error_text:
                    retries += 1
                    wait_time = 60  # Wait for 60 seconds as recommended
                    print(f"RateLimitError encountered (attempt {retries}/{max_retries}): {e}")
                    print(f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # For other errors, raise the exception immediately.
                    raise e

        if vectordb is None:
            raise Exception("Failed to create vectordb after multiple retries due to rate limits.")

        print("VectorDB is created and saved.")
        print("Number of vectors in vectordb:", vectordb._collection.count(), "\n\n")
        return vectordb



    

