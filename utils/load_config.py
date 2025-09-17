import openai
import os
from dotenv import load_dotenv
import yaml
import shutil
import tempfile

# Graceful import for langchain_openai
try:
    from langchain_openai import AzureOpenAIEmbeddings
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: langchain_openai not available in load_config.py")
    AzureOpenAIEmbeddings = None
    LANGCHAIN_OPENAI_AVAILABLE = False

# Graceful import for pyprojroot
try:
    from pyprojroot import here
    PYPROJROOT_AVAILABLE = True
except ImportError:
    print("Warning: pyprojroot not available")
    PYPROJROOT_AVAILABLE = False

load_dotenv()


class LoadConfig:
    """
    A class for loading configuration settings and managing directories.
    ...

    Methods:
        load_openai_cfg():
            Load Azure OpenAI configuration settings.
        create_directory(directory_path):
            Create a directory if it does not exist.
        remove_directory(directory_path):
            Removes the specified directory.
    """

    def __init__(self) -> None:
        # Initialize app_config with defaults
        app_config = {
            "llm_config": {"llm_system_role": "You are a helpful assistant."},
            "directories": {
                "persist_directory": "data/vectordb/processed",
                "custom_persist_directory": "data/vectordb/uploaded"
            }
        }
        
        # Find the config file based on environment
        config_path = None
        if PYPROJROOT_AVAILABLE:
            try:
                config_path = here("configs/app_config.yml")
            except Exception as e:
                print(f"Warning: pyprojroot here() failed: {e}")
        
        if not config_path:
            # Try multiple possible locations
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "app_config.yml"),
                os.path.join(os.getcwd(), "configs", "app_config.yml"),
                os.path.join("/home/user/app", "configs", "app_config.yml"),
                "configs/app_config.yml"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    print(f"Found config at: {config_path}")
                    break
            
            if not config_path:
                print(f"Warning: Could not find app_config.yml in any of: {possible_paths}")
                print("Using default configuration")
        
        # Load config from file if found
        if config_path and os.path.exists(config_path):
            with open(config_path) as cfg:
                app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # --- Populate Azure env vars from YAML when present (without overwriting explicit env) ---
        try:
            azure_cfg = app_config.get("llm_config", {}).get("azure", {}) or {}
            # Normalize and set only if not already set in real environment
            if isinstance(azure_cfg, dict):
                yaml_api_key = str(azure_cfg.get("api_key", "")).strip()
                yaml_endpoint = str(azure_cfg.get("endpoint", "")).strip().rstrip("/")
                yaml_api_version = str(azure_cfg.get("api_version", "")).strip()

                # Always set from YAML to avoid stale OS envs causing 401
                if yaml_api_key:
                    os.environ["AZURE_OPENAI_API_KEY"] = yaml_api_key
                if yaml_endpoint:
                    os.environ["AZURE_OPENAI_ENDPOINT"] = yaml_endpoint
                if yaml_api_version:
                    os.environ["AZURE_OPENAI_API_VERSION"] = yaml_api_version

            # Deployment names: use YAML engines as sensible defaults
            yaml_chat_deployment = (app_config.get("llm_config", {}) or {}).get("engine")
            yaml_embed_deployment = (app_config.get("embedding_model_config", {}) or {}).get("engine")
            if yaml_chat_deployment:
                os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = str(yaml_chat_deployment).strip()
            if yaml_embed_deployment:
                os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"] = str(yaml_embed_deployment).strip()
        except Exception as e:
            print(f"Warning: Failed to derive Azure env vars from YAML: {e}")

        # LLM configs
        # Prefer YAML-defined engine, fallback to env, then default
        try:
            yaml_engine = app_config.get("llm_config", {}).get("engine")
        except Exception:
            yaml_engine = None
        self.llm_engine = yaml_engine or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.llm_system_role = app_config["llm_config"]["llm_system_role"]
        
        # Directory configs - use here() if available, otherwise use relative paths
        if PYPROJROOT_AVAILABLE:
            base_persist_dir = str(here(app_config["directories"]["persist_directory"]))
        else:
            # Use absolute paths based on current working directory or known locations
            base_dir = os.environ.get("PROJECT_ROOT", os.getcwd())
            base_persist_dir = os.path.join(base_dir, app_config["directories"]["persist_directory"])  # needs to be string for summation in chromadb backend: self._settings.require("persist_directory") + "/chroma.sqlite3"
        
        # Handle custom_persist_directory - use temp dir on Streamlit Cloud
        if os.environ.get("STREAMLIT_CLOUD") == "true" or os.path.exists("/home/appuser"):
            # On Streamlit Cloud, use temporary directory for uploads
            temp_base = os.environ.get("TMPDIR", "/tmp")
            base_custom_dir = os.path.join(temp_base, "vectordb_uploads")
        elif PYPROJROOT_AVAILABLE:
            base_custom_dir = str(here(app_config["directories"]["custom_persist_directory"]))
        else:
            base_dir = os.environ.get("PROJECT_ROOT", os.getcwd())
            base_custom_dir = os.path.join(base_dir, app_config["directories"]["custom_persist_directory"])

        # Always use Azure embeddings - single consistent embedding model
        actual_engine = "text-embedding-ada-002"

        # Suffix persist directories by embedding engine to avoid dimension mismatches
        def _with_engine_suffix(path_str: str, engine_label: str) -> str:
            import re
            safe_engine = re.sub(r"[^A-Za-z0-9._-]+", "_", engine_label or "default")
            norm = os.path.normpath(path_str)

            # Check if the suffix is already present to avoid double suffixing
            if safe_engine in norm:
                return norm

            parent = os.path.dirname(norm)
            base = os.path.basename(norm)
            return os.path.join(parent, f"{base}_{safe_engine}")

        self.persist_directory = _with_engine_suffix(base_persist_dir, actual_engine)
        self.custom_persist_directory = _with_engine_suffix(base_custom_dir, actual_engine)
        
        # Azure OpenAI Embeddings (required)
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_available = bool(azure_key) and bool(azure_endpoint)
        
        if self.azure_available and LANGCHAIN_OPENAI_AVAILABLE:
            try:
                self.embedding_model = AzureOpenAIEmbeddings(
                    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"),
                    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key
                )
            except Exception as e:
                self.embedding_model = None
                print(f"Error: Azure embeddings initialization failed: {e}")
                print("Please check your Azure OpenAI credentials.")
        else:
            self.embedding_model = None
            if not self.azure_available:
                print("Warning: Azure OpenAI credentials not found. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.")
            elif not LANGCHAIN_OPENAI_AVAILABLE:
                print("Warning: langchain_openai not installed. Please install with: pip install langchain-openai")

        # Retrieval configs
        # Resolve data directory to absolute path and ensure it exists
        self.data_directory = app_config["directories"]["data_directory"]
        try:
            # Try multiple possible base directories
            possible_base_dirs = [
                os.environ.get("PROJECT_ROOT"),  # Environment variable
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Two levels up from utils/load_config.py
                os.getcwd(),  # Current working directory
                os.path.dirname(os.getcwd()),  # Parent of current directory
            ]
            
            # Find the correct base directory that contains the data folder
            base_dir = None
            for possible_dir in possible_base_dirs:
                if possible_dir and os.path.exists(os.path.join(possible_dir, self.data_directory)):
                    base_dir = possible_dir
                    break
            
            # If not found, try to find RAG-Chatbot directory
            if base_dir is None:
                for possible_dir in possible_base_dirs:
                    if possible_dir:
                        # Check if RAG-Chatbot is in the path
                        if "RAG-Chatbot" in possible_dir:
                            # Get the RAG-Chatbot directory
                            parts = possible_dir.split(os.sep)
                            rag_idx = parts.index("RAG-Chatbot")
                            base_dir = os.sep.join(parts[:rag_idx+1])
                            if os.path.exists(os.path.join(base_dir, self.data_directory)):
                                break
            
            # Use the found base_dir or fallback to cwd
            if base_dir is None:
                base_dir = os.getcwd()
                print(f"Warning: Could not find data directory, using cwd: {base_dir}")
            
            if not os.path.isabs(self.data_directory):
                self.data_directory = os.path.join(base_dir, self.data_directory)
                
            # Normalize the path
            self.data_directory = os.path.normpath(self.data_directory)
            
            # Only print in debug mode
            if os.environ.get("DEBUG_MODE"):
                print(f"Data directory resolved to: {self.data_directory}")
                print(f"Data directory exists: {os.path.exists(self.data_directory)}")
            
            if not os.path.exists(self.data_directory):
                os.makedirs(self.data_directory, exist_ok=True)
                print(f"Created data directory: {self.data_directory}")
        except Exception as e:
            print(f"Warning: Could not prepare data directory {self.data_directory}: {e}")
        self.k = app_config["retrieval_config"]["k"]
        # Always use Azure embedding engine
        self.embedding_model_engine = actual_engine
        self.chunk_size = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]

        # Summarizer config
        self.max_final_token = app_config["summarizer_config"]["max_final_token"]
        self.token_threshold = app_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.character_overlap = app_config["summarizer_config"]["character_overlap"]
        self.final_summarizer_llm_system_role = app_config[
            "summarizer_config"]["final_summarizer_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]

        # Memory
        self.number_of_q_a_pairs = app_config["memory"]["number_of_q_a_pairs"]
        
        # Hybrid RAG config
        self.hybrid_rag = app_config.get("hybrid_rag", {
            "enabled": False,
            "multimodal": {"enabled": False},
            "graph": {"enabled": False},
            "router": {"debug": False}
        })
        
        # Store full app_config for components that need it
        self.app_config = app_config

        # Load Azure OpenAI credentials
        self.load_openai_cfg()

        # Ensure directories exist
        self.create_directory(self.persist_directory)
        self.create_directory(self.custom_persist_directory)
        # Don't remove the custom directory - it stores important hash tracking

    def load_openai_cfg(self):
        """
        Load Azure OpenAI configuration settings.
        """
        # Configure Azure OpenAI only when credentials are available
        if self.azure_available:
            openai.api_type = "azure"
            openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
            openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
            openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        else:
            # Skip configuring OpenAI; running in non-Azure/offline mode
            pass

    def create_directory(self, directory_path: str):
        """
        Create a directory if it does not exist.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def remove_directory(self, directory_path: str):
        """
        Removes the specified directory.
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
