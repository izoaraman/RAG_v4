import time
import urllib.parse
import openai
import os
from .db_manager import db_manager
from typing import List, Tuple
import re
import ast
import html
import logging
import json
from .load_config import LoadConfig
from .vector_utils import list_all_source_documents
from .query_classifier import QueryClassifier

# Import enhanced components
try:
    from .reranker import create_reranker, HybridReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logging.warning("Enhanced reranker not available")

try:
    from .raptor import create_raptor_retriever, RAPTORRetriever
    RAPTOR_AVAILABLE = True
except ImportError:
    RAPTOR_AVAILABLE = False
    logging.warning("RAPTOR not available")

APPCFG = LoadConfig()

CODESPACE_NAME = os.getenv("CODESPACE_NAME")
SERVE_PORT = 8000  # Port used by serve.py
CODESPACE_URL = f"https://{CODESPACE_NAME}-{SERVE_PORT}.app.github.dev" if CODESPACE_NAME else f"http://localhost:{SERVE_PORT}"


def ensure_directory_access(directory: str):
    """Ensure the directory exists and has write access."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"Access denied to {directory}. Check permissions.")

def get_azure_blob_name(clean_filename: str) -> str:
    """
    Get the correct Azure blob name from azure_blob_metadata.json
    Maps from clean filename to the full blob name with timestamp prefix
    """
    try:
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'vectordb', 'azure_blob_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                azure_metadata = json.load(f)

            # Look for a document with matching clean filename
            for doc in azure_metadata.get('uploaded_documents', []):
                if doc.get('file_name') == clean_filename:
                    return doc.get('blob_name', clean_filename)

        # Fallback: if not found in metadata, assume the filename already has timestamp prefix
        return clean_filename

    except Exception as e:
        logging.warning(f"Failed to load Azure blob metadata: {e}")
        return clean_filename

class ChatBot:
    """
    Enhanced ChatBot with advanced retrieval and reranking capabilities
    """
    
    # Class-level configuration for enhanced features
    USE_RERANKER = True
    USE_RAPTOR = False  # Can be enabled for specific use cases
    RERANKER_TOP_K = 10  # Retrieve more documents for reranking
    FINAL_TOP_K = 5      # Final number of documents to use
    
    @staticmethod
    def _initialize_reranker():
        """Initialize reranker if available"""
        if RERANKER_AVAILABLE and ChatBot.USE_RERANKER:
            try:
                return create_reranker("hybrid")
            except Exception as e:
                logging.warning(f"Failed to initialize reranker: {e}")
                return None
        return None
    
    @staticmethod
    def _apply_reranking(query: str, docs: List, reranker) -> List:
        """Apply reranking to retrieved documents"""
        if not reranker or not docs:
            return docs
        
        try:
            # Convert LangChain documents to reranker format
            reranker_docs = []
            for doc in docs:
                doc_dict = {
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                }
                reranker_docs.append(doc_dict)
            
            # Apply reranking
            reranked_docs = reranker.rerank(
                query=query, 
                documents=reranker_docs, 
                top_k=ChatBot.FINAL_TOP_K
            )
            
            # Convert back to LangChain document format
            from langchain_core.documents import Document
            final_docs = []
            for doc_dict in reranked_docs:
                doc = Document(
                    page_content=doc_dict['page_content'],
                    metadata=doc_dict['metadata']
                )
                final_docs.append(doc)
            
            logging.info(f"Reranked {len(docs)} documents to {len(final_docs)} final documents")
            return final_docs
            
        except Exception as e:
            logging.error(f"Reranking failed: {e}")
            return docs[:ChatBot.FINAL_TOP_K]  # Fallback to top-k without reranking
    
    @staticmethod
    def respond(chatbot, message, data_type="Current documents", temperature=0.0):
        try:
            query_lower = message.lower()
            # Check if the query is asking for a full list of documents.
            if ("list" in query_lower and "document" in query_lower
                and ("all" in query_lower or "full" in query_lower or "vector" in query_lower)):
                from .vector_utils import list_all_source_documents
                unique_sources = list_all_source_documents(APPCFG.persist_directory, APPCFG.embedding_model)
                answer = "The source documents in the vector database are:\n" + "\n".join(unique_sources)
                chatbot.append({"role": "user", "content": message})
                chatbot.append({"role": "assistant", "content": answer})
                return answer, chatbot, answer

            # Initialize reranker
            reranker = ChatBot._initialize_reranker()

            # Select the appropriate persist directory
            if data_type == "Current documents":
                persist_dir = APPCFG.persist_directory
            elif data_type == "New document":
                persist_dir = APPCFG.custom_persist_directory
            else:
                raise ValueError("Invalid data_type provided.")
                
            # Ensure directory exists
            ensure_directory_access(persist_dir)
            
            # Use connection manager to handle database connections
            retrieval_k = ChatBot.RERANKER_TOP_K if reranker else APPCFG.k
            with db_manager.get_db(persist_dir, APPCFG.embedding_model) as vectordb:
                # Perform similarity search with scores
                try:
                    docs_with_scores = vectordb.similarity_search_with_score(message, k=retrieval_k)
                    docs = [doc for doc, score in docs_with_scores]
                    scores = [score for doc, score in docs_with_scores]
                except:
                    # Fallback if similarity_search_with_score is not available
                    docs = vectordb.similarity_search(message, k=retrieval_k)
                    scores = None
            
            # Apply reranking if available
            if reranker:
                docs = ChatBot._apply_reranking(message, docs, reranker)

            # Pass data_type to clean_references for proper formatting
            if not docs:
                retrieved_content = "No relevant documents found in the vector database."
                structured_sources = []
            else:
                retrieved_content, structured_sources = ChatBot.clean_references(docs, data_type)

            chat_history = "\n".join([f"{role}: {text}" for role, text in chatbot[-APPCFG.number_of_q_a_pairs:]])
            
            # Determine response strategy using intelligent classification
            strategy = QueryClassifier.determine_response_strategy(
                query=message,
                documents=docs,
                scores=scores if 'scores' in locals() else None,
                conversation_history=chatbot
            )
            
            # Log strategy for debugging
            logging.info(f"Response strategy: {strategy}")
            
            # Format prompt based on strategy
            # Don't include system_role in the prompt as it's already sent separately
            prompt = QueryClassifier.format_prompt_with_strategy(
                query=message,
                context=retrieved_content,
                strategy=strategy,
                system_role="",  # Empty string - system role is sent separately in messages
                data_type=data_type  # Pass data_type to control Sources formatting
            )
            
            # Add conversation history if relevant
            if chatbot and len(chatbot) > 0:
                prompt = f"Previous conversation:\n{chat_history}\n\n{prompt}"

            # Debug: log the prompt
            print("Generated Prompt:\n", prompt)
            
            # Primary: Azure OpenAI
            response_text = None
            azure_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
            azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or APPCFG.llm_engine
            tried_azure = False
            if azure_key and azure_endpoint:
                try:
                    print(f"DEBUG: Calling Azure OpenAI with deployment: {azure_deployment}")
                    print(f"DEBUG: Endpoint: {azure_endpoint}")
                    print(f"DEBUG: Prompt length: {len(prompt)} characters")

                    client = openai.AzureOpenAI(
                        api_key=azure_key,
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                        azure_endpoint=azure_endpoint
                    )
                    resp = client.chat.completions.create(
                        model=azure_deployment,  # In Azure, must be deployment name
                        messages=[
                            {"role": "system", "content": APPCFG.llm_system_role},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=8192,  # Add explicit max_tokens
                        timeout=60  # Add timeout
                    )
                    response_text = resp.choices[0].message.content
                    print(f"DEBUG: Azure response received, length: {len(response_text)} characters")
                    tried_azure = True
                except Exception as e:
                    err_text = str(e)
                    print(f"Azure chat call failed: {err_text}")
                    import traceback
                    print(f"DEBUG: Full traceback:\n{traceback.format_exc()}")
                    # If it's an auth/endpoint error, fall through to OpenAI fallback
                    if "401" not in err_text and "Access denied" not in err_text and "endpoint" not in err_text.lower():
                        raise

            # Fallback: OpenAI (non-Azure) if OPENAI_API_KEY is present
            if response_text is None:
                openai_key = os.getenv("OPENAI_API_KEY")
                if openai_key:
                    try:
                        std_client = openai.OpenAI(api_key=openai_key)
                        std_resp = std_client.chat.completions.create(
                            model=APPCFG.llm_engine,  # e.g., gpt-4o-mini
                            messages=[
                                {"role": "system", "content": APPCFG.llm_system_role},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=temperature,
                        )
                        response_text = std_resp.choices[0].message.content
                    except Exception as e2:
                        print(f"OpenAI standard call failed: {e2}")

            # If still no response, provide actionable error
            if response_text is None:
                if tried_azure:
                    response_text = (
                        "Authentication error calling Azure OpenAI. Verify AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                        "(correct resource URL/region), and ensure llm_config.engine matches your Azure deployment name."
                    )
                else:
                    response_text = (
                        "No valid LLM credentials found. Set AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT or OPENAI_API_KEY."
                    )
            
            chatbot.append({"role": "user", "content": message})
            chatbot.append({"role": "assistant", "content": response_text})

            # Return structured sources instead of raw retrieved content
            import json
            return response_text, chatbot, json.dumps(structured_sources)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            chatbot.append({"role": "assistant", "content": error_msg})
            return error_msg, chatbot, None

    @staticmethod
    def clean_references(documents: List, data_type: str = "Current documents") -> tuple:
        """
        Clean and format references from retrieved documents.
        Returns: (context_for_llm, structured_sources)
        - context_for_llm: String formatted for LLM context with numbered sources
        - structured_sources: List of dicts with source metadata for UI rendering
        """
        if not documents:
            return "No relevant documents found in the vector database.", []

        markdown_documents = ""
        structured_sources = []
        counter = 1

        # Azure Blob Storage base URL
        AZURE_BLOB_URL = "https://sandbox3190080146.blob.core.windows.net/documents/"

        for doc in documents:
            try:
                content = doc.page_content
                metadata = doc.metadata

                # Clean content and handle Unicode issues
                content = content.strip()
                content = re.sub(r'\s+', ' ', content).strip()

                # Handle common PDF encoding issues
                content = re.sub(r'â', '-', content)
                content = re.sub(r'â', '∈', content)
                content = re.sub(r'Ã', '×', content)
                content = re.sub(r'ï¬', 'fi', content)
                content = re.sub(r'Â·', '·', content)

                # Handle problematic Unicode characters (Private Use Area and others)
                # Remove or replace characters that can't be encoded in Windows charmap
                content = re.sub(r'[\uf000-\uf8ff]', '', content)  # Remove Private Use Area characters
                content = re.sub(r'[\u2000-\u206f]', ' ', content)  # Replace special spaces with regular space
                content = re.sub(r'[\u2070-\u209f]', '', content)   # Remove superscripts/subscripts that cause issues

                # Ensure content is safely encodable
                try:
                    content.encode('utf-8', errors='ignore').decode('utf-8')
                except UnicodeEncodeError:
                    # If still problematic, use safe encoding
                    content = content.encode('ascii', errors='ignore').decode('ascii')

                # Get the full filename from source metadata
                full_filename = os.path.basename(metadata['source']).replace("\\", "/")

                # Clean filename for display (remove timestamp/hash prefix if present)
                display_filename = full_filename
                # Pattern to match timestamp_hash_ prefix like '20250908_022717_05ab142c_'
                import re as regex
                prefix_pattern = r'^\d{8}_\d{6}_[a-f0-9]{8}_'
                if regex.match(prefix_pattern, full_filename):
                    display_filename = regex.sub(prefix_pattern, '', full_filename)

                # Extract snippet (first 150-200 chars of content for preview)
                snippet = content[:200] + "..." if len(content) > 200 else content

                # Try to extract section from metadata or content
                section = metadata.get('section', '')
                if not section and 'heading' in metadata:
                    section = metadata.get('heading')

                # Format for LLM context (numbered sources)
                markdown_documents += f"[{counter}] Source Document:\n"
                markdown_documents += f"Content: {content}\n"
                markdown_documents += f"Document: {display_filename}\n"
                markdown_documents += f"Page: {metadata['page']}\n"
                
                # Add proper citation format for LLM to use
                if data_type == "Current documents":
                    # Include the exact format the LLM should use for citation
                    markdown_documents += f"Citation format for this source: [{counter}] {display_filename} - Page {metadata['page']} View\n"
                else:
                    markdown_documents += f"Citation format for this source: [{counter}] {display_filename} - Page {metadata['page']} View\n"

                # Build structured source for UI
                source_dict = {
                    'number': counter,
                    'filename': display_filename,
                    'full_filename': full_filename,
                    'page': metadata.get('page', 'N/A'),
                    'snippet': snippet,
                    'section': section,
                    'content': content  # Full content for reference
                }

                # Add URL based on mode
                if data_type == "Current documents":
                    # For Current documents mode, construct Azure Blob URL
                    # Get the correct blob name from Azure metadata using clean filename
                    correct_blob_name = get_azure_blob_name(display_filename)

                    # Azure Blob URLs work with spaces, so don't URL encode spaces
                    # Only encode special characters that could break URLs, but keep spaces
                    safe_filename = correct_blob_name.replace('%', '%25').replace('#', '%23').replace('?', '%3F')
                    pdf_url = f"{AZURE_BLOB_URL}{safe_filename}"
                    source_dict['url'] = pdf_url
                else:
                    # For New document mode, no View links (local files only)
                    source_dict['url'] = None
                
                # Add a blank line between sources for clarity
                markdown_documents += "\n"

                structured_sources.append(source_dict)
                counter += 1

            except Exception as e:
                print(f"Error processing document {counter}: {str(e)}")
                continue

        return markdown_documents, structured_sources