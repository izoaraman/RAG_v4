from langchain_community.document_loaders import PyPDFLoader
from utils.utils import count_num_tokens
import openai
import os
import logging
from typing import List, Union, Dict, Any, Optional

# Import advanced summarization capabilities
try:
    from .advanced_summarizer import create_auto_summarizer, SummarizationMethod
    ADVANCED_SUMMARIZER_AVAILABLE = True
except ImportError:
    ADVANCED_SUMMARIZER_AVAILABLE = False
    logging.warning("Advanced summarizer not available")


class Summarizer:
    """
    Enhanced summarizer with multiple strategies and automatic method selection
    """
    
    def __init__(self, use_advanced: bool = True):
        """
        Initialize summarizer
        
        Args:
            use_advanced: Whether to use advanced summarization methods
        """
        self.use_advanced = use_advanced and ADVANCED_SUMMARIZER_AVAILABLE
        self.auto_summarizer = None
        
        if self.use_advanced:
            try:
                # Initialize with Azure OpenAI client
                import openai
                from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings
                
                azure_key = os.getenv("AZURE_OPENAI_API_KEY")
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                if azure_key and azure_endpoint:
                    llm = AzureOpenAI(
                        api_key=azure_key,
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                        azure_endpoint=azure_endpoint,
                        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"))
                    )
                    embeddings = AzureOpenAIEmbeddings(
                        api_key=azure_key,
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                        azure_endpoint=azure_endpoint,
                        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"))
                    )
                    self.auto_summarizer = create_auto_summarizer(llm, embeddings)
                    logging.info("Advanced summarizer initialized with Azure")
                else:
                    # Fall back to a lightweight local LLM placeholder + local embeddings
                    try:
                        from langchain_community.llms import HuggingFaceHub
                        from langchain_community.embeddings import HuggingFaceEmbeddings
                        llm = HuggingFaceHub(repo_id=os.getenv("LOCAL_LLM_REPO", "google/flan-t5-small"))
                        embeddings = HuggingFaceEmbeddings(model_name=os.getenv(
                            "LOCAL_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
                        self.auto_summarizer = create_auto_summarizer(llm, embeddings)
                        logging.info("Advanced summarizer initialized with local models (no Azure creds)")
                    except Exception as inner_e:
                        raise RuntimeError("Neither Azure nor local summarization backends available.") from inner_e
                
            except Exception as e:
                logging.warning(f"Failed to initialize advanced summarizer: {e}")
                self.use_advanced = False
    def summarize_documents(self, 
                           documents: List[Union[str, Dict[str, Any]]], 
                           method: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Summarize documents using advanced methods with automatic selection
        
        Args:
            documents: List of documents (strings or Document objects)
            method: Specific method to use ("map_reduce", "map_refine", "chain_of_density", "clustering_map_refine")
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dictionary with summary and metadata
        """
        if self.use_advanced and self.auto_summarizer:
            # Use advanced summarization with automatic method selection
            try:
                # Convert method string to enum if provided
                summarization_method = None
                if method:
                    method_mapping = {
                        "map_reduce": SummarizationMethod.MAP_REDUCE,
                        "map_refine": SummarizationMethod.MAP_REFINE,
                        "chain_of_density": SummarizationMethod.CHAIN_OF_DENSITY,
                        "clustering_map_refine": SummarizationMethod.CLUSTERING_MAP_REFINE
                    }
                    summarization_method = method_mapping.get(method.lower())
                
                result = self.auto_summarizer.summarize(
                    documents=documents,
                    method=summarization_method,
                    **kwargs
                )
                
                logging.info(f"Advanced summarization completed using method: {result.get('method', 'unknown')}")
                return result
                
            except Exception as e:
                logging.error(f"Advanced summarization failed: {e}")
                # Fallback to traditional method
                return self._fallback_summarize(documents, **kwargs)
        else:
            # Use traditional summarization
            return self._fallback_summarize(documents, **kwargs)
    
    def _fallback_summarize(self, documents: List[Union[str, Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """
        Fallback to traditional summarization method
        
        Args:
            documents: List of documents to summarize
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with summary and basic metadata
        """
        try:
            # Extract text content
            texts = []
            for doc in documents:
                if isinstance(doc, str):
                    texts.append(doc)
                elif isinstance(doc, dict) and 'page_content' in doc:
                    texts.append(doc['page_content'])
                elif hasattr(doc, 'page_content'):
                    texts.append(doc.page_content)
                else:
                    texts.append(str(doc))
            
            combined_text = "\n\n".join(texts)
            
            # Use traditional method for single document
            summary = self.get_llm_response(
                gpt_model=kwargs.get('gpt_model', 'gpt-4'),
                temperature=kwargs.get('temperature', 0.0),
                llm_system_role=kwargs.get('system_role', "Please summarize the document concisely and comprehensively."),
                prompt=combined_text
            )
            
            return {
                "summary": summary,
                "method": "traditional_fallback",
                "analysis": {
                    "n_documents": len(documents),
                    "total_characters": len(combined_text)
                },
                "metrics": {
                    "execution_time": 0,  # Not tracked in fallback
                    "method": "traditional"
                },
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Fallback summarization failed: {e}")
            return {
                "summary": f"Summarization failed: {str(e)}",
                "method": "failed",
                "success": False
            }
    
    @staticmethod
    def summarize_the_pdf(
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gpt_model: str,
        temperature: float,
        summarizer_llm_system_role: str,
        final_summarizer_llm_system_role: str,
        character_overlap: int
    ):

        docs = []
        docs.extend(PyPDFLoader(file_dir).load())
        print(f"Document length: {len(docs)}")
        max_summarizer_output_token = int(
            max_final_token/len(docs)) - token_threshold
        full_summary = ""
        counter = 1
        print("Generating the summary..")
        # if the document has more than one pages
        if len(docs) > 1:
            for i in range(len(docs)):
                # NOTE: This part can be optimized by considering a better technique for creating the prompt. (e.g: lanchain "chunksize" and "chunkoverlap" arguments.)

                if i == 0:  # For the first page
                    prompt = docs[i].page_content + \
                        docs[i+1].page_content[:character_overlap]
                # For pages except the fist and the last one.
                elif i < len(docs)-1:
                    prompt = docs[i-1].page_content[-character_overlap:] + \
                        docs[i].page_content + \
                        docs[i+1].page_content[:character_overlap]
                else:  # For the last page
                    prompt = docs[i-1].page_content[-character_overlap:] + \
                        docs[i].page_content
                summarizer_llm_system_role = summarizer_llm_system_role.format(
                    max_summarizer_output_token)
                full_summary += Summarizer.get_llm_response(
                    gpt_model,
                    temperature,
                    summarizer_llm_system_role,
                    prompt=prompt
                )
        else:  # if the document has only one page
            full_summary = docs[0].page_content

            print(f"Page {counter} was summarized. ", end="")
            counter += 1
        print("\nFull summary token length:", count_num_tokens(
            full_summary, model=gpt_model))
        final_summary = Summarizer.get_llm_response(
            gpt_model,
            temperature,
            final_summarizer_llm_system_role,
            prompt=full_summary
        )
        return final_summary

    @staticmethod
    def get_llm_response(gpt_model: str, temperature: float, llm_system_role: str, prompt: str):
        """Retrieve response from Azure OpenAI."""
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            azure_endpoint=(os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        )
        
        # Prefer explicit Azure deployment env var over passed gpt_model if available
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or gpt_model
        response = client.chat.completions.create(
            model=azure_deployment,  # In Azure, this should be the deployment name
            messages=[
                {"role": "system", "content": llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        
        return response.choices[0].message.content
