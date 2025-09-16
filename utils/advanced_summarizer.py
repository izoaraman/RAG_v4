"""
Advanced Summarization Module with Automatic Method Selection

This module provides multiple summarization strategies and automatically selects
the most suitable method based on document characteristics and performance requirements.

Supported Methods:
- Map-Reduce: Parallel processing of document chunks
- Map-Refine: Sequential refinement of summaries  
- Chain of Density: Iterative density-based summarization
- Clustering Map-Refine: Cluster-based selective summarization
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logging.warning("Clustering not available. Install scikit-learn for clustering-based summarization")

try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
    from langchain.text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available for advanced summarization")

from .load_config import LoadConfig
from .utils import count_num_tokens

# Global configuration
APPCFG = LoadConfig()


class SummarizationMethod(Enum):
    """Enumeration of available summarization methods"""
    MAP_REDUCE = "map_reduce"
    MAP_REFINE = "map_refine"
    CHAIN_OF_DENSITY = "chain_of_density"
    CLUSTERING_MAP_REFINE = "clustering_map_refine"


class SummarizationStrategy(ABC):
    """Abstract base class for summarization strategies"""
    
    def __init__(self, llm, max_tokens: int = 4000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.execution_time = 0
        self.token_usage = 0
    
    @abstractmethod
    def summarize(self, documents: List[Union[str, Document]], **kwargs) -> str:
        """Summarize documents using the specific strategy"""
        pass
    
    @abstractmethod
    def estimate_cost(self, documents: List[Union[str, Document]]) -> float:
        """Estimate computational cost for the strategy"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "method": self.__class__.__name__
        }


class MapReduceSummarizer(SummarizationStrategy):
    """
    Map-Reduce summarization strategy
    Parallel processing of document chunks followed by reduction
    """
    
    def __init__(self, llm, max_tokens: int = 4000, chunk_size: int = 2000):
        super().__init__(llm, max_tokens)
        self.chunk_size = chunk_size
        
        # Map prompt for extracting key points
        self.map_prompt = ChatPromptTemplate.from_template("""
아래 문서에서 핵심 내용을 추출하세요. 주요 논점, 중요한 정보, 핵심 아이디어를 간결하게 정리하세요.

문서:
{doc}

핵심 내용 (1-5개 요점):
""")
        
        # Reduce prompt for final summarization
        self.reduce_prompt = ChatPromptTemplate.from_template("""
다음은 여러 문서에서 추출한 핵심 내용들입니다. 이를 종합하여 포괄적이고 일관성 있는 요약문을 작성하세요.

핵심 내용들:
{summaries}

언어: {language}

최종 요약:
""")
        
        if LANGCHAIN_AVAILABLE:
            self.map_chain = self.map_prompt | self.llm | StrOutputParser()
            self.reduce_chain = self.reduce_prompt | self.llm | StrOutputParser()
        else:
            self.map_chain = None
            self.reduce_chain = None
    
    def estimate_cost(self, documents: List[Union[str, Document]]) -> float:
        """Estimate cost based on number of map operations"""
        total_tokens = sum(count_num_tokens(self._extract_text(doc), "cl100k_base") for doc in documents)
        n_chunks = max(1, total_tokens // self.chunk_size)
        # Cost includes map operations + 1 reduce operation
        return n_chunks + 1
    
    def _extract_text(self, doc: Union[str, Document]) -> str:
        """Extract text from document"""
        if isinstance(doc, str):
            return doc
        elif hasattr(doc, 'page_content'):
            return doc.page_content
        else:
            return str(doc)
    
    def _split_documents(self, documents: List[Union[str, Document]]) -> List[str]:
        """Split documents into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=200
        )
        
        all_text = "\n\n".join(self._extract_text(doc) for doc in documents)
        return text_splitter.split_text(all_text)
    
    def summarize(self, documents: List[Union[str, Document]], language: str = "Korean", **kwargs) -> str:
        """
        Perform Map-Reduce summarization
        
        Args:
            documents: List of documents to summarize
            language: Output language
            
        Returns:
            Final summary
        """
        start_time = time.time()
        
        # Split documents into chunks
        chunks = self._split_documents(documents)
        
        # Map phase: Extract key points from each chunk
        if self.map_chain is not None:
            map_inputs = [{"doc": chunk} for chunk in chunks]
            key_points = self.map_chain.batch(map_inputs)
        else:
            # Fallback when LangChain is not available
            key_points = [f"Mock key points from chunk {i+1}: {chunk[:100]}..." for i, chunk in enumerate(chunks)]
        
        # Count tokens used
        self.token_usage = sum(count_num_tokens(point, "cl100k_base") for point in key_points)
        
        # Reduce phase: Combine key points into final summary
        combined_summaries = "\n".join(f"- {point}" for point in key_points)
        if self.reduce_chain is not None:
            final_summary = self.reduce_chain.invoke({
                "summaries": combined_summaries,
                "language": language
            })
        else:
            # Fallback when LangChain is not available
            final_summary = f"Mock Map-Reduce summary in {language}: {combined_summaries[:200]}..."
        
        self.execution_time = time.time() - start_time
        logging.info(f"Map-Reduce completed in {self.execution_time:.2f}s with {len(chunks)} chunks")
        
        return final_summary


class MapRefineSummarizer(SummarizationStrategy):
    """
    Map-Refine summarization strategy
    Sequential refinement of summaries maintaining context
    """
    
    def __init__(self, llm, max_tokens: int = 4000, chunk_size: int = 2000):
        super().__init__(llm, max_tokens)
        self.chunk_size = chunk_size
        
        # Initial summary prompt
        self.map_prompt = ChatPromptTemplate.from_template("""
다음 문서의 핵심 내용을 요약하세요:

문서:
{documents}

언어: {language}

요약:
""")
        
        # Refinement prompt
        self.refine_prompt = ChatPromptTemplate.from_template("""
기존 요약이 있습니다. 새로운 문서의 내용을 바탕으로 요약을 개선하고 보완하세요.

기존 요약:
{previous_summary}

새로운 문서:
{current_summary}

언어: {language}

개선된 요약:
""")
        
        if LANGCHAIN_AVAILABLE:
            self.map_chain = self.map_prompt | self.llm | StrOutputParser()
            self.refine_chain = self.refine_prompt | self.llm | StrOutputParser()
        else:
            self.map_chain = None
            self.refine_chain = None
    
    def estimate_cost(self, documents: List[Union[str, Document]]) -> float:
        """Estimate cost based on sequential operations"""
        total_tokens = sum(count_num_tokens(self._extract_text(doc), "cl100k_base") for doc in documents)
        n_chunks = max(1, total_tokens // self.chunk_size)
        # Cost is n_chunks operations (sequential)
        return n_chunks
    
    def _extract_text(self, doc: Union[str, Document]) -> str:
        """Extract text from document"""
        if isinstance(doc, str):
            return doc
        elif hasattr(doc, 'page_content'):
            return doc.page_content
        else:
            return str(doc)
    
    def _split_documents(self, documents: List[Union[str, Document]]) -> List[str]:
        """Split documents into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=200
        )
        
        all_text = "\n\n".join(self._extract_text(doc) for doc in documents)
        return text_splitter.split_text(all_text)
    
    def summarize(self, documents: List[Union[str, Document]], language: str = "Korean", **kwargs) -> str:
        """
        Perform Map-Refine summarization
        
        Args:
            documents: List of documents to summarize
            language: Output language
            
        Returns:
            Final refined summary
        """
        start_time = time.time()
        
        # Split documents into chunks
        chunks = self._split_documents(documents)
        
        if not chunks:
            return "요약할 문서가 없습니다."
        
        # Initial summary from first chunk
        if self.map_chain is not None:
            current_summary = self.map_chain.invoke({
                "documents": chunks[0],
                "language": language
            })
        else:
            # Fallback when LangChain is not available
            current_summary = f"Mock initial summary in {language}: {chunks[0][:200]}..."
        
        self.token_usage = count_num_tokens(current_summary, "cl100k_base")
        
        # Refine with remaining chunks
        for chunk in chunks[1:]:
            if self.refine_chain is not None:
                current_summary = self.refine_chain.invoke({
                    "previous_summary": current_summary,
                    "current_summary": chunk,
                    "language": language
                })
            else:
                # Fallback when LangChain is not available
                current_summary = f"Mock refined summary: {current_summary[:100]}... + {chunk[:100]}..."
            self.token_usage += count_num_tokens(current_summary, "cl100k_base")
        
        self.execution_time = time.time() - start_time
        logging.info(f"Map-Refine completed in {self.execution_time:.2f}s with {len(chunks)} chunks")
        
        return current_summary


class ChainOfDensitySummarizer(SummarizationStrategy):
    """
    Chain of Density summarization strategy
    Iteratively increases information density while maintaining readability
    """
    
    def __init__(self, llm, max_tokens: int = 4000, max_words: int = 80, iterations: int = 3):
        super().__init__(llm, max_tokens)
        self.max_words = max_words
        self.iterations = iterations
        
        # Chain of Density prompt
        self.cod_prompt = ChatPromptTemplate.from_template("""
다음 지시사항에 따라 문서를 요약하세요:

1. 먼저 {max_words}단어 이하의 정보적이고 개체가 적은 요약을 작성하세요.
2. 그 다음 {iterations}번의 반복을 통해 누락된 중요한 개체들을 식별하고 통합하세요.
3. 각 반복에서 이전 요약을 다시 작성하여 {entity_range}개의 누락된 중요 개체를 포함시키세요.
4. 요약의 길이는 {max_words}단어를 유지하되 밀도를 높이세요.

문서:
{content}

다음 JSON 형식으로 응답하세요:
[
  {{
    "missing_entities": "첫 번째 요약에서 누락된 중요 개체들 (세미콜론으로 구분)",
    "denser_summary": "더 밀도 높은 요약문"
  }},
  ...
]

요약 반복 횟수: {iterations}
""")
        
        if LANGCHAIN_AVAILABLE:
            self.chain = self.cod_prompt | self.llm | SimpleJsonOutputParser()
        else:
            self.chain = None
    
    def estimate_cost(self, documents: List[Union[str, Document]]) -> float:
        """Estimate cost based on iterations"""
        # CoD requires multiple iterations on the same content
        return self.iterations + 1
    
    def _extract_text(self, doc: Union[str, Document]) -> str:
        """Extract text from document"""
        if isinstance(doc, str):
            return doc
        elif hasattr(doc, 'page_content'):
            return doc.page_content
        else:
            return str(doc)
    
    def summarize(self, documents: List[Union[str, Document]], 
                 entity_range: str = "1-3", **kwargs) -> str:
        """
        Perform Chain of Density summarization
        
        Args:
            documents: List of documents to summarize
            entity_range: Range of entities to add per iteration
            
        Returns:
            Final dense summary
        """
        start_time = time.time()
        
        # Combine all documents
        content = "\n\n".join(self._extract_text(doc) for doc in documents)
        
        # Apply Chain of Density
        try:
            if self.chain is not None:
                results = self.chain.invoke({
                    "content": content,
                    "max_words": self.max_words,
                    "iterations": self.iterations,
                    "entity_range": entity_range
                })
            else:
                # Fallback when LangChain is not available
                results = [
                    {
                        "missing_entities": "entity1; entity2; entity3",
                        "denser_summary": f"Mock Chain of Density summary: {content[:200]}... (density enhanced)"
                    }
                ]
            
            # Extract final summary
            if results and len(results) > 0:
                final_summary = results[-1].get("denser_summary", "요약 생성 실패")
            else:
                final_summary = "Chain of Density 처리 실패"
                
            self.token_usage = count_num_tokens(final_summary, "cl100k_base")
            
        except Exception as e:
            logging.error(f"Chain of Density failed: {e}")
            final_summary = f"요약 처리 중 오류 발생: {str(e)}"
        
        self.execution_time = time.time() - start_time
        logging.info(f"Chain of Density completed in {self.execution_time:.2f}s")
        
        return final_summary


class ClusteringMapRefineSummarizer(SummarizationStrategy):
    """
    Clustering Map-Refine summarization strategy
    Clusters documents and selects representative samples for efficient summarization
    """
    
    def __init__(self, llm, embeddings, max_tokens: int = 4000, n_clusters: int = 5):
        super().__init__(llm, max_tokens)
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        
        # Use Map-Refine for the selected documents
        self.map_refine = MapRefineSummarizer(llm, max_tokens)
    
    def estimate_cost(self, documents: List[Union[str, Document]]) -> float:
        """Estimate cost based on clustering reduction"""
        if not CLUSTERING_AVAILABLE:
            return len(documents)  # Fallback to full processing
        
        # Cost is reduced to number of clusters instead of all documents
        n_clusters = min(self.n_clusters, len(documents))
        return n_clusters
    
    def _extract_text(self, doc: Union[str, Document]) -> str:
        """Extract text from document"""
        if isinstance(doc, str):
            return doc
        elif hasattr(doc, 'page_content'):
            return doc.page_content
        else:
            return str(doc)
    
    def _cluster_documents(self, documents: List[Union[str, Document]]) -> List[int]:
        """
        Cluster documents and return representative document indices
        
        Args:
            documents: List of documents to cluster
            
        Returns:
            List of indices of representative documents
        """
        if not CLUSTERING_AVAILABLE:
            logging.warning("Clustering not available, using all documents")
            return list(range(len(documents)))
        
        # Extract texts and create embeddings
        texts = [self._extract_text(doc) for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings)
        
        # Perform clustering
        n_clusters = min(self.n_clusters, len(documents))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # Find representative documents (closest to cluster centers)
        representative_indices = []
        for i in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            
            # Find documents in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Calculate distances to cluster center
            cluster_embeddings = embeddings_array[cluster_indices]
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            
            # Select closest document
            closest_idx = cluster_indices[np.argmin(distances)]
            representative_indices.append(int(closest_idx))
        
        return sorted(representative_indices)
    
    def summarize(self, documents: List[Union[str, Document]], **kwargs) -> str:
        """
        Perform Clustering Map-Refine summarization
        
        Args:
            documents: List of documents to summarize
            
        Returns:
            Final summary from representative documents
        """
        start_time = time.time()
        
        if len(documents) <= self.n_clusters:
            # If we have few documents, process all
            selected_docs = documents
            logging.info("Using all documents (fewer than cluster count)")
        else:
            # Cluster and select representatives
            representative_indices = self._cluster_documents(documents)
            selected_docs = [documents[i] for i in representative_indices]
            logging.info(f"Selected {len(selected_docs)} representative documents from {len(documents)} total")
        
        # Apply Map-Refine to selected documents
        summary = self.map_refine.summarize(selected_docs, **kwargs)
        
        # Update metrics
        self.execution_time = time.time() - start_time
        self.token_usage = self.map_refine.token_usage
        
        logging.info(f"Clustering Map-Refine completed in {self.execution_time:.2f}s")
        return summary


class AutoSummarizationSelector:
    """
    Automatic summarization method selector
    Chooses the optimal summarization strategy based on document characteristics
    """
    
    def __init__(self, llm, embeddings, max_execution_time: float = 60.0):
        """
        Initialize auto selector
        
        Args:
            llm: Language model for summarization
            embeddings: Embedding model for clustering
            max_execution_time: Maximum acceptable execution time in seconds
        """
        self.llm = llm
        self.embeddings = embeddings
        self.max_execution_time = max_execution_time
        
        # Initialize strategies
        self.strategies = {
            SummarizationMethod.MAP_REDUCE: MapReduceSummarizer(llm),
            SummarizationMethod.MAP_REFINE: MapRefineSummarizer(llm),
            SummarizationMethod.CHAIN_OF_DENSITY: ChainOfDensitySummarizer(llm),
            SummarizationMethod.CLUSTERING_MAP_REFINE: ClusteringMapRefineSummarizer(llm, embeddings)
        }
        
        logging.info("AutoSummarizationSelector initialized with all strategies")
    
    def _analyze_documents(self, documents: List[Union[str, Document]]) -> Dict[str, Any]:
        """
        Analyze document characteristics to inform method selection
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary with document analysis
        """
        # Extract texts
        texts = []
        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
            elif hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            else:
                texts.append(str(doc))
        
        # Calculate metrics
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(count_num_tokens(text, "cl100k_base") for text in texts)
        avg_doc_length = total_chars / len(documents) if documents else 0
        
        analysis = {
            "n_documents": len(documents),
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "avg_document_length": avg_doc_length,
            "is_large_collection": len(documents) > 10,
            "is_long_documents": avg_doc_length > 5000,
            "is_high_token_count": total_tokens > 10000
        }
        
        return analysis
    
    def _select_optimal_method(self, analysis: Dict[str, Any]) -> SummarizationMethod:
        """
        Select optimal summarization method based on analysis
        
        Decision logic:
        - Chain of Density: Best quality for small to medium documents
        - Map-Reduce: Fast parallel processing for large collections
        - Clustering Map-Refine: Efficient for very large collections
        - Map-Refine: Sequential processing maintaining context
        
        Args:
            analysis: Document analysis results
            
        Returns:
            Selected summarization method
        """
        n_docs = analysis["n_documents"]
        total_tokens = analysis["total_tokens"]
        is_large = analysis["is_large_collection"]
        is_long = analysis["is_long_documents"]
        
        # Decision tree based on document characteristics
        if total_tokens < 5000 and n_docs <= 5:
            # Small collection, high quality summarization
            method = SummarizationMethod.CHAIN_OF_DENSITY
            reason = "Small document collection - using Chain of Density for highest quality"
            
        elif is_large and CLUSTERING_AVAILABLE:
            # Large collection, use clustering for efficiency
            method = SummarizationMethod.CLUSTERING_MAP_REFINE
            reason = "Large document collection - using Clustering Map-Refine for efficiency"
            
        elif is_long or total_tokens > 15000:
            # Long documents or high token count, parallel processing
            method = SummarizationMethod.MAP_REDUCE
            reason = "Long documents or high token count - using Map-Reduce for parallel processing"
            
        else:
            # Default to Map-Refine for balanced approach
            method = SummarizationMethod.MAP_REFINE
            reason = "Balanced requirements - using Map-Refine for context preservation"
        
        logging.info(f"Selected method: {method.value} - {reason}")
        return method
    
    def summarize(self, documents: List[Union[str, Document]], 
                 method: Optional[SummarizationMethod] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Automatically select and apply optimal summarization method
        
        Args:
            documents: List of documents to summarize
            method: Optional specific method to use (overrides auto-selection)
            **kwargs: Additional arguments for summarization
            
        Returns:
            Dictionary with summary and metadata
        """
        if not documents:
            return {
                "summary": "요약할 문서가 없습니다.",
                "method": "none",
                "analysis": {},
                "metrics": {}
            }
        
        # Analyze documents
        analysis = self._analyze_documents(documents)
        
        # Select method
        if method is None:
            selected_method = self._select_optimal_method(analysis)
        else:
            selected_method = method
            logging.info(f"Using specified method: {selected_method.value}")
        
        # Get strategy and check availability
        strategy = self.strategies[selected_method]
        
        # Fallback if clustering not available
        if (selected_method == SummarizationMethod.CLUSTERING_MAP_REFINE and 
            not CLUSTERING_AVAILABLE):
            logging.warning("Clustering not available, falling back to Map-Reduce")
            strategy = self.strategies[SummarizationMethod.MAP_REDUCE]
            selected_method = SummarizationMethod.MAP_REDUCE
        
        # Estimate cost and check time constraints
        estimated_cost = strategy.estimate_cost(documents)
        
        # Apply summarization
        try:
            summary = strategy.summarize(documents, **kwargs)
            metrics = strategy.get_metrics()
            
            return {
                "summary": summary,
                "method": selected_method.value,
                "analysis": analysis,
                "metrics": metrics,
                "estimated_cost": estimated_cost,
                "success": True
            }
            
        except Exception as e:
            logging.error(f"Summarization failed with {selected_method.value}: {e}")
            
            # Fallback to simpler method
            if selected_method != SummarizationMethod.MAP_REDUCE:
                logging.info("Falling back to Map-Reduce")
                fallback_strategy = self.strategies[SummarizationMethod.MAP_REDUCE]
                try:
                    summary = fallback_strategy.summarize(documents, **kwargs)
                    metrics = fallback_strategy.get_metrics()
                    
                    return {
                        "summary": summary,
                        "method": "map_reduce_fallback",
                        "analysis": analysis,
                        "metrics": metrics,
                        "estimated_cost": fallback_strategy.estimate_cost(documents),
                        "success": True,
                        "fallback_reason": str(e)
                    }
                except Exception as fallback_error:
                    logging.error(f"Fallback also failed: {fallback_error}")
            
            return {
                "summary": f"요약 생성 실패: {str(e)}",
                "method": selected_method.value,
                "analysis": analysis,
                "metrics": {"error": str(e)},
                "estimated_cost": estimated_cost,
                "success": False
            }


def create_auto_summarizer(llm, embeddings, **kwargs) -> AutoSummarizationSelector:
    """
    Factory function to create auto summarization selector
    
    Args:
        llm: Language model for summarization
        embeddings: Embedding model for clustering
        **kwargs: Additional arguments
        
    Returns:
        Initialized auto summarization selector
    """
    return AutoSummarizationSelector(llm, embeddings, **kwargs)