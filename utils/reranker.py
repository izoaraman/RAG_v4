"""
Enhanced Reranker Module for RAG System

This module provides both FlashRank and Cross-Encoder reranking capabilities
for improving retrieval accuracy in RAG systems. The rerankers work locally
without requiring external API calls for fast response times.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    logging.warning("FlashRank not available. Install with: pip install flashrank")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")


class BaseReranker(ABC):
    """Abstract base class for rerankers"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: The search query
            documents: List of documents with 'page_content' and 'metadata'
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with relevance scores
        """
        pass


class FlashRankReranker(BaseReranker):
    """
    FlashRank-based reranker for fast local reranking
    Uses lightweight models for rapid inference without external API calls
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initialize FlashRank reranker with local model download
        
        Args:
            model_name: Model to use for reranking
                      Options: "ms-marco-MiniLM-L-12-v2" (default, balanced speed/accuracy), 
                              "ms-marco-MultiBERT-L-12" (higher accuracy, slower),
                              "ms-marco-TinyBERT-L-2-v2" (fastest, lower accuracy),
                              "rank-T5-flan" (advanced, slower)
        """
        if not FLASHRANK_AVAILABLE:
            raise ImportError("FlashRank not available. Install with: pip install flashrank")
            
        self.model_name = model_name
        
        # Initialize FlashRank with local model (no API calls)
        # This will download the model locally on first use
        try:
            # Try the current FlashRank API
            self.ranker = Ranker(model_name=model_name, cache_dir="./.flashrank_cache")
            logging.info(f"FlashRankReranker initialized with local model: {model_name}")
        except TypeError as e:
            # Fallback for older FlashRank API
            try:
                self.ranker = Ranker(model=model_name)
                logging.info(f"FlashRankReranker initialized with model (legacy API): {model_name}")
            except Exception as e2:
                # Final fallback to default model
                try:
                    self.ranker = Ranker()
                    logging.warning(f"Using default FlashRank model due to errors: {e}, {e2}")
                except Exception as e3:
                    raise ImportError(f"Failed to initialize FlashRank with any method: {e3}")
        except Exception as e:
            raise ImportError(f"Failed to initialize FlashRank: {e}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using FlashRank
        
        Args:
            query: The search query
            documents: List of documents with 'page_content' and 'metadata'
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with relevance scores
        """
        if not documents:
            return []
        
        # Prepare passages for FlashRank
        passages = []
        for i, doc in enumerate(documents):
            passage_text = doc.get('page_content', '') if isinstance(doc, dict) else str(doc.page_content)
            passages.append({
                "id": i,
                "text": passage_text,
                "meta": doc.get('metadata', {}) if isinstance(doc, dict) else getattr(doc, 'metadata', {})
            })
        
        # Create rerank request
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # Get reranked results
        results = self.ranker.rerank(rerank_request)
        
        # Format results with scores and limit to top_k
        reranked_docs = []
        for result in results[:top_k]:
            original_doc = documents[result["id"]]
            
            # Create enhanced document with relevance score
            enhanced_doc = {
                'page_content': result["text"],
                'metadata': {
                    **result.get("meta", {}),
                    'relevance_score': result["score"],
                    'rerank_position': len(reranked_docs) + 1
                }
            }
            reranked_docs.append(enhanced_doc)
        
        logging.info(f"FlashRank reranked {len(documents)} documents, returning top {len(reranked_docs)}")
        return reranked_docs


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder based reranker using sentence-transformers
    Provides high accuracy reranking with detailed similarity scoring
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize Cross-Encoder reranker
        
        Args:
            model_name: Model to use for reranking
                      Options: "BAAI/bge-reranker-v2-m3" (multilingual), 
                              "ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
            
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        logging.info(f"CrossEncoderReranker initialized with model: {model_name}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cross-Encoder
        
        Args:
            query: The search query
            documents: List of documents with 'page_content' and 'metadata'
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with relevance scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            passage_text = doc.get('page_content', '') if isinstance(doc, dict) else str(doc.page_content)
            pairs.append([query, passage_text])
        
        # Get similarity scores
        scores = self.model.predict(pairs)
        
        # Create scored documents
        scored_docs = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            passage_text = doc.get('page_content', '') if isinstance(doc, dict) else str(doc.page_content)
            metadata = doc.get('metadata', {}) if isinstance(doc, dict) else getattr(doc, 'metadata', {})
            
            enhanced_doc = {
                'page_content': passage_text,
                'metadata': {
                    **metadata,
                    'relevance_score': float(score),
                    'original_position': i
                }
            }
            scored_docs.append(enhanced_doc)
        
        # Sort by score (descending) and return top_k
        reranked_docs = sorted(scored_docs, key=lambda x: x['metadata']['relevance_score'], reverse=True)[:top_k]
        
        # Add rerank position
        for i, doc in enumerate(reranked_docs):
            doc['metadata']['rerank_position'] = i + 1
        
        logging.info(f"CrossEncoder reranked {len(documents)} documents, returning top {len(reranked_docs)}")
        return reranked_docs


class HybridReranker(BaseReranker):
    """
    Hybrid reranker that combines multiple reranking strategies
    Uses both FlashRank and Cross-Encoder for optimal results
    """
    
    def __init__(self, 
                 primary_model: str = "flashrank",
                 flashrank_model: str = "ms-marco-MiniLM-L-12-v2",
                 cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                 hybrid_weight: float = 0.7):
        """
        Initialize Hybrid reranker
        
        Args:
            primary_model: Primary reranking model ("flashrank" or "cross_encoder")
            flashrank_model: FlashRank model name (default: ms-marco-MiniLM-L-12-v2 for balanced speed/accuracy)
            cross_encoder_model: Cross-Encoder model name  
            hybrid_weight: Weight for primary model (0.0-1.0)
        """
        self.primary_model = primary_model
        self.hybrid_weight = hybrid_weight
        
        # Initialize available rerankers
        self.rerankers = {}
        
        if FLASHRANK_AVAILABLE:
            self.rerankers["flashrank"] = FlashRankReranker(flashrank_model)
        
        if CROSS_ENCODER_AVAILABLE:
            self.rerankers["cross_encoder"] = CrossEncoderReranker(cross_encoder_model)
        
        if not self.rerankers:
            raise ImportError("No reranking models available. Install flashrank or sentence-transformers")
        
        logging.info(f"HybridReranker initialized with primary: {primary_model}, available: {list(self.rerankers.keys())}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using hybrid approach
        
        Args:
            query: The search query
            documents: List of documents with 'page_content' and 'metadata'
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with combined relevance scores
        """
        if not documents:
            return []
        
        # If only one reranker available, use it
        if len(self.rerankers) == 1:
            reranker = list(self.rerankers.values())[0]
            return reranker.rerank(query, documents, top_k)
        
        # Use hybrid approach if both available
        if "flashrank" in self.rerankers and "cross_encoder" in self.rerankers:
            # Get results from both rerankers with extended top_k for better coverage
            extended_k = min(len(documents), top_k * 2)
            
            flashrank_results = self.rerankers["flashrank"].rerank(query, documents, extended_k)
            cross_encoder_results = self.rerankers["cross_encoder"].rerank(query, documents, extended_k)
            
            # Combine scores using weighted average
            combined_docs = {}
            
            # Process FlashRank results
            for doc in flashrank_results:
                doc_id = doc['page_content']  # Use content as ID for matching
                combined_docs[doc_id] = {
                    'doc': doc,
                    'flashrank_score': doc['metadata']['relevance_score'],
                    'cross_encoder_score': 0.0
                }
            
            # Process Cross-Encoder results
            for doc in cross_encoder_results:
                doc_id = doc['page_content']
                if doc_id in combined_docs:
                    combined_docs[doc_id]['cross_encoder_score'] = doc['metadata']['relevance_score']
                else:
                    combined_docs[doc_id] = {
                        'doc': doc,
                        'flashrank_score': 0.0,
                        'cross_encoder_score': doc['metadata']['relevance_score']
                    }
            
            # Calculate combined scores
            final_docs = []
            for doc_data in combined_docs.values():
                primary_score = (doc_data['flashrank_score'] if self.primary_model == "flashrank" 
                               else doc_data['cross_encoder_score'])
                secondary_score = (doc_data['cross_encoder_score'] if self.primary_model == "flashrank"
                                 else doc_data['flashrank_score'])
                
                combined_score = (self.hybrid_weight * primary_score + 
                                (1 - self.hybrid_weight) * secondary_score)
                
                doc = doc_data['doc'].copy()
                doc['metadata'] = {
                    **doc['metadata'],
                    'combined_relevance_score': combined_score,
                    'flashrank_score': doc_data['flashrank_score'],
                    'cross_encoder_score': doc_data['cross_encoder_score']
                }
                final_docs.append(doc)
            
            # Sort by combined score and return top_k
            final_docs.sort(key=lambda x: x['metadata']['combined_relevance_score'], reverse=True)
            result_docs = final_docs[:top_k]
            
            # Add final position
            for i, doc in enumerate(result_docs):
                doc['metadata']['rerank_position'] = i + 1
            
            logging.info(f"Hybrid reranker processed {len(documents)} documents, returning top {len(result_docs)}")
            return result_docs
        
        # Fallback to primary model if available
        if self.primary_model in self.rerankers:
            return self.rerankers[self.primary_model].rerank(query, documents, top_k)
        
        # Use any available reranker
        reranker = list(self.rerankers.values())[0]
        return reranker.rerank(query, documents, top_k)


class BasicReranker(BaseReranker):
    """
    Basic reranker that uses simple text similarity without external dependencies
    Fallback option when other rerankers are unavailable
    """
    
    def __init__(self):
        """Initialize basic reranker"""
        logging.info("BasicReranker initialized (no external dependencies)")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using basic text similarity
        
        Args:
            query: The search query
            documents: List of documents with 'page_content' and 'metadata'
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents with basic relevance scores
        """
        if not documents:
            return []
        
        scored_docs = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc in documents:
            try:
                content = doc.get('page_content', '') if isinstance(doc, dict) else str(doc.page_content)
                content_lower = content.lower()
                content_words = set(content_lower.split())
                
                # Simple word overlap score
                overlap = len(query_words.intersection(content_words))
                total_query_words = len(query_words)
                score = overlap / total_query_words if total_query_words > 0 else 0.0
                
                # Boost score if query appears as substring
                if query_lower in content_lower:
                    score += 0.5
                
                # Create scored document
                scored_doc = doc.copy() if isinstance(doc, dict) else {
                    'page_content': doc.page_content,
                    'metadata': getattr(doc, 'metadata', {})
                }
                scored_doc['metadata']['relevance_score'] = score
                scored_docs.append(scored_doc)
                
            except Exception as e:
                logging.warning(f"Error scoring document: {e}")
                continue
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x['metadata'].get('relevance_score', 0), reverse=True)
        
        return scored_docs[:top_k]


def create_reranker(reranker_type: str = "hybrid", **kwargs) -> BaseReranker:
    """
    Factory function to create appropriate reranker
    
    Args:
        reranker_type: Type of reranker ("flashrank", "cross_encoder", "hybrid")
        **kwargs: Additional arguments for reranker initialization
        
    Returns:
        Initialized reranker instance
    """
    if reranker_type == "flashrank":
        if not FLASHRANK_AVAILABLE:
            logging.warning("FlashRank not available, falling back to CrossEncoder")
            reranker_type = "cross_encoder"
        else:
            return FlashRankReranker(**kwargs)
    
    if reranker_type == "cross_encoder":
        if not CROSS_ENCODER_AVAILABLE:
            logging.warning("CrossEncoder not available, falling back to FlashRank")
            reranker_type = "flashrank"
        else:
            return CrossEncoderReranker(**kwargs)
    
    if reranker_type == "hybrid":
        try:
            return HybridReranker(**kwargs)
        except Exception as e:
            logging.warning(f"Hybrid reranker failed, falling back to basic: {e}")
            return BasicReranker()
    
    if reranker_type == "basic":
        return BasicReranker()
    
    # Fallback chain
    try:
        if FLASHRANK_AVAILABLE:
            return FlashRankReranker()
        elif CROSS_ENCODER_AVAILABLE:
            return CrossEncoderReranker()
        else:
            logging.warning("No advanced reranking libraries available, using basic reranker")
            return BasicReranker()
    except Exception as e:
        logging.warning(f"All advanced rerankers failed, using basic reranker: {e}")
        return BasicReranker()