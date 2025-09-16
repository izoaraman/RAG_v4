"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) Implementation

This module implements RAPTOR for hierarchical document retrieval in RAG systems.
RAPTOR creates a tree structure of document summaries at different abstraction levels,
enabling both detailed and high-level information retrieval.

Reference: https://arxiv.org/pdf/2401.18059.pdf
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

try:
    import umap
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    logging.warning("Clustering libraries not available. Install with: pip install umap-learn scikit-learn")

try:
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain.text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available for RAPTOR implementation")

from .load_config import LoadConfig

# Global configuration
APPCFG = LoadConfig()
RANDOM_SEED = 42


class BaseRAPTOR(ABC):
    """Abstract base class for RAPTOR implementations"""
    
    @abstractmethod
    def build_tree(self, texts: List[str]) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Build RAPTOR tree structure"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using RAPTOR tree"""
        pass


class RAPTORClustering:
    """
    Clustering utilities for RAPTOR tree construction
    Implements GMM-based clustering with UMAP dimensionality reduction
    """
    
    @staticmethod
    def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Apply global dimensionality reduction using UMAP
        
        Args:
            embeddings: High-dimensional embedding vectors
            dim: Target dimensionality
            n_neighbors: Number of neighbors for UMAP (auto-calculated if None)
            metric: Distance metric for UMAP
            
        Returns:
            Reduced dimensionality embeddings
        """
        if not CLUSTERING_AVAILABLE:
            raise ImportError("Clustering libraries not available")
            
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)

        return umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=dim, 
            metric=metric,
            random_state=RANDOM_SEED
        ).fit_transform(embeddings)

    @staticmethod
    def local_cluster_embeddings(
        embeddings: np.ndarray, 
        dim: int, 
        num_neighbors: int = 10, 
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Apply local dimensionality reduction using UMAP
        
        Args:
            embeddings: High-dimensional embedding vectors
            dim: Target dimensionality
            num_neighbors: Number of neighbors for UMAP
            metric: Distance metric for UMAP
            
        Returns:
            Reduced dimensionality embeddings
        """
        if not CLUSTERING_AVAILABLE:
            raise ImportError("Clustering libraries not available")
            
        return umap.UMAP(
            n_neighbors=num_neighbors, 
            n_components=dim, 
            metric=metric,
            random_state=RANDOM_SEED
        ).fit_transform(embeddings)

    @staticmethod
    def get_optimal_clusters(
        embeddings: np.ndarray, 
        max_clusters: int = 50, 
        random_state: int = RANDOM_SEED
    ) -> int:
        """
        Find optimal number of clusters using BIC score
        
        Args:
            embeddings: Embedding vectors to cluster
            max_clusters: Maximum number of clusters to test
            random_state: Random seed for reproducibility
            
        Returns:
            Optimal number of clusters
        """
        if not CLUSTERING_AVAILABLE:
            raise ImportError("Clustering libraries not available")
            
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)

        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))

        return n_clusters[np.argmin(bics)]

    @staticmethod
    def gmm_cluster(embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED):
        """
        Perform GMM clustering with probability threshold
        
        Args:
            embeddings: Embedding vectors to cluster
            threshold: Probability threshold for cluster assignment
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (cluster_labels, n_clusters)
        """
        if not CLUSTERING_AVAILABLE:
            raise ImportError("Clustering libraries not available")
            
        n_clusters = RAPTORClustering.get_optimal_clusters(embeddings)
        
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        
        return labels, n_clusters

    @staticmethod
    def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
    ) -> List[np.ndarray]:
        """
        Perform hierarchical clustering: global reduction -> global clustering -> local clustering
        
        Args:
            embeddings: Embedding vectors to cluster
            dim: Target dimensionality for UMAP
            threshold: Probability threshold for GMM clustering
            
        Returns:
            List of local cluster labels for each embedding
        """
        if not CLUSTERING_AVAILABLE:
            raise ImportError("Clustering libraries not available")
            
        if len(embeddings) <= dim + 1:
            return [np.array([0]) for _ in range(len(embeddings))]

        # Global dimensionality reduction
        reduced_embeddings_global = RAPTORClustering.global_cluster_embeddings(embeddings, dim)
        
        # Global clustering
        global_clusters, n_global_clusters = RAPTORClustering.gmm_cluster(
            reduced_embeddings_global, threshold
        )

        # Initialize local clusters
        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Process each global cluster
        for i in range(n_global_clusters):
            # Extract embeddings belonging to current global cluster
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
                
            if len(global_cluster_embeddings_) <= dim + 1:
                # Small clusters: direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = RAPTORClustering.local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = RAPTORClustering.gmm_cluster(
                    reduced_embeddings_local, threshold
                )

            # Assign local cluster IDs
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters


class RAPTORTreeBuilder:
    """
    RAPTOR tree builder that creates hierarchical document representations
    """
    
    def __init__(self, llm, embeddings, summarization_prompt: Optional[str] = None):
        """
        Initialize RAPTOR tree builder
        
        Args:
            llm: Language model for summarization
            embeddings: Embedding model for vector generation
            summarization_prompt: Custom prompt for summarization
        """
        self.llm = llm
        self.embeddings = embeddings
        
        # Default summarization prompt
        if summarization_prompt is None:
            self.summarization_prompt = """여기 문서들의 일부가 있습니다.

아래 문서들의 핵심 내용을 간결하고 포괄적으로 요약해주세요.
중요한 정보, 주요 개념, 핵심 아이디어를 포함하되 불필요한 세부사항은 제외하세요.

문서:
{context}

요약:"""
        else:
            self.summarization_prompt = summarization_prompt
        
        if LANGCHAIN_AVAILABLE:
            self.prompt_template = ChatPromptTemplate.from_template(self.summarization_prompt)
            self.chain = self.prompt_template | self.llm | StrOutputParser()
        else:
            self.prompt_template = None
            self.chain = None
        
        logging.info("RAPTORTreeBuilder initialized")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text list
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        text_embeddings = self.embeddings.embed_documents(texts)
        return np.array(text_embeddings)

    def format_texts(self, df: pd.DataFrame) -> str:
        """
        Format DataFrame texts into single string
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            Formatted text string
        """
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    def embed_cluster_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Embed and cluster texts
        
        Args:
            texts: List of texts to process
            
        Returns:
            DataFrame with text, embeddings, and cluster assignments
        """
        # Generate embeddings
        text_embeddings_np = self.embed_texts(texts)
        
        # Perform clustering
        cluster_labels = RAPTORClustering.perform_clustering(text_embeddings_np, 10, 0.1)
        
        # Create DataFrame
        df = pd.DataFrame()
        df["text"] = texts
        df["embd"] = list(text_embeddings_np)
        df["cluster"] = cluster_labels
        
        return df

    def embed_cluster_summarize_texts(
        self, texts: List[str], level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Embed, cluster, and summarize texts
        
        Args:
            texts: List of texts to process
            level: Processing level for tracking
            
        Returns:
            Tuple of (clusters_df, summary_df)
        """
        # Embed and cluster texts
        df_clusters = self.embed_cluster_texts(texts)

        # Expand DataFrame for easier processing
        expanded_list = []
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append({
                    "text": row["text"],
                    "embd": row["embd"],
                    "cluster": cluster
                })

        expanded_df = pd.DataFrame(expanded_list)
        all_clusters = expanded_df["cluster"].unique()

        logging.info(f"Generated {len(all_clusters)} clusters at level {level}")

        # Summarize each cluster
        summaries = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self.format_texts(df_cluster)
            if self.chain is not None:
                summary = self.chain.invoke({"context": formatted_txt})
            else:
                # Fallback when LangChain is not available
                context_preview = formatted_txt[:200] + "..." if len(formatted_txt) > 200 else formatted_txt
                summary = f"Mock RAPTOR summary for cluster {i}: {context_preview}"
            summaries.append(summary)

        # Create summary DataFrame
        df_summary = pd.DataFrame({
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        })

        return df_clusters, df_summary

    def recursive_embed_cluster_summarize(
        self, texts: List[str], level: int = 1, n_levels: int = 3
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Recursively build RAPTOR tree through clustering and summarization
        
        Args:
            texts: Initial text documents
            level: Current processing level
            n_levels: Maximum number of levels
            
        Returns:
            Dictionary mapping levels to (clusters_df, summary_df) tuples
        """
        results = {}

        # Process current level
        df_clusters, df_summary = self.embed_cluster_summarize_texts(texts, level)
        results[level] = (df_clusters, df_summary)

        # Check for recursion conditions
        unique_clusters = df_summary["cluster"].nunique()

        if level < n_levels and unique_clusters > 1:
            # Use summaries as input for next level
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )
            results.update(next_level_results)

        return results


class RAPTORRetriever(BaseRAPTOR):
    """
    RAPTOR-based retriever that uses hierarchical tree structure for document retrieval
    """
    
    def __init__(self, 
                 llm,
                 embeddings,
                 persist_directory: str,
                 chunk_size: int = 100,
                 chunk_overlap: int = 0,
                 n_levels: int = 3):
        """
        Initialize RAPTOR retriever
        
        Args:
            llm: Language model for summarization
            embeddings: Embedding model
            persist_directory: Directory to persist vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            n_levels: Number of tree levels
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available for RAPTOR")
            
        self.llm = llm
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_levels = n_levels
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Initialize tree builder
        self.tree_builder = RAPTORTreeBuilder(llm, embeddings)
        
        # Initialize vector store
        self.vectorstore = None
        self.tree_results = None
        
        logging.info(f"RAPTORRetriever initialized with {n_levels} levels")

    def build_tree(self, texts: List[str]) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Build RAPTOR tree from input texts
        
        Args:
            texts: List of input documents/texts
            
        Returns:
            Dictionary mapping levels to processing results
        """
        # Split texts into chunks
        if isinstance(texts, str):
            texts = [texts]
        
        all_text = "\n\n\n --- \n\n\n".join(texts)
        leaf_texts = self.text_splitter.split_text(all_text)
        
        logging.info(f"Split {len(texts)} documents into {len(leaf_texts)} chunks")
        
        # Build tree structure
        self.tree_results = self.tree_builder.recursive_embed_cluster_summarize(
            leaf_texts, level=1, n_levels=self.n_levels
        )
        
        # Create comprehensive text collection for vector store
        all_texts = leaf_texts.copy()
        
        # Add summaries from each level
        for level in sorted(self.tree_results.keys()):
            summaries = self.tree_results[level][1]["summaries"].tolist()
            all_texts.extend(summaries)
        
        # Build vector store
        self.vectorstore = Chroma.from_texts(
            texts=all_texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        logging.info(f"Built RAPTOR tree with {len(all_texts)} total text segments")
        return self.tree_results

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using RAPTOR tree structure
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if self.vectorstore is None:
            raise RuntimeError("RAPTOR tree not built. Call build_tree() first.")
        
        # Retrieve from vector store
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        
        # Enhance with RAPTOR metadata
        enhanced_docs = []
        for i, doc in enumerate(docs):
            # Add RAPTOR-specific metadata
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "raptor_retrieval": True,
                    "retrieval_rank": i + 1,
                    "retrieval_method": "RAPTOR"
                }
            )
            enhanced_docs.append(enhanced_doc)
        
        logging.info(f"RAPTOR retrieved {len(enhanced_docs)} documents for query")
        return enhanced_docs

    def get_tree_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the built RAPTOR tree
        
        Returns:
            Dictionary with tree statistics
        """
        if self.tree_results is None:
            return {"error": "Tree not built"}
        
        summary = {
            "n_levels": len(self.tree_results),
            "levels": {}
        }
        
        for level, (clusters_df, summary_df) in self.tree_results.items():
            summary["levels"][level] = {
                "n_clusters": len(summary_df),
                "n_documents": len(clusters_df),
                "avg_cluster_size": len(clusters_df) / len(summary_df) if len(summary_df) > 0 else 0
            }
        
        return summary


def create_raptor_retriever(
    llm,
    embeddings,
    texts: List[str],
    persist_directory: str,
    **kwargs
) -> RAPTORRetriever:
    """
    Factory function to create and build RAPTOR retriever
    
    Args:
        llm: Language model for summarization
        embeddings: Embedding model
        texts: Input texts to build tree from
        persist_directory: Directory to persist vector store
        **kwargs: Additional arguments for RAPTORRetriever
        
    Returns:
        Built RAPTOR retriever
    """
    retriever = RAPTORRetriever(
        llm=llm,
        embeddings=embeddings,
        persist_directory=persist_directory,
        **kwargs
    )
    
    retriever.build_tree(texts)
    return retriever