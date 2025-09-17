"""
Graph Retriever for Hybrid RAG System

This module provides graph-based retrieval capabilities including:
- Graph traversal and exploration
- Multi-hop reasoning
- Entity-centric retrieval
- Relationship-aware search
- Subgraph extraction and ranking
- Path-based evidence collection
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import asyncio
from pathlib import Path
import heapq
from collections import defaultdict, deque
import math

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

from langchain.schema import Document
from .index_graph import GraphIndexer, GraphNode, GraphEdge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphPath:
    """Represents a path through the graph"""
    nodes: List[str]
    edges: List[str]
    total_weight: float
    path_type: str  # semantic, structural, hybrid
    explanation: str = ""

@dataclass
class GraphRetrievalResult:
    """Represents a graph-based retrieval result"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    subgraph: nx.DiGraph
    score: float
    retrieval_method: str
    explanation: str = ""
    paths: List[GraphPath] = field(default_factory=list)

class GraphTraverser:
    """Handles graph traversal algorithms"""

    def __init__(self, graph: nx.DiGraph, nodes: Dict[str, GraphNode], edges: Dict[str, GraphEdge]):
        self.graph = graph
        self.nodes = nodes
        self.edges = edges

    def bfs_traversal(self, start_nodes: List[str], max_depth: int = 3, max_nodes: int = 50) -> Set[str]:
        """Breadth-first search traversal from start nodes"""
        visited = set()
        queue = deque([(node, 0) for node in start_nodes if node in self.graph])

        while queue and len(visited) < max_nodes:
            current_node, depth = queue.popleft()

            if current_node in visited or depth > max_depth:
                continue

            visited.add(current_node)

            if depth < max_depth:
                # Add neighbors
                for neighbor in list(self.graph.successors(current_node)) + list(self.graph.predecessors(current_node)):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        return visited

    def dfs_traversal(self, start_nodes: List[str], max_depth: int = 3, max_nodes: int = 50) -> Set[str]:
        """Depth-first search traversal from start nodes"""
        visited = set()

        def dfs_recursive(node: str, depth: int):
            if node in visited or depth > max_depth or len(visited) >= max_nodes:
                return

            visited.add(node)

            if depth < max_depth:
                for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                    if neighbor not in visited:
                        dfs_recursive(neighbor, depth + 1)

        for start_node in start_nodes:
            if start_node in self.graph:
                dfs_recursive(start_node, 0)

        return visited

    def find_shortest_paths(self, source: str, targets: List[str], max_length: int = 5) -> Dict[str, GraphPath]:
        """Find shortest paths from source to multiple targets"""
        paths = {}

        for target in targets:
            if source not in self.graph or target not in self.graph:
                continue

            try:
                if nx.has_path(self.graph, source, target):
                    path_nodes = nx.shortest_path(self.graph, source, target)

                    if len(path_nodes) <= max_length + 1:  # +1 because path includes both endpoints
                        path_edges = []
                        total_weight = 0

                        for i in range(len(path_nodes) - 1):
                            edge_data = self.graph.get_edge_data(path_nodes[i], path_nodes[i + 1])
                            if edge_data:
                                # Handle multiple edges between same nodes
                                edge_info = list(edge_data.values())[0]
                                weight = edge_info.get('weight', 1.0)
                                total_weight += weight
                                path_edges.append(f"{path_nodes[i]}_{path_nodes[i + 1]}")

                        paths[target] = GraphPath(
                            nodes=path_nodes,
                            edges=path_edges,
                            total_weight=total_weight,
                            path_type="shortest",
                            explanation=f"Shortest path from {source} to {target}"
                        )

            except nx.NetworkXNoPath:
                continue

        return paths

    def find_all_simple_paths(self, source: str, target: str, max_length: int = 4) -> List[GraphPath]:
        """Find all simple paths between two nodes"""
        if source not in self.graph or target not in self.graph:
            return []

        paths = []
        try:
            simple_paths = nx.all_simple_paths(self.graph, source, target, cutoff=max_length)

            for path_nodes in simple_paths:
                path_edges = []
                total_weight = 0

                for i in range(len(path_nodes) - 1):
                    edge_data = self.graph.get_edge_data(path_nodes[i], path_nodes[i + 1])
                    if edge_data:
                        edge_info = list(edge_data.values())[0]
                        weight = edge_info.get('weight', 1.0)
                        total_weight += weight
                        path_edges.append(f"{path_nodes[i]}_{path_nodes[i + 1]}")

                paths.append(GraphPath(
                    nodes=path_nodes,
                    edges=path_edges,
                    total_weight=total_weight,
                    path_type="simple",
                    explanation=f"Simple path from {source} to {target}"
                ))

        except nx.NetworkXNoPath:
            pass

        return paths

    def find_semantic_paths(self, start_nodes: List[str], query_embedding: np.ndarray, max_depth: int = 3) -> List[GraphPath]:
        """Find paths based on semantic similarity"""
        semantic_paths = []

        for start_node in start_nodes:
            if start_node not in self.graph:
                continue

            # Use modified Dijkstra's algorithm with semantic similarity
            distances = {start_node: 0}
            previous = {start_node: None}
            unvisited = set(self.graph.nodes())

            while unvisited:
                # Find unvisited node with minimum distance
                current_node = min(
                    (node for node in unvisited if node in distances),
                    key=lambda x: distances[x],
                    default=None
                )

                if current_node is None or distances[current_node] > max_depth:
                    break

                unvisited.remove(current_node)

                # Check neighbors
                for neighbor in self.graph.successors(current_node):
                    if neighbor not in unvisited:
                        continue

                    # Compute semantic distance
                    neighbor_node = self.nodes.get(neighbor)
                    if neighbor_node and neighbor_node.embeddings is not None:
                        similarity = self._compute_cosine_similarity(query_embedding, neighbor_node.embeddings)
                        semantic_distance = 1 - similarity  # Convert similarity to distance
                    else:
                        semantic_distance = 1.0

                    edge_data = self.graph.get_edge_data(current_node, neighbor)
                    if edge_data:
                        edge_weight = list(edge_data.values())[0].get('weight', 1.0)
                    else:
                        edge_weight = 1.0

                    # Combine semantic and structural distances
                    combined_distance = distances[current_node] + (semantic_distance + edge_weight) / 2

                    if neighbor not in distances or combined_distance < distances[neighbor]:
                        distances[neighbor] = combined_distance
                        previous[neighbor] = current_node

            # Reconstruct paths to all reachable nodes
            for end_node in distances:
                if end_node != start_node and distances[end_node] <= max_depth:
                    path_nodes = []
                    current = end_node
                    while current is not None:
                        path_nodes.append(current)
                        current = previous[current]
                    path_nodes.reverse()

                    if len(path_nodes) > 1:
                        semantic_paths.append(GraphPath(
                            nodes=path_nodes,
                            edges=[],  # Simplified for semantic paths
                            total_weight=distances[end_node],
                            path_type="semantic",
                            explanation=f"Semantic path from {start_node} to {end_node}"
                        ))

        return semantic_paths

    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if emb1.size == 0 or emb2.size == 0:
            return 0.0

        dot_product = np.dot(emb1.flatten(), emb2.flatten())
        norm1 = np.linalg.norm(emb1.flatten())
        norm2 = np.linalg.norm(emb2.flatten())

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class GraphRetriever:
    """
    Main graph retriever that provides various graph-based retrieval methods
    """

    def __init__(
        self,
        graph_indexer: GraphIndexer,
        retrieval_strategies: List[str] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.5
    ):
        self.graph_indexer = graph_indexer
        self.graph = graph_indexer.graph
        self.nodes = graph_indexer.nodes
        self.edges = graph_indexer.edges

        self.traverser = GraphTraverser(self.graph, self.nodes, self.edges)

        self.retrieval_strategies = retrieval_strategies or [
            'node_similarity',
            'subgraph_expansion',
            'path_based',
            'entity_centric'
        ]

        self.max_results = max_results
        self.similarity_threshold = similarity_threshold

        logger.info(f"Initialized GraphRetriever with {len(self.nodes)} nodes and {len(self.edges)} edges")

    def retrieve_by_node_similarity(self, query: str, k: int = 10) -> GraphRetrievalResult:
        """Retrieve nodes based on direct similarity to query"""
        similar_nodes = self.graph_indexer.find_similar_nodes(
            query=query,
            k=k,
            node_types=['document', 'chunk', 'entity']
        )

        result_nodes = []
        result_edges = []
        subgraph = nx.DiGraph()

        for node_id, score in similar_nodes:
            if score >= self.similarity_threshold:
                node = self.nodes[node_id]
                result_nodes.append(node)
                subgraph.add_node(node_id, **node.__dict__)

                # Add immediate edges
                for neighbor in list(self.graph.successors(node_id)) + list(self.graph.predecessors(node_id)):
                    if neighbor in [n.id for n in result_nodes]:
                        for edge_key in self.graph.get_edge_data(node_id, neighbor, default={}):
                            edge_id = f"{node_id}_{neighbor}"
                            if edge_id in self.edges:
                                edge = self.edges[edge_id]
                                result_edges.append(edge)
                                subgraph.add_edge(node_id, neighbor, **edge.__dict__)

        overall_score = sum(score for _, score in similar_nodes[:len(result_nodes)]) / max(1, len(result_nodes))

        return GraphRetrievalResult(
            nodes=result_nodes,
            edges=result_edges,
            subgraph=subgraph,
            score=overall_score,
            retrieval_method='node_similarity',
            explanation=f"Retrieved {len(result_nodes)} nodes based on similarity to query"
        )

    def retrieve_by_subgraph_expansion(self, query: str, expansion_depth: int = 2, k: int = 5) -> GraphRetrievalResult:
        """Retrieve by expanding around most similar nodes"""
        # Find initial seed nodes
        similar_nodes = self.graph_indexer.find_similar_nodes(query=query, k=k)
        seed_nodes = [node_id for node_id, score in similar_nodes if score >= self.similarity_threshold]

        if not seed_nodes:
            return GraphRetrievalResult(
                nodes=[], edges=[], subgraph=nx.DiGraph(), score=0.0,
                retrieval_method='subgraph_expansion',
                explanation="No similar nodes found for expansion"
            )

        # Expand around seed nodes
        expanded_nodes = self.traverser.bfs_traversal(
            start_nodes=seed_nodes,
            max_depth=expansion_depth,
            max_nodes=self.max_results * 2
        )

        # Build subgraph
        subgraph = nx.DiGraph()
        result_nodes = []
        result_edges = []

        for node_id in expanded_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                result_nodes.append(node)
                subgraph.add_node(node_id, **node.__dict__)

        # Add edges between nodes in the subgraph
        for node_id in expanded_nodes:
            for neighbor in self.graph.successors(node_id):
                if neighbor in expanded_nodes:
                    edge_data = self.graph.get_edge_data(node_id, neighbor)
                    if edge_data:
                        edge_info = list(edge_data.values())[0]
                        edge_id = f"{node_id}_{neighbor}"
                        if edge_id in self.edges:
                            edge = self.edges[edge_id]
                            result_edges.append(edge)
                            subgraph.add_edge(node_id, neighbor, **edge.__dict__)

        # Compute overall score
        seed_scores = [score for _, score in similar_nodes[:len(seed_nodes)]]
        overall_score = sum(seed_scores) / max(1, len(seed_scores))

        return GraphRetrievalResult(
            nodes=result_nodes,
            edges=result_edges,
            subgraph=subgraph,
            score=overall_score,
            retrieval_method='subgraph_expansion',
            explanation=f"Expanded from {len(seed_nodes)} seed nodes to {len(result_nodes)} nodes"
        )

    def retrieve_by_path_based(self, query: str, max_path_length: int = 4, k: int = 5) -> GraphRetrievalResult:
        """Retrieve based on paths between relevant nodes"""
        # Find relevant nodes
        similar_nodes = self.graph_indexer.find_similar_nodes(query=query, k=k*2)
        relevant_nodes = [node_id for node_id, score in similar_nodes if score >= self.similarity_threshold]

        if len(relevant_nodes) < 2:
            return GraphRetrievalResult(
                nodes=[], edges=[], subgraph=nx.DiGraph(), score=0.0,
                retrieval_method='path_based',
                explanation="Insufficient relevant nodes for path-based retrieval"
            )

        # Find paths between relevant nodes
        all_paths = []
        subgraph = nx.DiGraph()
        result_nodes = set()
        result_edges = set()

        for i, source in enumerate(relevant_nodes):
            targets = relevant_nodes[i+1:i+6]  # Limit targets to avoid explosion
            paths = self.traverser.find_shortest_paths(source, targets, max_path_length)

            for target, path in paths.items():
                all_paths.append(path)

                # Add path nodes and edges to result
                for node_id in path.nodes:
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        result_nodes.add(node)
                        subgraph.add_node(node_id, **node.__dict__)

                for edge_id in path.edges:
                    if edge_id in self.edges:
                        edge = self.edges[edge_id]
                        result_edges.add(edge)
                        subgraph.add_edge(edge.source, edge.target, **edge.__dict__)

        # Compute overall score based on path weights and node similarities
        path_scores = [1.0 / (path.total_weight + 1) for path in all_paths]
        overall_score = sum(path_scores) / max(1, len(path_scores))

        return GraphRetrievalResult(
            nodes=list(result_nodes)[:self.max_results],
            edges=list(result_edges),
            subgraph=subgraph,
            score=overall_score,
            retrieval_method='path_based',
            explanation=f"Found {len(all_paths)} paths connecting relevant nodes",
            paths=all_paths
        )

    def retrieve_by_entity_centric(self, query: str, k: int = 10) -> GraphRetrievalResult:
        """Retrieve by focusing on entities and their relationships"""
        # Find relevant entities
        entity_nodes = self.graph_indexer.find_similar_nodes(
            query=query,
            k=k,
            node_types=['entity']
        )

        if not entity_nodes:
            return GraphRetrievalResult(
                nodes=[], edges=[], subgraph=nx.DiGraph(), score=0.0,
                retrieval_method='entity_centric',
                explanation="No relevant entities found"
            )

        subgraph = nx.DiGraph()
        result_nodes = []
        result_edges = []

        # For each relevant entity, get its immediate context
        for entity_id, entity_score in entity_nodes:
            if entity_score < self.similarity_threshold:
                continue

            entity_node = self.nodes[entity_id]
            result_nodes.append(entity_node)
            subgraph.add_node(entity_id, **entity_node.__dict__)

            # Get entity's document/chunk context
            for neighbor in list(self.graph.predecessors(entity_id)) + list(self.graph.successors(entity_id)):
                neighbor_node = self.nodes.get(neighbor)
                if neighbor_node and neighbor_node.node_type in ['document', 'chunk']:
                    result_nodes.append(neighbor_node)
                    subgraph.add_node(neighbor, **neighbor_node.__dict__)

                    # Add edges
                    if self.graph.has_edge(entity_id, neighbor):
                        edge_data = self.graph.get_edge_data(entity_id, neighbor)
                        edge_info = list(edge_data.values())[0]
                        edge_id = f"{entity_id}_{neighbor}"
                        if edge_id in self.edges:
                            edge = self.edges[edge_id]
                            result_edges.append(edge)
                            subgraph.add_edge(entity_id, neighbor, **edge.__dict__)

                    if self.graph.has_edge(neighbor, entity_id):
                        edge_data = self.graph.get_edge_data(neighbor, entity_id)
                        edge_info = list(edge_data.values())[0]
                        edge_id = f"{neighbor}_{entity_id}"
                        if edge_id in self.edges:
                            edge = self.edges[edge_id]
                            result_edges.append(edge)
                            subgraph.add_edge(neighbor, entity_id, **edge.__dict__)

        # Remove duplicates
        unique_nodes = []
        seen_ids = set()
        for node in result_nodes:
            if node.id not in seen_ids:
                unique_nodes.append(node)
                seen_ids.add(node.id)

        overall_score = sum(score for _, score in entity_nodes[:len(unique_nodes)]) / max(1, len(unique_nodes))

        return GraphRetrievalResult(
            nodes=unique_nodes[:self.max_results],
            edges=result_edges,
            subgraph=subgraph,
            score=overall_score,
            retrieval_method='entity_centric',
            explanation=f"Retrieved context around {len(entity_nodes)} relevant entities"
        )

    def retrieve_semantic_paths(self, query: str, max_depth: int = 3, k: int = 5) -> GraphRetrievalResult:
        """Retrieve using semantic similarity-guided traversal"""
        # Encode query
        query_embedding = self.graph_indexer.compute_embeddings([query])[0]

        # Find initial relevant nodes
        similar_nodes = self.graph_indexer.find_similar_nodes(query=query, k=k)
        start_nodes = [node_id for node_id, score in similar_nodes if score >= self.similarity_threshold]

        if not start_nodes:
            return GraphRetrievalResult(
                nodes=[], edges=[], subgraph=nx.DiGraph(), score=0.0,
                retrieval_method='semantic_paths',
                explanation="No starting nodes found for semantic traversal"
            )

        # Find semantic paths
        semantic_paths = self.traverser.find_semantic_paths(start_nodes, query_embedding, max_depth)

        # Build result from paths
        subgraph = nx.DiGraph()
        result_nodes = set()
        result_edges = set()

        for path in semantic_paths:
            for node_id in path.nodes:
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    result_nodes.add(node)
                    subgraph.add_node(node_id, **node.__dict__)

        # Add edges between nodes in paths
        for path in semantic_paths:
            for i in range(len(path.nodes) - 1):
                source, target = path.nodes[i], path.nodes[i + 1]
                if self.graph.has_edge(source, target):
                    edge_data = self.graph.get_edge_data(source, target)
                    edge_info = list(edge_data.values())[0]
                    edge_id = f"{source}_{target}"
                    if edge_id in self.edges:
                        edge = self.edges[edge_id]
                        result_edges.add(edge)
                        subgraph.add_edge(source, target, **edge.__dict__)

        # Compute overall score
        path_scores = [1.0 / (path.total_weight + 1) for path in semantic_paths]
        overall_score = sum(path_scores) / max(1, len(path_scores))

        return GraphRetrievalResult(
            nodes=list(result_nodes)[:self.max_results],
            edges=list(result_edges),
            subgraph=subgraph,
            score=overall_score,
            retrieval_method='semantic_paths',
            explanation=f"Found {len(semantic_paths)} semantic paths",
            paths=semantic_paths
        )

    def hybrid_retrieve(self, query: str, strategy_weights: Dict[str, float] = None) -> GraphRetrievalResult:
        """Combine multiple retrieval strategies"""
        weights = strategy_weights or {
            'node_similarity': 0.3,
            'subgraph_expansion': 0.25,
            'path_based': 0.25,
            'entity_centric': 0.2
        }

        # Run each retrieval strategy
        results = {}
        if 'node_similarity' in weights:
            results['node_similarity'] = self.retrieve_by_node_similarity(query)

        if 'subgraph_expansion' in weights:
            results['subgraph_expansion'] = self.retrieve_by_subgraph_expansion(query)

        if 'path_based' in weights:
            results['path_based'] = self.retrieve_by_path_based(query)

        if 'entity_centric' in weights:
            results['entity_centric'] = self.retrieve_by_entity_centric(query)

        # Combine results
        combined_nodes = {}
        combined_edges = {}
        combined_subgraph = nx.DiGraph()
        all_paths = []

        total_score = 0.0
        explanations = []

        for strategy, result in results.items():
            weight = weights.get(strategy, 0.0)
            total_score += result.score * weight

            explanations.append(f"{strategy}: {result.score:.3f}")

            # Merge nodes with weighted scoring
            for node in result.nodes:
                if node.id in combined_nodes:
                    # Average the scores (could use other fusion methods)
                    combined_nodes[node.id] = (combined_nodes[node.id] + result.score * weight) / 2
                else:
                    combined_nodes[node.id] = result.score * weight

            # Merge edges
            for edge in result.edges:
                edge_key = f"{edge.source}_{edge.target}_{edge.edge_type}"
                combined_edges[edge_key] = edge

            # Merge subgraphs
            combined_subgraph = nx.compose(combined_subgraph, result.subgraph)

            # Collect paths
            all_paths.extend(result.paths)

        # Convert to final result format
        final_nodes = []
        for node_id, score in sorted(combined_nodes.items(), key=lambda x: x[1], reverse=True):
            if node_id in self.nodes:
                final_nodes.append(self.nodes[node_id])

        final_edges = list(combined_edges.values())

        return GraphRetrievalResult(
            nodes=final_nodes[:self.max_results],
            edges=final_edges,
            subgraph=combined_subgraph,
            score=total_score,
            retrieval_method='hybrid',
            explanation=f"Hybrid retrieval combining: {', '.join(explanations)}",
            paths=all_paths
        )

    def retrieve(self, query: str, method: str = 'hybrid', **kwargs) -> GraphRetrievalResult:
        """Main retrieval method that dispatches to specific strategies"""
        if method == 'node_similarity':
            return self.retrieve_by_node_similarity(query, **kwargs)
        elif method == 'subgraph_expansion':
            return self.retrieve_by_subgraph_expansion(query, **kwargs)
        elif method == 'path_based':
            return self.retrieve_by_path_based(query, **kwargs)
        elif method == 'entity_centric':
            return self.retrieve_by_entity_centric(query, **kwargs)
        elif method == 'semantic_paths':
            return self.retrieve_semantic_paths(query, **kwargs)
        elif method == 'hybrid':
            return self.hybrid_retrieve(query, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

    def explain_retrieval(self, result: GraphRetrievalResult) -> Dict[str, Any]:
        """Provide detailed explanation of retrieval result"""
        explanation = {
            'method': result.retrieval_method,
            'score': result.score,
            'num_nodes': len(result.nodes),
            'num_edges': len(result.edges),
            'node_types': {},
            'edge_types': {},
            'paths': []
        }

        # Analyze node types
        for node in result.nodes:
            node_type = node.node_type
            explanation['node_types'][node_type] = explanation['node_types'].get(node_type, 0) + 1

        # Analyze edge types
        for edge in result.edges:
            edge_type = edge.edge_type
            explanation['edge_types'][edge_type] = explanation['edge_types'].get(edge_type, 0) + 1

        # Include path information
        for path in result.paths:
            explanation['paths'].append({
                'length': len(path.nodes),
                'weight': path.total_weight,
                'type': path.path_type,
                'explanation': path.explanation
            })

        return explanation

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever's capabilities"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'graph_connected_components': nx.number_connected_components(self.graph.to_undirected()),
            'available_strategies': self.retrieval_strategies,
            'similarity_threshold': self.similarity_threshold,
            'max_results': self.max_results
        }


# Example usage and testing
if __name__ == "__main__":
    from .index_graph import GraphIndexer
    from langchain.schema import Document

    # Initialize graph indexer
    indexer = GraphIndexer()

    # Sample documents
    docs = [
        Document(
            page_content="Apple Inc. is a technology company founded by Steve Jobs.",
            metadata={"source": "tech.txt"}
        ),
        Document(
            page_content="Steve Jobs was a visionary leader in the technology industry.",
            metadata={"source": "leaders.txt"}
        )
    ]

    # Index documents
    indexer.index_documents(docs)

    # Initialize graph retriever
    retriever = GraphRetriever(indexer)

    # Test different retrieval methods
    query = "technology company founded by Steve Jobs"

    print("Testing Node Similarity Retrieval:")
    result1 = retriever.retrieve(query, method='node_similarity')
    print(f"Found {len(result1.nodes)} nodes with score {result1.score:.3f}")

    print("\nTesting Hybrid Retrieval:")
    result2 = retriever.retrieve(query, method='hybrid')
    print(f"Found {len(result2.nodes)} nodes with score {result2.score:.3f}")

    # Explain results
    explanation = retriever.explain_retrieval(result2)
    print(f"\nRetrieval explanation: {explanation}")

    # Get stats
    stats = retriever.get_retrieval_stats()
    print(f"\nRetriever stats: {stats}")