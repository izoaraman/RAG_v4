"""
Graph Indexing Module for Hybrid RAG System

This module provides comprehensive graph-based indexing capabilities including:
- Document relationship extraction
- Entity recognition and linking
- Knowledge graph construction
- Graph embeddings generation
- Hierarchical document structure analysis
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import asyncio
from pathlib import Path
import hashlib
import pickle

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
import spacy
from spacy import displacy
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.graph_vectorstores.networkx import NetworkXVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    content: str
    node_type: str  # document, entity, concept, chunk
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    relationships: Set[str] = field(default_factory=set)

@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph"""
    source: str
    target: str
    edge_type: str  # contains, relates_to, similar_to, references
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class GraphIndexer:
    """
    Main graph indexing class that handles document processing,
    entity extraction, relationship building, and graph construction.
    """

    def __init__(
        self,
        store_path: str = "hybrid_rag/stores/graph_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        entity_model: str = "en_core_web_sm",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7
    ):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model)
        self.entity_model = spacy.load(entity_model)

        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Graph components
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.similarity_threshold = similarity_threshold

        # Embedding index for similarity search
        self.embedding_dimension = None
        self.faiss_index = None
        self.node_id_to_index = {}
        self.index_to_node_id = {}

        # Cache for expensive operations
        self.entity_cache = {}
        self.embedding_cache = {}

        logger.info(f"Initialized GraphIndexer with store path: {self.store_path}")

    def generate_node_id(self, content: str, node_type: str) -> str:
        """Generate unique node ID based on content and type"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{node_type}_{content_hash}"

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using spaCy"""
        if text in self.entity_cache:
            return self.entity_cache[text]

        doc = self.entity_model(text)
        entities = []

        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            }
            entities.append(entity_info)

        self.entity_cache[text] = entities
        return entities

    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities in text"""
        doc = self.entity_model(text)
        relationships = []

        # Simple dependency parsing based relationships
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj'] and token.head.pos_ == 'VERB':
                relationships.append({
                    'subject': token.text,
                    'predicate': token.head.text,
                    'object': [child.text for child in token.head.children
                              if child.dep_ in ['dobj', 'attr']],
                    'confidence': 0.8
                })

        # Entity co-occurrence relationships
        entities = [ent.text for ent in doc.ents]
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                relationships.append({
                    'subject': ent1,
                    'predicate': 'co_occurs_with',
                    'object': ent2,
                    'confidence': 0.6
                })

        return relationships

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts"""
        cache_key = hashlib.md5('|'.join(texts).encode()).hexdigest()
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        self.embedding_cache[cache_key] = embeddings
        return embeddings

    def add_document_node(self, document: Document) -> str:
        """Add a document node to the graph"""
        doc_id = self.generate_node_id(document.page_content, "document")

        # Compute embedding for the full document
        embedding = self.compute_embeddings([document.page_content])[0]

        node = GraphNode(
            id=doc_id,
            content=document.page_content,
            node_type="document",
            metadata=document.metadata,
            embeddings=embedding
        )

        self.nodes[doc_id] = node
        self.graph.add_node(doc_id, **node.__dict__)

        logger.debug(f"Added document node: {doc_id}")
        return doc_id

    def add_chunk_nodes(self, document: Document, doc_node_id: str) -> List[str]:
        """Split document into chunks and add as nodes"""
        chunks = self.text_splitter.split_documents([document])
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            chunk_id = self.generate_node_id(chunk.page_content, f"chunk_{i}")

            # Compute embedding for chunk
            embedding = self.compute_embeddings([chunk.page_content])[0]

            chunk_node = GraphNode(
                id=chunk_id,
                content=chunk.page_content,
                node_type="chunk",
                metadata={**chunk.metadata, 'chunk_index': i, 'parent_doc': doc_node_id},
                embeddings=embedding
            )

            self.nodes[chunk_id] = chunk_node
            self.graph.add_node(chunk_id, **chunk_node.__dict__)

            # Add edge from document to chunk
            edge = GraphEdge(
                source=doc_node_id,
                target=chunk_id,
                edge_type="contains",
                weight=1.0,
                metadata={'chunk_index': i}
            )

            edge_id = f"{doc_node_id}_{chunk_id}_contains"
            self.edges[edge_id] = edge
            self.graph.add_edge(doc_node_id, chunk_id, **edge.__dict__)

            chunk_ids.append(chunk_id)

        logger.debug(f"Added {len(chunk_ids)} chunk nodes for document {doc_node_id}")
        return chunk_ids

    def add_entity_nodes(self, text: str, parent_node_id: str) -> List[str]:
        """Extract entities and add them as nodes"""
        entities = self.extract_entities(text)
        entity_ids = []

        for entity in entities:
            entity_id = self.generate_node_id(entity['text'], "entity")

            # Skip if entity already exists
            if entity_id in self.nodes:
                # Add relationship to parent
                edge = GraphEdge(
                    source=parent_node_id,
                    target=entity_id,
                    edge_type="mentions",
                    weight=0.8
                )
                edge_id = f"{parent_node_id}_{entity_id}_mentions"
                if edge_id not in self.edges:
                    self.edges[edge_id] = edge
                    self.graph.add_edge(parent_node_id, entity_id, **edge.__dict__)
                continue

            # Compute embedding for entity
            embedding = self.compute_embeddings([entity['text']])[0]

            entity_node = GraphNode(
                id=entity_id,
                content=entity['text'],
                node_type="entity",
                metadata={
                    'entity_type': entity['label'],
                    'description': entity['description'],
                    'start': entity['start'],
                    'end': entity['end']
                },
                embeddings=embedding
            )

            self.nodes[entity_id] = entity_node
            self.graph.add_node(entity_id, **entity_node.__dict__)

            # Add edge from parent to entity
            edge = GraphEdge(
                source=parent_node_id,
                target=entity_id,
                edge_type="mentions",
                weight=0.8
            )

            edge_id = f"{parent_node_id}_{entity_id}_mentions"
            self.edges[edge_id] = edge
            self.graph.add_edge(parent_node_id, entity_id, **edge.__dict__)

            entity_ids.append(entity_id)

        logger.debug(f"Added {len(entity_ids)} entity nodes for {parent_node_id}")
        return entity_ids

    def build_similarity_edges(self):
        """Build similarity edges between nodes based on embedding similarity"""
        logger.info("Building similarity edges...")

        # Collect all embeddings
        node_ids = []
        embeddings = []

        for node_id, node in self.nodes.items():
            if node.embeddings is not None:
                node_ids.append(node_id)
                embeddings.append(node.embeddings)

        if len(embeddings) < 2:
            logger.warning("Not enough embeddings to build similarity index")
            return

        embeddings_array = np.vstack(embeddings)

        # Build FAISS index for efficient similarity search
        self.embedding_dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.faiss_index.add(embeddings_array)

        # Create mapping dictionaries
        self.node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        self.index_to_node_id = {i: node_id for i, node_id in enumerate(node_ids)}

        # Find similar nodes and create edges
        similarity_count = 0
        for i, node_id in enumerate(node_ids):
            # Search for similar nodes
            query_embedding = embeddings_array[i:i+1]  # Shape (1, dimension)
            scores, indices = self.faiss_index.search(query_embedding, k=min(10, len(node_ids)))

            for score, idx in zip(scores[0], indices[0]):
                if idx == i or score < self.similarity_threshold:
                    continue

                similar_node_id = self.index_to_node_id[idx]

                # Don't create similarity edges between parent-child nodes
                if (self.graph.has_edge(node_id, similar_node_id) or
                    self.graph.has_edge(similar_node_id, node_id)):
                    continue

                edge = GraphEdge(
                    source=node_id,
                    target=similar_node_id,
                    edge_type="similar_to",
                    weight=float(score),
                    metadata={'similarity_score': float(score)}
                )

                edge_id = f"{node_id}_{similar_node_id}_similar"
                if edge_id not in self.edges:
                    self.edges[edge_id] = edge
                    self.graph.add_edge(node_id, similar_node_id, **edge.__dict__)
                    similarity_count += 1

        logger.info(f"Created {similarity_count} similarity edges")

    def build_relationship_edges(self):
        """Build edges based on extracted relationships"""
        logger.info("Building relationship edges...")

        relationship_count = 0
        for node_id, node in self.nodes.items():
            if node.node_type in ['document', 'chunk']:
                relationships = self.extract_relationships(node.content)

                for rel in relationships:
                    subject = rel['subject']
                    predicate = rel['predicate']
                    objects = rel.get('object', [])

                    # Find nodes that match the subject and objects
                    subject_nodes = [nid for nid, n in self.nodes.items()
                                   if n.node_type == 'entity' and subject.lower() in n.content.lower()]

                    for obj in objects:
                        if isinstance(obj, str):
                            object_nodes = [nid for nid, n in self.nodes.items()
                                          if n.node_type == 'entity' and obj.lower() in n.content.lower()]

                            # Create edges between matching entities
                            for subj_node in subject_nodes:
                                for obj_node in object_nodes:
                                    if subj_node != obj_node:
                                        edge = GraphEdge(
                                            source=subj_node,
                                            target=obj_node,
                                            edge_type=predicate,
                                            weight=rel.get('confidence', 0.5),
                                            metadata={'extracted_from': node_id}
                                        )

                                        edge_id = f"{subj_node}_{obj_node}_{predicate}"
                                        if edge_id not in self.edges:
                                            self.edges[edge_id] = edge
                                            self.graph.add_edge(subj_node, obj_node, **edge.__dict__)
                                            relationship_count += 1

        logger.info(f"Created {relationship_count} relationship edges")

    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Index a list of documents into the graph"""
        logger.info(f"Starting to index {len(documents)} documents")

        stats = {
            'documents': 0,
            'chunks': 0,
            'entities': 0,
            'edges': 0
        }

        for doc in documents:
            # Add main document node
            doc_node_id = self.add_document_node(doc)
            stats['documents'] += 1

            # Add chunk nodes
            chunk_ids = self.add_chunk_nodes(doc, doc_node_id)
            stats['chunks'] += len(chunk_ids)

            # Add entity nodes for document
            doc_entity_ids = self.add_entity_nodes(doc.page_content, doc_node_id)
            stats['entities'] += len(doc_entity_ids)

            # Add entity nodes for each chunk
            for chunk_id in chunk_ids:
                chunk = self.nodes[chunk_id]
                chunk_entity_ids = self.add_entity_nodes(chunk.content, chunk_id)
                stats['entities'] += len(chunk_entity_ids)

        # Build relationships and similarity edges
        self.build_relationship_edges()
        self.build_similarity_edges()

        stats['edges'] = len(self.edges)

        logger.info(f"Indexing complete. Stats: {stats}")
        return stats

    def find_similar_nodes(self, query: str, k: int = 10, node_types: List[str] = None) -> List[Tuple[str, float]]:
        """Find nodes similar to the given query"""
        if self.faiss_index is None:
            logger.warning("FAISS index not built. Building similarity edges first.")
            self.build_similarity_edges()

        if self.faiss_index is None:
            return []

        # Compute query embedding
        query_embedding = self.compute_embeddings([query])[0]
        query_embedding = query_embedding.reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search for similar nodes
        scores, indices = self.faiss_index.search(query_embedding, k=min(k*2, len(self.node_id_to_index)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue

            node_id = self.index_to_node_id[idx]
            node = self.nodes[node_id]

            # Filter by node type if specified
            if node_types and node.node_type not in node_types:
                continue

            results.append((node_id, float(score)))

            if len(results) >= k:
                break

        return results

    def get_node_neighbors(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get neighbors of a node up to max_depth"""
        if node_id not in self.graph:
            return {}

        neighbors = {'nodes': set(), 'edges': set()}

        # BFS to find neighbors
        queue = [(node_id, 0)]
        visited = {node_id}

        while queue:
            current_node, depth = queue.pop(0)

            if depth < max_depth:
                # Get successors and predecessors
                for neighbor in list(self.graph.successors(current_node)) + list(self.graph.predecessors(current_node)):
                    if neighbor not in visited:
                        neighbors['nodes'].add(neighbor)
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

                    # Add edges
                    if self.graph.has_edge(current_node, neighbor):
                        neighbors['edges'].add(f"{current_node}_{neighbor}")
                    if self.graph.has_edge(neighbor, current_node):
                        neighbors['edges'].add(f"{neighbor}_{current_node}")

        return {
            'nodes': [self.nodes[nid] for nid in neighbors['nodes'] if nid in self.nodes],
            'edges': [self.edges[eid] for eid in neighbors['edges'] if eid in self.edges]
        }

    def save_index(self):
        """Save the graph index to disk"""
        logger.info(f"Saving graph index to {self.store_path}")

        # Save NetworkX graph
        nx.write_gpickle(self.graph, self.store_path / "graph.gpickle")

        # Save nodes and edges
        with open(self.store_path / "nodes.json", 'w') as f:
            nodes_data = {}
            for node_id, node in self.nodes.items():
                node_data = node.__dict__.copy()
                # Convert numpy array to list for JSON serialization
                if node_data['embeddings'] is not None:
                    node_data['embeddings'] = node_data['embeddings'].tolist()
                nodes_data[node_id] = node_data
            json.dump(nodes_data, f, indent=2)

        with open(self.store_path / "edges.json", 'w') as f:
            json.dump({eid: edge.__dict__ for eid, edge in self.edges.items()}, f, indent=2)

        # Save FAISS index if exists
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.store_path / "faiss_index.idx"))

            # Save index mappings
            with open(self.store_path / "index_mappings.json", 'w') as f:
                json.dump({
                    'node_id_to_index': self.node_id_to_index,
                    'index_to_node_id': self.index_to_node_id,
                    'embedding_dimension': self.embedding_dimension
                }, f, indent=2)

        # Save caches
        with open(self.store_path / "caches.pickle", 'wb') as f:
            pickle.dump({
                'entity_cache': self.entity_cache,
                'embedding_cache': self.embedding_cache
            }, f)

        logger.info("Graph index saved successfully")

    def load_index(self):
        """Load the graph index from disk"""
        logger.info(f"Loading graph index from {self.store_path}")

        try:
            # Load NetworkX graph
            if (self.store_path / "graph.gpickle").exists():
                self.graph = nx.read_gpickle(self.store_path / "graph.gpickle")

            # Load nodes
            if (self.store_path / "nodes.json").exists():
                with open(self.store_path / "nodes.json", 'r') as f:
                    nodes_data = json.load(f)

                self.nodes = {}
                for node_id, node_data in nodes_data.items():
                    # Convert embeddings back to numpy array
                    if node_data['embeddings'] is not None:
                        node_data['embeddings'] = np.array(node_data['embeddings'])

                    self.nodes[node_id] = GraphNode(**node_data)

            # Load edges
            if (self.store_path / "edges.json").exists():
                with open(self.store_path / "edges.json", 'r') as f:
                    edges_data = json.load(f)

                self.edges = {eid: GraphEdge(**edge_data) for eid, edge_data in edges_data.items()}

            # Load FAISS index
            faiss_index_path = self.store_path / "faiss_index.idx"
            mappings_path = self.store_path / "index_mappings.json"

            if faiss_index_path.exists() and mappings_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_index_path))

                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                    self.node_id_to_index = mappings['node_id_to_index']
                    self.index_to_node_id = {int(k): v for k, v in mappings['index_to_node_id'].items()}
                    self.embedding_dimension = mappings['embedding_dimension']

            # Load caches
            caches_path = self.store_path / "caches.pickle"
            if caches_path.exists():
                with open(caches_path, 'rb') as f:
                    caches = pickle.load(f)
                    self.entity_cache = caches.get('entity_cache', {})
                    self.embedding_cache = caches.get('embedding_cache', {})

            logger.info(f"Graph index loaded successfully. Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")
            return True

        except Exception as e:
            logger.error(f"Error loading graph index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph index"""
        node_types = {}
        edge_types = {}

        for node in self.nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        for edge in self.edges.values():
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': node_types,
            'edge_types': edge_types,
            'graph_connected_components': nx.number_connected_components(self.graph.to_undirected()),
            'has_faiss_index': self.faiss_index is not None,
            'embedding_dimension': self.embedding_dimension
        }

    def visualize_subgraph(self, center_node_id: str, max_depth: int = 2, output_path: str = None):
        """Visualize a subgraph around a center node"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            logger.error("Matplotlib not available for visualization")
            return None

        # Get subgraph
        neighbors = self.get_node_neighbors(center_node_id, max_depth)

        # Create subgraph
        subgraph = nx.DiGraph()

        # Add center node
        center_node = self.nodes[center_node_id]
        subgraph.add_node(center_node_id, **center_node.__dict__)

        # Add neighbor nodes
        for node in neighbors['nodes']:
            subgraph.add_node(node.id, **node.__dict__)

        # Add edges
        for edge in neighbors['edges']:
            if edge.source in subgraph and edge.target in subgraph:
                subgraph.add_edge(edge.source, edge.target, **edge.__dict__)

        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=3, iterations=50)

        # Draw nodes with different colors for different types
        node_colors = {
            'document': 'lightblue',
            'chunk': 'lightgreen',
            'entity': 'lightcoral',
            'concept': 'lightyellow'
        }

        for node_type, color in node_colors.items():
            nodes = [n for n, d in subgraph.nodes(data=True) if d.get('node_type') == node_type]
            nx.draw_networkx_nodes(subgraph, pos, nodelist=nodes,
                                 node_color=color, node_size=1000, alpha=0.8)

        # Draw edges with different styles for different types
        edge_styles = {
            'contains': 'solid',
            'mentions': 'dashed',
            'similar_to': 'dotted',
            'relates_to': 'dashdot'
        }

        for edge_type, style in edge_styles.items():
            edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get('edge_type') == edge_type]
            nx.draw_networkx_edges(subgraph, pos, edgelist=edges, style=style, alpha=0.6)

        # Draw labels
        labels = {n: self.nodes[n].content[:20] + '...' if len(self.nodes[n].content) > 20
                 else self.nodes[n].content for n in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

        plt.title(f"Graph around node: {center_node_id}")
        plt.axis('off')

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.show()
        return subgraph


class GraphIndexManager:
    """Manager class to handle multiple graph indices"""

    def __init__(self, base_store_path: str = "hybrid_rag/stores"):
        self.base_store_path = Path(base_store_path)
        self.indices: Dict[str, GraphIndexer] = {}

    def create_index(self, name: str, **kwargs) -> GraphIndexer:
        """Create a new graph index"""
        store_path = self.base_store_path / f"graph_{name}"
        indexer = GraphIndexer(store_path=str(store_path), **kwargs)
        self.indices[name] = indexer
        return indexer

    def get_index(self, name: str) -> Optional[GraphIndexer]:
        """Get an existing index"""
        return self.indices.get(name)

    def load_index(self, name: str) -> Optional[GraphIndexer]:
        """Load an index from disk"""
        store_path = self.base_store_path / f"graph_{name}"
        if not store_path.exists():
            return None

        indexer = GraphIndexer(store_path=str(store_path))
        if indexer.load_index():
            self.indices[name] = indexer
            return indexer

        return None

    def list_indices(self) -> List[str]:
        """List all available indices"""
        indices = []
        if self.base_store_path.exists():
            for path in self.base_store_path.iterdir():
                if path.is_dir() and path.name.startswith("graph_"):
                    indices.append(path.name[6:])  # Remove "graph_" prefix
        return indices


# Example usage and testing
if __name__ == "__main__":
    # Initialize the graph indexer
    indexer = GraphIndexer()

    # Example documents
    sample_docs = [
        Document(
            page_content="Apple Inc. is a technology company founded by Steve Jobs and Steve Wozniak. The company is headquartered in Cupertino, California.",
            metadata={"source": "tech_companies.txt", "page": 1}
        ),
        Document(
            page_content="Steve Jobs was the co-founder and longtime CEO of Apple Inc. He was known for his innovation in personal computing and mobile devices.",
            metadata={"source": "tech_leaders.txt", "page": 1}
        )
    ]

    # Index the documents
    stats = indexer.index_documents(sample_docs)
    print(f"Indexing stats: {stats}")

    # Save the index
    indexer.save_index()

    # Test similarity search
    similar_nodes = indexer.find_similar_nodes("Apple technology company", k=5)
    print(f"Similar nodes: {similar_nodes}")

    # Get index statistics
    index_stats = indexer.get_stats()
    print(f"Index stats: {index_stats}")