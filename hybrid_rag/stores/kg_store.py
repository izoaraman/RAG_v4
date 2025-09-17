"""
Knowledge Graph Store for Hybrid RAG
Manages persistence and retrieval of knowledge graphs
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import hashlib

import networkx as nx
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphMetadata:
    """Metadata for a knowledge graph"""
    name: str
    created_at: str
    updated_at: str
    num_nodes: int
    num_edges: int
    num_communities: int
    source_docs: List[str]
    description: str = ""

    def to_dict(self):
        return {
            'name': self.name,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'num_communities': self.num_communities,
            'source_docs': self.source_docs,
            'description': self.description
        }


class KnowledgeGraphStore:
    """
    Store and manage multiple knowledge graphs
    Provides persistence, versioning, and query capabilities
    """

    def __init__(self, store_path: str = "data/kg/"):
        """
        Initialize knowledge graph store

        Args:
            store_path: Directory to store graphs
        """
        self.store_path = store_path
        self.graphs: Dict[str, nx.MultiDiGraph] = {}
        self.metadata: Dict[str, GraphMetadata] = {}

        # Ensure store directory exists
        os.makedirs(store_path, exist_ok=True)

        # Load metadata index
        self.metadata_file = os.path.join(store_path, "graph_metadata.json")
        self.load_metadata_index()

    def load_metadata_index(self):
        """Load metadata index from disk"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for name, meta_dict in data.items():
                        self.metadata[name] = GraphMetadata(**meta_dict)
                logger.info(f"Loaded metadata for {len(self.metadata)} graphs")
            except Exception as e:
                logger.warning(f"Failed to load metadata index: {e}")

    def save_metadata_index(self):
        """Save metadata index to disk"""
        try:
            data = {name: meta.to_dict() for name, meta in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Metadata index saved")
        except Exception as e:
            logger.error(f"Failed to save metadata index: {e}")

    def save_graph(
        self,
        name: str,
        graph: nx.MultiDiGraph,
        entities: Dict[str, Any],
        relations: List[Any],
        communities: Dict[int, List[str]],
        community_summaries: Dict[int, str],
        source_docs: List[str],
        description: str = ""
    ) -> bool:
        """
        Save a knowledge graph to store

        Args:
            name: Unique name for the graph
            graph: NetworkX graph object
            entities: Entity dictionary
            relations: List of relations
            communities: Community mapping
            community_summaries: Community summaries
            source_docs: Source document IDs
            description: Optional description

        Returns:
            Success status
        """
        try:
            # Create versioned filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{name}_{timestamp}"

            # Save pickle file
            pickle_path = os.path.join(self.store_path, f"{base_name}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump({
                    'graph': graph,
                    'entities': entities,
                    'relations': relations,
                    'communities': communities,
                    'community_summaries': community_summaries,
                    'source_docs': source_docs,
                    'timestamp': timestamp
                }, f)

            # Save JSONL for inspection
            jsonl_path = os.path.join(self.store_path, f"{base_name}.jsonl")
            self._save_jsonl(
                jsonl_path, graph, entities, relations,
                communities, community_summaries
            )

            # Update metadata
            now = datetime.now().isoformat()
            if name in self.metadata:
                self.metadata[name].updated_at = now
                self.metadata[name].num_nodes = graph.number_of_nodes()
                self.metadata[name].num_edges = graph.number_of_edges()
                self.metadata[name].num_communities = len(communities)
                self.metadata[name].source_docs = source_docs
            else:
                self.metadata[name] = GraphMetadata(
                    name=name,
                    created_at=now,
                    updated_at=now,
                    num_nodes=graph.number_of_nodes(),
                    num_edges=graph.number_of_edges(),
                    num_communities=len(communities),
                    source_docs=source_docs,
                    description=description
                )

            # Save metadata index
            self.save_metadata_index()

            # Cache in memory
            self.graphs[name] = graph

            logger.info(f"Graph '{name}' saved successfully")
            logger.info(f"  Nodes: {graph.number_of_nodes()}")
            logger.info(f"  Edges: {graph.number_of_edges()}")
            logger.info(f"  Communities: {len(communities)}")

            return True

        except Exception as e:
            logger.error(f"Failed to save graph '{name}': {e}")
            return False

    def _save_jsonl(
        self,
        path: str,
        graph: nx.MultiDiGraph,
        entities: Dict[str, Any],
        relations: List[Any],
        communities: Dict[int, List[str]],
        community_summaries: Dict[int, str]
    ):
        """Save graph as JSONL for inspection"""
        with open(path, 'w') as f:
            # Metadata
            f.write(json.dumps({
                'type': 'metadata',
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'communities': len(communities)
            }) + '\n')

            # Entities
            for entity_id, entity in entities.items():
                f.write(json.dumps({
                    'type': 'entity',
                    'data': entity.to_dict() if hasattr(entity, 'to_dict') else str(entity)
                }) + '\n')

            # Relations
            for relation in relations:
                f.write(json.dumps({
                    'type': 'relation',
                    'data': relation.to_dict() if hasattr(relation, 'to_dict') else str(relation)
                }) + '\n')

            # Communities
            for comm_id, nodes in communities.items():
                f.write(json.dumps({
                    'type': 'community',
                    'id': comm_id,
                    'nodes': nodes,
                    'summary': community_summaries.get(comm_id, '')
                }) + '\n')

    def load_graph(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load a knowledge graph from store

        Args:
            name: Name of the graph
            version: Optional version timestamp

        Returns:
            Dictionary with graph components or None
        """
        try:
            # Check cache first
            if name in self.graphs and version is None:
                logger.info(f"Loading graph '{name}' from cache")
                # Return cached version (need to load full data)

            # Find the graph file
            if version:
                pickle_path = os.path.join(self.store_path, f"{name}_{version}.pkl")
            else:
                # Find latest version
                files = [f for f in os.listdir(self.store_path)
                        if f.startswith(f"{name}_") and f.endswith('.pkl')]
                if not files:
                    logger.error(f"No graph found with name '{name}'")
                    return None
                files.sort()  # Latest version last
                pickle_path = os.path.join(self.store_path, files[-1])

            # Load from pickle
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)

            # Cache the graph
            self.graphs[name] = data['graph']

            logger.info(f"Graph '{name}' loaded successfully")
            logger.info(f"  Nodes: {data['graph'].number_of_nodes()}")
            logger.info(f"  Edges: {data['graph'].number_of_edges()}")

            return data

        except Exception as e:
            logger.error(f"Failed to load graph '{name}': {e}")
            return None

    def list_graphs(self) -> List[GraphMetadata]:
        """
        List all available graphs

        Returns:
            List of graph metadata
        """
        return list(self.metadata.values())

    def delete_graph(self, name: str, version: Optional[str] = None) -> bool:
        """
        Delete a graph from store

        Args:
            name: Name of the graph
            version: Optional version to delete (None = all versions)

        Returns:
            Success status
        """
        try:
            if version:
                # Delete specific version
                files = [
                    f"{name}_{version}.pkl",
                    f"{name}_{version}.jsonl"
                ]
            else:
                # Delete all versions
                files = [f for f in os.listdir(self.store_path)
                        if f.startswith(f"{name}_")]

                # Remove from metadata
                if name in self.metadata:
                    del self.metadata[name]
                    self.save_metadata_index()

            # Delete files
            for filename in files:
                filepath = os.path.join(self.store_path, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Deleted {filename}")

            # Remove from cache
            if name in self.graphs:
                del self.graphs[name]

            return True

        except Exception as e:
            logger.error(f"Failed to delete graph '{name}': {e}")
            return False

    def query_graph(
        self,
        name: str,
        query_type: str,
        **kwargs
    ) -> Any:
        """
        Query a graph for specific information

        Args:
            name: Name of the graph
            query_type: Type of query (neighbors, path, subgraph, etc.)
            **kwargs: Query parameters

        Returns:
            Query results
        """
        # Load graph if not cached
        if name not in self.graphs:
            data = self.load_graph(name)
            if not data:
                return None
            self.graphs[name] = data['graph']

        graph = self.graphs[name]

        try:
            if query_type == "neighbors":
                # Get neighbors of a node
                node_id = kwargs.get('node_id')
                if node_id and node_id in graph:
                    return list(graph.neighbors(node_id))

            elif query_type == "shortest_path":
                # Find shortest path between nodes
                source = kwargs.get('source')
                target = kwargs.get('target')
                if source in graph and target in graph:
                    try:
                        path = nx.shortest_path(graph, source, target)
                        return path
                    except nx.NetworkXNoPath:
                        return []

            elif query_type == "subgraph":
                # Extract subgraph around nodes
                nodes = kwargs.get('nodes', [])
                depth = kwargs.get('depth', 1)

                # Expand to include neighbors up to depth
                expanded_nodes = set(nodes)
                for _ in range(depth):
                    new_nodes = set()
                    for node in expanded_nodes:
                        if node in graph:
                            new_nodes.update(graph.neighbors(node))
                    expanded_nodes.update(new_nodes)

                # Extract subgraph
                subgraph = graph.subgraph(expanded_nodes)
                return subgraph

            elif query_type == "node_info":
                # Get node attributes
                node_id = kwargs.get('node_id')
                if node_id in graph:
                    return graph.nodes[node_id]

            elif query_type == "edge_info":
                # Get edge attributes
                source = kwargs.get('source')
                target = kwargs.get('target')
                if graph.has_edge(source, target):
                    return graph.edges[source, target]

            elif query_type == "centrality":
                # Calculate node centrality
                centrality_type = kwargs.get('centrality_type', 'degree')

                if centrality_type == 'degree':
                    return dict(nx.degree_centrality(graph))
                elif centrality_type == 'betweenness':
                    return dict(nx.betweenness_centrality(graph))
                elif centrality_type == 'closeness':
                    return dict(nx.closeness_centrality(graph))
                elif centrality_type == 'pagerank':
                    return dict(nx.pagerank(graph))

            elif query_type == "stats":
                # Get graph statistics
                return {
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges(),
                    'density': nx.density(graph),
                    'is_connected': nx.is_weakly_connected(graph),
                    'num_components': nx.number_weakly_connected_components(graph),
                    'avg_degree': sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
                }

            else:
                logger.warning(f"Unknown query type: {query_type}")
                return None

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return None

    def merge_graphs(
        self,
        name1: str,
        name2: str,
        new_name: str
    ) -> bool:
        """
        Merge two graphs into a new graph

        Args:
            name1: First graph name
            name2: Second graph name
            new_name: Name for merged graph

        Returns:
            Success status
        """
        try:
            # Load both graphs
            data1 = self.load_graph(name1)
            data2 = self.load_graph(name2)

            if not data1 or not data2:
                logger.error("Failed to load one or both graphs")
                return False

            # Create merged graph
            merged_graph = nx.compose(data1['graph'], data2['graph'])

            # Merge entities
            merged_entities = {**data1['entities'], **data2['entities']}

            # Merge relations
            merged_relations = data1['relations'] + data2['relations']

            # Merge communities (renumber to avoid conflicts)
            merged_communities = data1['communities'].copy()
            offset = max(merged_communities.keys()) + 1 if merged_communities else 0
            for comm_id, nodes in data2['communities'].items():
                merged_communities[comm_id + offset] = nodes

            # Merge community summaries
            merged_summaries = data1['community_summaries'].copy()
            for comm_id, summary in data2['community_summaries'].items():
                merged_summaries[comm_id + offset] = summary

            # Merge source docs
            merged_docs = list(set(data1['source_docs'] + data2['source_docs']))

            # Save merged graph
            success = self.save_graph(
                new_name,
                merged_graph,
                merged_entities,
                merged_relations,
                merged_communities,
                merged_summaries,
                merged_docs,
                description=f"Merged from {name1} and {name2}"
            )

            if success:
                logger.info(f"Graphs merged successfully into '{new_name}'")

            return success

        except Exception as e:
            logger.error(f"Failed to merge graphs: {e}")
            return False