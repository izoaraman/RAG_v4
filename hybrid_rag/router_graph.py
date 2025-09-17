"""
LangGraph Router for Hybrid RAG System

This module provides intelligent routing and orchestration for the hybrid RAG system using LangGraph.
It includes:
- Query analysis and routing
- Multi-step reasoning workflows
- Agent-based document processing
- Dynamic retrieval strategy selection
- Result fusion and ranking
- Conversational memory management
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import asyncio
from pathlib import Path
from enum import Enum
import uuid

from langchain.schema import Document, BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import AzureChatOpenAI

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    logger.warning("LangGraph not available. Install with: pip install langgraph")

from pydantic import BaseModel, Field

from .index_graph import GraphIndexer
from .retriever_multimodal import MultimodalRetriever
from .retriever_graph import GraphRetriever
from .index_multimodal import MultimodalIndexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SUMMARIZATION = "summarization"
    MULTIMODAL = "multimodal"
    GRAPH_TRAVERSAL = "graph_traversal"
    CONVERSATIONAL = "conversational"

class RetrievalStrategy(Enum):
    """Available retrieval strategies"""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    MULTIMODAL_ONLY = "multimodal_only"
    HYBRID_VECTOR_GRAPH = "hybrid_vector_graph"
    HYBRID_ALL = "hybrid_all"
    ADAPTIVE = "adaptive"

@dataclass
class QueryAnalysis(BaseModel):
    """Analysis of user query"""
    query_type: QueryType
    entities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    intent: str = ""
    complexity: float = 0.0  # 0-1 scale
    requires_multimodal: bool = False
    requires_graph_reasoning: bool = False
    temporal_aspect: bool = False
    suggested_strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE

@dataclass
class RouterState:
    """State maintained by the router throughout processing"""
    session_id: str
    query: str
    query_analysis: Optional[QueryAnalysis] = None
    retrieval_strategy: Optional[RetrievalStrategy] = None
    vector_results: List[Document] = field(default_factory=list)
    graph_results: Any = None
    multimodal_results: List[Any] = field(default_factory=list)
    fused_results: List[Document] = field(default_factory=list)
    final_response: str = ""
    conversation_history: List[BaseMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_count: int = 0
    max_steps: int = 10

class QueryAnalyzer:
    """Analyzes user queries to determine optimal processing strategy"""

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze the following user query and provide structured analysis:

        Query: {query}

        Please analyze:
        1. Query type (factual, analytical, comparative, summarization, multimodal, graph_traversal, conversational)
        2. Key entities mentioned
        3. Important keywords
        4. User's intent
        5. Complexity level (0.0 to 1.0)
        6. Whether it requires multimodal processing
        7. Whether it requires graph-based reasoning
        8. Whether it has temporal aspects
        9. Suggested retrieval strategy

        Provide your analysis in the following JSON format:
        {{
            "query_type": "one of the types above",
            "entities": ["entity1", "entity2"],
            "keywords": ["keyword1", "keyword2"],
            "intent": "brief description of user intent",
            "complexity": 0.0-1.0,
            "requires_multimodal": true/false,
            "requires_graph_reasoning": true/false,
            "temporal_aspect": true/false,
            "suggested_strategy": "vector_only|graph_only|multimodal_only|hybrid_vector_graph|hybrid_all|adaptive"
        }}
        """)

    def analyze_query(self, query: str, conversation_history: List[BaseMessage] = None) -> QueryAnalysis:
        """Analyze a user query and return structured analysis"""
        try:
            # Include conversation context if available
            context = ""
            if conversation_history:
                recent_messages = conversation_history[-4:]  # Last 2 exchanges
                context = "\n".join([f"{msg.type}: {msg.content}" for msg in recent_messages])

            formatted_prompt = self.analysis_prompt.format(
                query=query,
                context=context if context else "No previous context"
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])

            # Parse JSON response
            try:
                analysis_data = json.loads(response.content)
                return QueryAnalysis(**analysis_data)
            except json.JSONDecodeError:
                # Fallback to basic analysis
                return self._fallback_analysis(query)

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return self._fallback_analysis(query)

    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """Provide basic analysis when LLM analysis fails"""
        # Simple heuristics
        query_lower = query.lower()

        # Determine query type
        if any(word in query_lower for word in ['what', 'who', 'when', 'where']):
            query_type = QueryType.FACTUAL
        elif any(word in query_lower for word in ['analyze', 'explain', 'why', 'how']):
            query_type = QueryType.ANALYTICAL
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            query_type = QueryType.COMPARATIVE
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            query_type = QueryType.SUMMARIZATION
        elif any(word in query_lower for word in ['image', 'picture', 'video', 'audio']):
            query_type = QueryType.MULTIMODAL
        elif any(word in query_lower for word in ['relationship', 'connect', 'related']):
            query_type = QueryType.GRAPH_TRAVERSAL
        else:
            query_type = QueryType.CONVERSATIONAL

        # Basic entity extraction (very simple)
        entities = []
        # This is a placeholder - in practice, you'd use NER models

        # Determine strategy
        if query_type == QueryType.MULTIMODAL:
            strategy = RetrievalStrategy.MULTIMODAL_ONLY
        elif query_type == QueryType.GRAPH_TRAVERSAL:
            strategy = RetrievalStrategy.GRAPH_ONLY
        elif query_type in [QueryType.ANALYTICAL, QueryType.COMPARATIVE]:
            strategy = RetrievalStrategy.HYBRID_ALL
        else:
            strategy = RetrievalStrategy.ADAPTIVE

        return QueryAnalysis(
            query_type=query_type,
            entities=entities,
            keywords=query.split()[:5],  # Simple keyword extraction
            intent=f"User wants to {query_type.value} information",
            complexity=0.5,  # Default medium complexity
            requires_multimodal=query_type == QueryType.MULTIMODAL,
            requires_graph_reasoning=query_type == QueryType.GRAPH_TRAVERSAL,
            temporal_aspect='time' in query_lower or 'when' in query_lower,
            suggested_strategy=strategy
        )


class RetrievalOrchestrator:
    """Orchestrates different retrieval methods based on query analysis"""

    def __init__(
        self,
        vector_retriever: Any,  # Your existing vector retriever
        graph_retriever: GraphRetriever,
        multimodal_retriever: MultimodalRetriever,
        llm: AzureChatOpenAI
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.multimodal_retriever = multimodal_retriever
        self.llm = llm

    async def retrieve_vector(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve using vector similarity"""
        try:
            if hasattr(self.vector_retriever, 'similarity_search'):
                results = self.vector_retriever.similarity_search(query, k=k)
            elif hasattr(self.vector_retriever, 'get_relevant_documents'):
                results = self.vector_retriever.get_relevant_documents(query)
            else:
                # Fallback - assume it's callable
                results = self.vector_retriever(query, k=k)

            return results if isinstance(results, list) else []
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            return []

    async def retrieve_graph(self, query: str, method: str = 'hybrid') -> Any:
        """Retrieve using graph-based methods"""
        try:
            return self.graph_retriever.retrieve(query, method=method)
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return None

    async def retrieve_multimodal(self, query: str, k: int = 5) -> List[Any]:
        """Retrieve using multimodal methods"""
        try:
            return self.multimodal_retriever.retrieve_text(query, k=k)
        except Exception as e:
            logger.error(f"Error in multimodal retrieval: {e}")
            return []

    def fuse_results(
        self,
        vector_results: List[Document],
        graph_results: Any,
        multimodal_results: List[Any],
        fusion_method: str = "weighted_rank"
    ) -> List[Document]:
        """Fuse results from different retrieval methods"""
        fused_docs = []

        # Convert all results to Document format for consistency
        all_results = []

        # Add vector results
        for doc in vector_results:
            all_results.append((doc, 'vector', 1.0))

        # Add graph results
        if graph_results and hasattr(graph_results, 'nodes'):
            for node in graph_results.nodes:
                doc = Document(
                    page_content=node.content,
                    metadata={**node.metadata, 'node_type': node.node_type, 'source_method': 'graph'}
                )
                all_results.append((doc, 'graph', graph_results.score))

        # Add multimodal results
        for result in multimodal_results:
            if hasattr(result, 'document'):
                doc = Document(
                    page_content=result.document.text_content,
                    metadata={**result.document.metadata, 'source_method': 'multimodal'}
                )
                all_results.append((doc, 'multimodal', result.score))

        # Apply fusion strategy
        if fusion_method == "weighted_rank":
            # Weight by source method and rank
            method_weights = {'vector': 1.0, 'graph': 0.8, 'multimodal': 0.9}
            scored_results = []

            for doc, method, score in all_results:
                final_score = score * method_weights.get(method, 1.0)
                scored_results.append((doc, final_score))

            # Sort by score and remove duplicates
            scored_results.sort(key=lambda x: x[1], reverse=True)
            seen_content = set()

            for doc, score in scored_results:
                content_hash = hash(doc.page_content[:200])  # Use first 200 chars for dedup
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    doc.metadata['fusion_score'] = score
                    fused_docs.append(doc)

                if len(fused_docs) >= 10:  # Limit final results
                    break

        return fused_docs


class HybridRAGRouter:
    """Main router class using LangGraph for workflow orchestration"""

    def __init__(
        self,
        llm: AzureChatOpenAI,
        vector_retriever: Any,
        graph_indexer: GraphIndexer,
        multimodal_indexer: MultimodalIndexer,
        memory_window: int = 10
    ):
        self.llm = llm
        self.vector_retriever = vector_retriever

        # Initialize retrievers
        self.graph_retriever = GraphRetriever(graph_indexer)
        self.multimodal_retriever = MultimodalRetriever()

        # Initialize components
        self.query_analyzer = QueryAnalyzer(llm)
        self.orchestrator = RetrievalOrchestrator(
            vector_retriever, self.graph_retriever, self.multimodal_retriever, llm
        )

        # Memory management
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history"
        )

        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        self.compiled_workflow = None

        # Session management
        self.sessions: Dict[str, RouterState] = {}

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(RouterState)

        # Define nodes (processing steps)
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("route_retrieval", self._route_retrieval_node)
        workflow.add_node("vector_retrieve", self._vector_retrieve_node)
        workflow.add_node("graph_retrieve", self._graph_retrieve_node)
        workflow.add_node("multimodal_retrieve", self._multimodal_retrieve_node)
        workflow.add_node("fuse_results", self._fuse_results_node)
        workflow.add_node("generate_response", self._generate_response_node)

        # Define edges (workflow transitions)
        workflow.set_entry_point("analyze_query")

        workflow.add_edge("analyze_query", "route_retrieval")

        # Conditional routing based on strategy
        workflow.add_conditional_edges(
            "route_retrieval",
            self._decide_retrieval_path,
            {
                "vector_only": "vector_retrieve",
                "graph_only": "graph_retrieve",
                "multimodal_only": "multimodal_retrieve",
                "hybrid": "vector_retrieve",  # Start with vector for hybrid
                "end": END
            }
        )

        # Vector retrieval path
        workflow.add_conditional_edges(
            "vector_retrieve",
            self._decide_after_vector,
            {
                "add_graph": "graph_retrieve",
                "add_multimodal": "multimodal_retrieve",
                "fuse": "fuse_results",
                "generate": "generate_response"
            }
        )

        # Graph retrieval path
        workflow.add_conditional_edges(
            "graph_retrieve",
            self._decide_after_graph,
            {
                "add_multimodal": "multimodal_retrieve",
                "fuse": "fuse_results",
                "generate": "generate_response"
            }
        )

        # Multimodal retrieval path
        workflow.add_edge("multimodal_retrieve", "fuse_results")
        workflow.add_edge("fuse_results", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow

    async def _analyze_query_node(self, state: RouterState) -> RouterState:
        """Analyze the user query"""
        logger.info(f"Analyzing query: {state.query}")

        state.query_analysis = self.query_analyzer.analyze_query(
            state.query,
            state.conversation_history
        )

        state.retrieval_strategy = state.query_analysis.suggested_strategy
        state.step_count += 1

        logger.info(f"Query analysis: {state.query_analysis.query_type}, Strategy: {state.retrieval_strategy}")
        return state

    def _decide_retrieval_path(self, state: RouterState) -> str:
        """Decide which retrieval path to take"""
        if not state.query_analysis:
            return "end"

        strategy = state.retrieval_strategy

        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return "vector_only"
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            return "graph_only"
        elif strategy == RetrievalStrategy.MULTIMODAL_ONLY:
            return "multimodal_only"
        elif strategy in [RetrievalStrategy.HYBRID_VECTOR_GRAPH, RetrievalStrategy.HYBRID_ALL, RetrievalStrategy.ADAPTIVE]:
            return "hybrid"
        else:
            return "end"

    async def _route_retrieval_node(self, state: RouterState) -> RouterState:
        """Route to appropriate retrieval methods"""
        # This is handled by conditional edges
        return state

    async def _vector_retrieve_node(self, state: RouterState) -> RouterState:
        """Execute vector retrieval"""
        logger.info("Executing vector retrieval")

        try:
            k = 5 if state.query_analysis.complexity < 0.7 else 8
            state.vector_results = await self.orchestrator.retrieve_vector(state.query, k=k)
            logger.info(f"Retrieved {len(state.vector_results)} vector results")
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            state.vector_results = []

        state.step_count += 1
        return state

    def _decide_after_vector(self, state: RouterState) -> str:
        """Decide what to do after vector retrieval"""
        if not state.query_analysis:
            return "generate"

        strategy = state.retrieval_strategy

        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return "generate"
        elif strategy == RetrievalStrategy.HYBRID_VECTOR_GRAPH:
            return "add_graph"
        elif strategy == RetrievalStrategy.HYBRID_ALL:
            return "add_graph"
        else:  # ADAPTIVE
            # Decide based on query analysis
            if state.query_analysis.requires_graph_reasoning:
                return "add_graph"
            elif state.query_analysis.requires_multimodal:
                return "add_multimodal"
            else:
                return "generate"

    async def _graph_retrieve_node(self, state: RouterState) -> RouterState:
        """Execute graph retrieval"""
        logger.info("Executing graph retrieval")

        try:
            method = 'entity_centric' if state.query_analysis.entities else 'hybrid'
            state.graph_results = await self.orchestrator.retrieve_graph(state.query, method=method)
            logger.info(f"Retrieved graph results with score: {state.graph_results.score if state.graph_results else 0}")
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            state.graph_results = None

        state.step_count += 1
        return state

    def _decide_after_graph(self, state: RouterState) -> str:
        """Decide what to do after graph retrieval"""
        if not state.query_analysis:
            return "fuse"

        if (state.retrieval_strategy == RetrievalStrategy.HYBRID_ALL or
            state.query_analysis.requires_multimodal):
            return "add_multimodal"
        else:
            return "fuse"

    async def _multimodal_retrieve_node(self, state: RouterState) -> RouterState:
        """Execute multimodal retrieval"""
        logger.info("Executing multimodal retrieval")

        try:
            state.multimodal_results = await self.orchestrator.retrieve_multimodal(state.query, k=5)
            logger.info(f"Retrieved {len(state.multimodal_results)} multimodal results")
        except Exception as e:
            logger.error(f"Error in multimodal retrieval: {e}")
            state.multimodal_results = []

        state.step_count += 1
        return state

    async def _fuse_results_node(self, state: RouterState) -> RouterState:
        """Fuse results from different retrieval methods"""
        logger.info("Fusing retrieval results")

        state.fused_results = self.orchestrator.fuse_results(
            state.vector_results,
            state.graph_results,
            state.multimodal_results
        )

        logger.info(f"Fused into {len(state.fused_results)} final results")
        state.step_count += 1
        return state

    async def _generate_response_node(self, state: RouterState) -> RouterState:
        """Generate final response using LLM"""
        logger.info("Generating final response")

        try:
            # Prepare context from retrieved documents
            if state.fused_results:
                context_docs = state.fused_results
            elif state.vector_results:
                context_docs = state.vector_results
            else:
                context_docs = []

            context = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(context_docs[:5])  # Limit context
            ])

            # Prepare conversation history
            history_context = ""
            if state.conversation_history:
                recent_history = state.conversation_history[-4:]  # Last 2 exchanges
                history_context = "\n".join([
                    f"{msg.type}: {msg.content}" for msg in recent_history
                ])

            # Generate response prompt
            response_prompt = f"""
Based on the following context and conversation history, please provide a comprehensive answer to the user's question.

Conversation History:
{history_context}

Current Question: {state.query}

Retrieved Context:
{context}

Please provide a helpful, accurate, and well-structured response. If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can.
"""

            response = self.llm.invoke([HumanMessage(content=response_prompt)])
            state.final_response = response.content

            # Update conversation history
            state.conversation_history.extend([
                HumanMessage(content=state.query),
                AIMessage(content=state.final_response)
            ])

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.final_response = "I apologize, but I encountered an error while generating a response. Please try rephrasing your question."

        state.step_count += 1
        return state

    def get_session(self, session_id: str) -> RouterState:
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = RouterState(
                session_id=session_id,
                query=""
            )
        return self.sessions[session_id]

    async def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Process a user query through the complete workflow"""
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Get or create session state
        state = self.get_session(session_id)
        state.query = query
        state.step_count = 0

        # Compile workflow if not done
        if self.compiled_workflow is None:
            try:
                # Use SQLite checkpointer for memory
                memory = SqliteSaver.from_conn_string(":memory:")
                self.compiled_workflow = self.workflow.compile(checkpointer=memory)
            except:
                # Fallback without checkpointer
                self.compiled_workflow = self.workflow.compile()

        try:
            # Execute workflow
            config = {"configurable": {"thread_id": session_id}}
            result = await self.compiled_workflow.ainvoke(state, config)

            # Prepare response
            response = {
                "session_id": session_id,
                "query": query,
                "response": result.final_response,
                "query_analysis": result.query_analysis.__dict__ if result.query_analysis else {},
                "retrieval_stats": {
                    "vector_results": len(result.vector_results),
                    "graph_results": bool(result.graph_results),
                    "multimodal_results": len(result.multimodal_results),
                    "fused_results": len(result.fused_results),
                    "strategy_used": result.retrieval_strategy.value if result.retrieval_strategy else "unknown"
                },
                "metadata": result.metadata,
                "steps_taken": result.step_count
            }

            # Update session
            self.sessions[session_id] = result

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "session_id": session_id,
                "query": query,
                "response": "I apologize, but I encountered an error while processing your query. Please try again.",
                "error": str(e)
            }

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        if session_id in self.sessions:
            history = []
            for msg in self.sessions[session_id].conversation_history:
                history.append({
                    "type": msg.type,
                    "content": msg.content
                })
            return history
        return []

    def clear_session(self, session_id: str):
        """Clear a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "active_sessions": len(self.sessions),
            "total_queries_processed": sum(s.step_count for s in self.sessions.values()),
            "available_strategies": [s.value for s in RetrievalStrategy],
            "supported_query_types": [q.value for q in QueryType]
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from langchain_community.chat_models import AzureChatOpenAI

    # Mock components for testing
    class MockVectorRetriever:
        def similarity_search(self, query: str, k: int = 5):
            return [Document(page_content=f"Mock result for: {query}", metadata={"source": "mock"})]

    class MockGraphIndexer:
        def __init__(self):
            self.graph = None
            self.nodes = {}
            self.edges = {}

    class MockMultimodalIndexer:
        pass

    async def test_router():
        # Initialize mock LLM (you would use real Azure OpenAI here)
        llm = AzureChatOpenAI(
            deployment_name="gpt-4",
            model_name="gpt-4",
            azure_endpoint="your-endpoint",
            api_key="your-key",
            api_version="2024-12-01-preview"
        )

        # Initialize router with mock components
        router = HybridRAGRouter(
            llm=llm,
            vector_retriever=MockVectorRetriever(),
            graph_indexer=MockGraphIndexer(),
            multimodal_indexer=MockMultimodalIndexer()
        )

        # Test query processing
        test_queries = [
            "What is machine learning?",
            "Compare deep learning and traditional ML",
            "Show me images related to neural networks"
        ]

        for query in test_queries:
            print(f"\nProcessing: {query}")
            result = await router.process_query(query)
            print(f"Response: {result['response']}")
            print(f"Strategy: {result['retrieval_stats']['strategy_used']}")

    # Run test
    # asyncio.run(test_router())