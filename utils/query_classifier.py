"""
Query Classification and Response Depth Management
"""

import re
from typing import Dict, List, Tuple, Literal, Optional
from enum import Enum
import logging

class QueryIntent(Enum):
    """Query intent categories"""
    QUICK_FACT = "quick_fact"      # Single fact, definition, date
    TARGETED = "targeted"           # Specific analysis, comparison
    FULL = "full"                   # Comprehensive analysis, framework
    CONVERSATIONAL = "conversational"  # Casual, chat-like queries
    CLARIFICATION = "clarification"    # Follow-up or clarification questions

class QueryClassifier:
    """Classify queries and determine appropriate response depth"""
    
    # Keyword patterns for different intents
    QUICK_PATTERNS = [
        r"^what is\b",
        r"^when\b",
        r"^who\b",
        r"^where\b",
        r"^how much\b",
        r"^how many\b",
        r"^define\b",
        r"^meaning of\b",
        r"^list\b",
        r"^name\b",
        r"^give me\b",
        r"^show me\b",
        r"^tell me\b"
    ]
    
    FULL_PATTERNS = [
        r"\bassess\b",
        r"\banalyze\b",
        r"\banalyse\b",
        r"\bevaluate\b",
        r"\bcompare\b",
        r"\bframework\b",
        r"\bstrategy\b",
        r"\bplan\b",
        r"\brecommend\b",
        r"\bimplication\b",
        r"\brisk\b",
        r"\bcompliance\b",
        r"\binvestigat\b",
        r"\breview\b",
        r"\bcomprehensive\b",
        r"\bdetailed\b",
        r"\bexplain\b.*\bprocess\b",
        r"\bhow does\b.*\bwork\b",
        r"\bin-depth\b",
        r"\bthorough\b",
        r"\bcomplete\b.*\banalysis\b"
    ]
    
    TARGETED_PATTERNS = [
        r"\bspecific\b",
        r"\bparticular\b",
        r"\bregarding\b",
        r"\babout\b",
        r"\brelated to\b",
        r"\bconcerning\b",
        r"\bwith respect to\b",
        r"\bfocus on\b",
        r"\baspect\b",
        r"\bpoint\b"
    ]
    
    CONVERSATIONAL_PATTERNS = [
        r"^can you\b",
        r"^could you\b",
        r"^would you\b",
        r"^please\b",
        r"^thanks\b",
        r"^thank you\b",
        r"^hi\b",
        r"^hello\b",
        r"^hey\b",
        r"\?$"  # Questions ending with ?
    ]
    
    CLARIFICATION_PATTERNS = [
        r"^what do you mean\b",
        r"^can you clarify\b",
        r"^what about\b",
        r"^how about\b",
        r"^and\b",
        r"^but\b",
        r"^also\b",
        r"^furthermore\b",
        r"^moreover\b"
    ]
    
    @classmethod
    def classify_query(cls, query: str, conversation_history: List = None) -> QueryIntent:
        """
        Classify query intent based on patterns and keywords
        
        Args:
            query: User's query string
            conversation_history: Previous conversation for context
            
        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower().strip()
        
        # Check for clarification patterns first (highest priority in follow-ups)
        if conversation_history and len(conversation_history) > 0:
            for pattern in cls.CLARIFICATION_PATTERNS:
                if re.search(pattern, query_lower):
                    return QueryIntent.CLARIFICATION
        
        # Check for conversational patterns
        conversational_score = sum(1 for p in cls.CONVERSATIONAL_PATTERNS if re.search(p, query_lower))
        if conversational_score >= 2:  # Multiple conversational markers
            return QueryIntent.CONVERSATIONAL
        
        # Check for full analysis patterns (high priority)
        full_score = sum(1 for p in cls.FULL_PATTERNS if re.search(p, query_lower))
        if full_score >= 2:  # Strong signal for comprehensive analysis
            return QueryIntent.FULL
        elif full_score == 1:
            # Check if it conflicts with quick patterns
            quick_score = sum(1 for p in cls.QUICK_PATTERNS if re.search(p, query_lower))
            if quick_score == 0:
                return QueryIntent.FULL
        
        # Check for quick fact patterns
        for pattern in cls.QUICK_PATTERNS:
            if re.search(pattern, query_lower):
                # Unless overridden by full patterns
                if not any(re.search(p, query_lower) for p in cls.FULL_PATTERNS):
                    return QueryIntent.QUICK_FACT
        
        # Check for targeted patterns
        targeted_score = sum(1 for p in cls.TARGETED_PATTERNS if re.search(p, query_lower))
        if targeted_score >= 1:
            return QueryIntent.TARGETED
        
        # Default based on query complexity and length
        word_count = len(query_lower.split())
        
        if word_count < 5:
            return QueryIntent.QUICK_FACT
        elif word_count < 15:
            # Check question complexity
            if '?' in query_lower and any(word in query_lower for word in ['how', 'why', 'what']):
                return QueryIntent.TARGETED
            return QueryIntent.QUICK_FACT
        elif word_count > 30:
            return QueryIntent.FULL
        else:
            return QueryIntent.TARGETED
    
    @classmethod
    def analyze_retrieval_quality(cls, documents: List, scores: List[float] = None) -> Dict:
        """
        Analyze retrieval quality signals
        
        Args:
            documents: Retrieved documents
            scores: Similarity scores (if available)
            
        Returns:
            Dictionary with quality metrics
        """
        analysis = {
            "num_docs": len(documents),
            "score_gap": 0.0,
            "top_score": 0.0,
            "coverage_breadth": 0,
            "unique_sources": set(),
            "confidence": "medium"
        }
        
        if not documents:
            analysis["confidence"] = "low"
            return analysis
        
        # Calculate unique sources and pages
        for doc in documents:
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', 'unknown')
                analysis["unique_sources"].add(source)
        
        analysis["coverage_breadth"] = len(analysis["unique_sources"])
        
        # Analyze scores if available
        if scores and len(scores) >= 2:
            analysis["top_score"] = scores[0]
            analysis["score_gap"] = scores[0] - scores[1]
            
            # Determine confidence based on scores
            if analysis["top_score"] > 0.8 and analysis["score_gap"] > 0.2:
                analysis["confidence"] = "high"
            elif analysis["top_score"] < 0.5:
                analysis["confidence"] = "low"
        
        return analysis
    
    @classmethod
    def determine_response_strategy(
        cls,
        query: str,
        documents: List,
        scores: List[float] = None,
        conversation_history: List = None,
        override_intent: Optional[QueryIntent] = None
    ) -> Dict:
        """
        Determine optimal response strategy based on multiple signals
        
        Args:
            query: User query
            documents: Retrieved documents
            scores: Similarity scores
            conversation_history: Previous conversation turns
            override_intent: Optional manual override for intent
            
        Returns:
            Strategy dictionary with response parameters
        """
        # Classify query intent
        intent = override_intent or cls.classify_query(query, conversation_history)
        
        # Analyze retrieval quality
        retrieval_analysis = cls.analyze_retrieval_quality(documents, scores)
        
        # Check conversation context
        is_followup = False
        has_deep_context = False
        if conversation_history and len(conversation_history) > 2:
            is_followup = True
            if len(conversation_history) > 6:
                has_deep_context = True
                # Upgrade intent if this is a deep follow-up
                if intent == QueryIntent.QUICK_FACT:
                    intent = QueryIntent.TARGETED
        
        # Determine strategy
        strategy = {
            "intent": intent.value,
            "confidence": retrieval_analysis["confidence"],
            "max_response_length": 200,  # default
            "include_citations": True,
            "include_analysis": False,
            "include_caveats": False,
            "response_style": "concise",
            "use_framework": False,  # New flag for Analysis Framework
            "adaptive_depth": True   # Allow dynamic adjustment
        }
        
        # Adjust based on intent and confidence
        if intent == QueryIntent.QUICK_FACT:
            strategy.update({
                "max_response_length": 100 if retrieval_analysis["confidence"] == "high" else 150,
                "response_style": "direct",
                "include_citations": retrieval_analysis["confidence"] != "high",
                "include_caveats": retrieval_analysis["confidence"] == "low",
                "use_framework": False
            })
                
        elif intent == QueryIntent.TARGETED:
            strategy.update({
                "max_response_length": 300 if retrieval_analysis["confidence"] == "high" else 400,
                "include_analysis": True,
                "response_style": "analytical",
                "include_caveats": retrieval_analysis["confidence"] == "low",
                "use_framework": False  # Only use framework when explicitly needed
            })
                
        elif intent == QueryIntent.FULL:
            strategy.update({
                "max_response_length": 600 if not has_deep_context else 800,
                "include_analysis": True,
                "response_style": "comprehensive",
                "include_citations": True,
                "use_framework": True  # Use framework for full analysis
            })
            
        elif intent == QueryIntent.CONVERSATIONAL:
            strategy.update({
                "max_response_length": 200,
                "response_style": "friendly",
                "include_citations": False,
                "include_analysis": False,
                "use_framework": False
            })
            
        elif intent == QueryIntent.CLARIFICATION:
            strategy.update({
                "max_response_length": 250,
                "response_style": "explanatory",
                "include_citations": True,
                "include_analysis": is_followup,
                "use_framework": False
            })
        
        # Dynamic adjustments based on evidence quality
        if retrieval_analysis["confidence"] == "high" and retrieval_analysis["score_gap"] > 0.3:
            # Very confident - can be more concise
            strategy["max_response_length"] = int(strategy["max_response_length"] * 0.8)
            strategy["adaptive_depth"] = False
            
        elif retrieval_analysis["confidence"] == "low":
            # Low confidence - provide more context and caveats
            strategy["max_response_length"] = int(strategy["max_response_length"] * 1.3)
            strategy["include_caveats"] = True
            
        # Adjust for document coverage breadth
        if retrieval_analysis["coverage_breadth"] > 3:
            strategy["response_style"] = "comparative"
            strategy["max_response_length"] = min(600, int(strategy["max_response_length"] * 1.3))
        elif retrieval_analysis["coverage_breadth"] == 1:
            # Single source - be more focused
            strategy["response_style"] = "focused"
            
        # Adjust for follow-up context
        if is_followup and not intent == QueryIntent.CLARIFICATION:
            strategy["max_response_length"] = int(strategy["max_response_length"] * 1.1)
            
        # Log the determined strategy for debugging
        logging.info(f"Query intent: {intent.value}, Confidence: {retrieval_analysis['confidence']}, Style: {strategy['response_style']}")
        
        return strategy
    
    @classmethod
    def format_prompt_with_strategy(
        cls,
        query: str,
        context: str,
        strategy: Dict,
        system_role: str,
        data_type: str = "Current documents"
    ) -> str:
        """
        Format the prompt based on response strategy
        
        Args:
            query: User query
            context: Retrieved context
            strategy: Response strategy from determine_response_strategy
            system_role: Base system role
            
        Returns:
            Formatted prompt string
        """
        # Build MINIMAL dynamic instructions based on strategy
        # Avoid duplicating what's already in the system role
        instructions = []
        
        # Only add the most essential intent-specific guidance
        if strategy["intent"] == "quick_fact":
            instructions.append("Provide a direct, concise answer focusing only on the specific fact requested.")
            
        elif strategy["intent"] == "targeted":
            instructions.append("Focus on the specific aspect mentioned while providing relevant context.")
            
        elif strategy["intent"] == "full":
            if strategy.get("use_framework", False):
                # Only add framework if truly needed for complex analysis
                framework_prompt = """
Use this structure for your comprehensive analysis:
1. Facts & Evidence: Key information from documents
2. Context & Background: Relevant circumstances  
3. Analysis: Critical examination
4. Implications: What this means
5. Recommendations: Actionable insights (if applicable)

"""
            else:
                framework_prompt = ""
                instructions.append("Provide comprehensive coverage of all relevant aspects.")
            
        elif strategy["intent"] == "conversational":
            instructions.append("Respond naturally and conversationally.")
            
        elif strategy["intent"] == "clarification":
            instructions.append("Address the follow-up question directly, referencing previous context.")
        
        # Only add style modifier if it significantly changes the approach
        style_modifiers = {
            "direct": "Be extremely concise.",
            "analytical": "Include analytical depth with reasoning.",
            "comprehensive": "Cover multiple perspectives thoroughly.",
            "comparative": "Compare and contrast information from different sources."
        }
        
        if strategy["response_style"] in style_modifiers:
            instructions.append(style_modifiers[strategy["response_style"]])
        
        # Only mention uncertainty if confidence is low
        if strategy["confidence"] == "low" and strategy["include_caveats"]:
            instructions.append("Note any limitations in the available information.")
        
        # Add instruction for both modes - UI handles Sources display but allow inline citations
        instructions.append("Include inline citations [1][2][3] throughout your response to reference sources. DO NOT include a separate Sources section at the end. The UI will handle displaying sources with clickable links separately.")
        
        # Construct a MINIMAL prompt
        # Skip system_role as it's sent separately in messages
        instruction_text = ""
        if instructions:
            instruction_text = f"Approach: {' '.join(instructions)}\n\n"
        
        # Use simpler formatting
        enhanced_prompt = f"""{instruction_text}Target length: ~{strategy["max_response_length"]} words
{framework_prompt if 'framework_prompt' in locals() else ''}
Context from documents:
{context}

Question: {query}"""
        
        return enhanced_prompt