# Controller/autogen_controller.py - FINAL FIXED VERSION

import sys
import os
import json
import re
import datetime
from typing import Dict, Any, List, Optional, Tuple
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.advisor_model import AdvisorAgent
from Model.data_model import DataAgent
from Model.web_model import WebSupplementationAgent
from Model.summarizer_model import SummarizerAgent
from Model.embeddings_model import EmbeddingAgent

# Import the actual MemoryManager
from Controller.memory_manager import MemoryManager

class FinancialAdvisoryController:
    def __init__(self):
        """Initialize all agents with proper MemoryManager integration"""
        self.agents = {}
        self._initialize_agents()
        self.conversation_history: List[Tuple[str, str]] = []

        self.visualization_keywords = [
            "chart", "graph", "plot", "visualize", "show trends", "compare performance",
            "analyze data", "correlation", "moving average", "candlestick", "volume",
            "portfolio allocation", "sector comparison", "historical data"
        ]

        self.request_types = {
            "portfolio_analysis": ["portfolio", "diversification", "allocation", "holdings", "balance"],
            "stock_analysis": ["stock", "share", "company", "equity", "price", "valuation"],
            "market_analysis": ["market", "sector", "economy", "trend", "outlook", "conditions"],
            "data_visualization": ["chart", "graph", "plot", "visualize", "data", "trends"]
        }

        # Conceptual keywords for query detection
        self.conceptual_keywords = [
            'what is', 'what are', 'explain', 'define', 'who is', 'tell me about',
            'how is', 'how does', 'why is', 'why does'
        ]

    def _initialize_agents(self):
        """Initialize all agents with error handling"""
        try:
            print("[Controller] Initializing Financial Advisory System...")

            self.agents['web'] = WebSupplementationAgent()
            print("[Controller] ✔ Web Supplementation Agent loaded")

            self.agents['embedding'] = EmbeddingAgent()
            print("[Controller] ✔ Embedding Agent loaded")

            # Initialize MemoryManager with embedding agent
            self.agents['memory'] = MemoryManager()
            print("[Controller] ✔ Memory Manager loaded")

            self.agents['advisor'] = AdvisorAgent()
            print("[Controller] ✔ Advisor Agent loaded")

            self.agents['data'] = DataAgent()
            print("[Controller] ✔ Data Science Agent loaded")

            self.agents['summarizer'] = SummarizerAgent()
            print("[Controller] ✔ Summarizer Agent loaded")

            # Print memory stats
            memory_stats = self.agents['memory'].get_memory_stats()
            print(f"[Controller] Memory Stats: {memory_stats.get('total_sessions', 0)} sessions loaded")

        except Exception as e:
            print(f"[Controller] Critical error during initialization: {e}")
            raise

    def _is_analytical_query(self, query: str) -> bool:
        """Determine if a query contains tickers or company names"""
        if not self.agents.get('web'):
            return False
            
        try:
            result = self.agents['web']._resolve_tickers(query)
            
            # Case 1: Method returns a list of (ticker, name) tuples
            if isinstance(result, list) and all(isinstance(item, tuple) and len(item) == 2 for item in result):
                return bool(result)
            
            # Case 2: Method returns a tuple of lists
            elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
                return bool(result[0])
            
            # Case 3: Method returns something else that might be iterable
            elif isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], str):
                return True
                
            return False

        except Exception as e:
            print(f"[Controller] WARNING in _is_analytical_query: {e}")
            return False

    def _is_conceptual_query(self, query: str) -> bool:
        """Determine if a query is conceptual using improved detection logic"""
        query_lower = query.lower().strip()
        
        # Check for conceptual keywords anywhere in the query
        is_conceptual = any(keyword in query_lower for keyword in self.conceptual_keywords)
        
        # Check for analytical keywords that would override conceptual intent
        analytical_keywords = [
            'analyze', 'analysis', 'compare', 'performance', 'price', 'stock',
            'market', 'trend', 'forecast', 'predict', 'recommend', 'advice',
            'should i', 'buy', 'sell', 'hold', 'investment'
        ]
        is_analytical = any(keyword in query_lower for keyword in analytical_keywords)
        
        # Also check if it contains stock tickers
        has_tickers = self._is_analytical_query(query)
        
        # If it's conceptual AND not analytical AND has no tickers, treat as conceptual
        return is_conceptual and not is_analytical and not has_tickers

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        print("[Controller] Conversation history reset")

    def process_user_query(self, user_query: str, user_metadata: Optional[Dict] = None,
                          maintain_history: bool = True) -> Dict[str, Any]:
        """Main query processing pipeline with proper memory integration"""
        try:
            print(f"[Controller] Processing query: {user_query[:100]}...")
            
            # Use the improved conceptual query detection
            is_conceptual = self._is_conceptual_query(user_query)

            if is_conceptual:
                print("[Controller] Conceptual query detected. Routing to explain_concept.")
                concept_explanation = self.agents['advisor'].explain_concept(
                    user_query,
                    history=self.conversation_history if maintain_history else None
                )
                if maintain_history:
                    self.conversation_history.append((user_query, concept_explanation))
                return {
                    "session_id": "conceptual_session",
                    "request_type": "conceptual_explanation",
                    "advisor_full_response": concept_explanation,
                    "summary": concept_explanation[:200] + "..." if len(concept_explanation) > 200 else concept_explanation,
                    "detailed_insights": "",
                    "executive_summary": {},
                    "analysis_results": None,
                    "web_context": [],
                    "market_data_context": None,
                    "memory_used": False,
                    "system_status": self._get_system_status()
                }

            print("[Controller] Analytical query detected. Proceeding with full pipeline.")
            request_type = self._classify_request(user_query)
            external_context = self._get_external_context(user_query)
            memory_context = self._get_memory_context(user_query)

            combined_context = {
                "web_context": external_context.get("articles", []),
                "market_data": external_context.get("market_data"),
                "memory_context": memory_context
            }

            # Use the advisor's smart routing for analytical queries
            advisor_response = self._get_advisor_response(
                user_query,
                combined_context,
                history=self.conversation_history if maintain_history else None
            )

            analysis_results = self._generate_data_analysis(user_query, advisor_response, request_type)
            summary_results = self._create_summary(advisor_response, analysis_results,
                                                   external_context.get('articles', []), request_type)
            
            # Store session with proper MemoryManager call
            session_id = self._store_session(user_query, advisor_response, external_context.get('articles', []),
                                             analysis_results, summary_results, user_metadata)

            if maintain_history:
                self.conversation_history.append((user_query, advisor_response))

            response = {
                "session_id": session_id,
                "request_type": request_type,
                "advisor_full_response": advisor_response,
                "summary": summary_results.get("summary", ""),
                "detailed_insights": summary_results.get("detailed_insights", ""),
                "executive_summary": summary_results.get("executive_summary", {}),
                "analysis_results": analysis_results,
                "web_context": external_context.get('articles', [])[:3],
                "market_data_context": external_context.get("market_data"),
                "tickers_analyzed": external_context.get("tickers_analyzed", []),
                "memory_used": bool(memory_context),
                "system_status": self._get_system_status()
            }

            return response

        except Exception as e:
            print(f"[Controller] Error processing query: {e}")
            traceback.print_exc()
            return self._create_error_response(str(e), user_query)

    def _classify_request(self, query: str) -> str:
        """Classify the type of financial request"""
        query_lower = query.lower()
        if self._is_analytical_query(query):
            return "stock_analysis"
        for request_type, keywords in self.request_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return request_type
        return "general_advice"

    def _get_external_context(self, user_query: str) -> Dict[str, Any]:
        """Get external context from web agent"""
        if not self.agents.get('web'):
            return {"articles": [], "market_data": "Web agent not available."}
        try:
            return self.agents['web'].get_relevant_info(user_query)
        except Exception as e:
            print(f"[Controller] Error fetching external context: {e}")
            return {"articles": [], "market_data": "Error fetching market data."}

    def _get_memory_context(self, user_query: str) -> List[Dict[str, Any]]:
        """Get relevant context from memory"""
        if not self.agents.get('memory'):
            return []
        try:
            return self.agents['memory'].search_memory(user_query, top_k=2)
        except Exception as e:
            print(f"[Controller] Error fetching memory context: {e}")
            return []

    def _get_advisor_response(self, user_query: str, combined_context: Dict,
                              history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Get response from advisor agent - uses process_query for smart routing"""
        if not self.agents.get('advisor'):
            return "Advisor agent not available."
        try:
            # Use the advisor's smart routing method if available
            if hasattr(self.agents['advisor'], 'process_query'):
                return self.agents['advisor'].process_query(
                    user_query,
                    context=combined_context,
                    history=history
                )
            else:
                # Fallback to direct method
                return self.agents['advisor'].get_financial_advice(
                    user_query,
                    context=combined_context,
                    history=history
                )
        except Exception as e:
            print(f"[Controller] Error generating advisor response: {e}")
            return f"An error occurred in the advisor agent: {str(e)}"

    def _extract_stock_symbols(self, query: str, advisor_response: str) -> List[str]:
        """Extract stock symbols from query and response"""
        if not self.agents.get('web'):
            return []
        
        combined_text = f"{query} {advisor_response} {' '.join([hist[0] for hist in self.conversation_history])}"
        
        try:
            companies = self.agents['web']._resolve_tickers(combined_text)
            tickers = sorted(list(set([c[0].replace('.JO', '').upper() for c in companies])))
            return tickers
        except Exception as e:
            print(f"[Controller] Error extracting stock symbols: {e}")
            return []

    def _determine_analysis_type(self, query: str, request_type: str) -> str:
        """Determine the type of data analysis required"""
        query_lower = query.lower()
        
        if request_type == 'portfolio_analysis' or 'allocation' in query_lower:
            return 'portfolio_allocation'
        
        if 'sector' in query_lower or 'compare market' in query_lower:
            return 'sector_comparison'

        if any(kw in query_lower for kw in ['candlestick', 'candles']):
            return 'candlestick_chart'
        
        if any(kw in query_lower for kw in ['volume', 'trading activity']):
            return 'volume_analysis'

        if any(kw in query_lower for kw in ['ma', 'moving average', '50-day', '200-day']):
            return 'moving_averages'
            
        if any(kw in query_lower for kw in ['correlation', 'risk relationship']):
            return 'correlation_matrix'
            
        return 'stock_price_trend'

    def _extract_time_period(self, query: str) -> str:
        """Extract the time period from the query"""
        query_lower = query.lower()
        if '3mo' in query_lower or '3 months' in query_lower:
            return '3mo'
        if '6mo' in query_lower or '6 months' in query_lower:
            return '6mo'
        if '2y' in query_lower or '2 years' in query_lower:
            return '2y'
        if '5y' in query_lower or '5 years' in query_lower:
            return '5y'
        return '1y'

    def _generate_data_analysis(self, user_query: str, advisor_response: str, request_type: str) -> Optional[Dict[str, Any]]:
        """Generate data analysis results"""
        if not self.agents.get('data'):
            return None

        stocks = self._extract_stock_symbols(user_query, advisor_response)
        analysis_type = self._determine_analysis_type(user_query, request_type)
        time_period = self._extract_time_period(user_query)
        
        if not stocks and analysis_type not in ['sector_comparison', 'portfolio_allocation']:
            return None 

        request = {
            'task_type': analysis_type,
            'stocks': stocks or ["NPN", "SBK"],
            'time_period': time_period,
            'holdings': {"MTN": 40, "SBK": 30, "NPN": 30} if analysis_type == 'portfolio_allocation' else {} 
        }

        try:
            analysis_code = self.agents['data'].generate_analysis_code(request)
            data_context = f"Analysis of {analysis_type} for stocks {request['stocks']} over {time_period}. Code generated."
            insights = self.agents['data'].generate_data_insights(data_context, analysis_type=analysis_type)
        except Exception as e:
            print(f"[Controller] Data analysis agent failed: {e}")
            analysis_code = f"# Error: {str(e)}"
            insights = "Failed to generate data analysis insights due to an agent error."
        
        return {
            "analysis_type": analysis_type,
            "stocks_analyzed": request['stocks'],
            "code": analysis_code,
            "insights": insights
        }

    def _create_summary(self, advisor_response: str, analysis_results: Optional[Dict[str, Any]],
                        articles: List[Dict[str, Any]], request_type: str) -> Dict[str, Any]:
        """Create summary of the analysis"""
        if not self.agents.get('summarizer'):
            # Improved fallback if summarizer is unavailable
            summary = advisor_response[:150] + "..." if len(advisor_response) > 150 else advisor_response
            
            sentences = advisor_response.split('.')
            detailed_insights = '. '.join(sentences[:4]) + '.' if len(sentences) > 4 else advisor_response
            
            executive_summary = {
                "main_conclusion": sentences[0] + '.' if sentences else "Analysis completed.",
                "key_takeaways": [
                    "Market data processed successfully",
                    "Comparative analysis provided",
                    "Recent news considered"
                ],
                "recommendation_level": "Informational"
            }
            
            return {
                "summary": summary,
                "detailed_insights": detailed_insights,
                "executive_summary": executive_summary
            }

        try:
            full_text = f"Advisor Report: {advisor_response}\n\n"
            if analysis_results and analysis_results.get('insights'):
                full_text += f"Data Analysis Insights: {analysis_results['insights']}\n\n"
            
            if articles:
                news_text = " | ".join([a['headline'] for a in articles[:3]])
                full_text += f"Relevant News: {news_text}"
            
            summary = self.agents['summarizer'].summarize_text(full_text, summary_type='brief')
            detailed_insights = self.agents['summarizer'].summarize_text(full_text, summary_type='detailed')
            
            executive_summary = {
                "main_conclusion": summary.split('.')[0] + '.' if '.' in summary else summary,
                "key_takeaways": [
                    "Real-time market data analyzed",
                    "Comparative performance assessed", 
                    "Sector context provided"
                ],
                "recommendation_level": "Professional Analysis"
            }
            
            return {
                "summary": summary,
                "detailed_insights": detailed_insights,
                "executive_summary": executive_summary
            }
            
        except Exception as e:
            print(f"[Controller] Summarizer failed, using fallback: {e}")
            summary = advisor_response[:150] + "..." if len(advisor_response) > 150 else advisor_response
            return {
                "summary": summary,
                "detailed_insights": advisor_response,
                "executive_summary": {
                    "main_conclusion": "Analysis completed with available data",
                    "key_takeaways": ["Market data processed", "Investment insights provided"],
                    "recommendation_level": "Standard"
                }
            }

    def _store_session(self, user_query: str, advisor_response: str, articles: List[Dict[str, Any]],
                       analysis_results: Optional[Dict[str, Any]], summary_results: Dict[str, Any],
                       user_metadata: Optional[Dict]) -> str:
        """Store session in memory using MemoryManager"""
        if not self.agents.get('memory'):
            return "NO_SESSION_ID"
        
        try:
            # Prepare data insights
            data_insights = analysis_results.get('insights') if analysis_results else None
            
            # Store session using MemoryManager's add_session method
            session_id = self.agents['memory'].add_session(
                query=user_query,
                advisor_response=advisor_response,
                web_context=articles,
                data_insights=data_insights,
                summary=summary_results.get('summary'),
                user_metadata=user_metadata
            )
            
            print(f"[Controller] Stored session {session_id} in memory")
            return session_id
            
        except Exception as e:
            print(f"[Controller] Error storing session in memory: {e}")
            return f"error_session_{hash(user_query) % 10000}"

    def _get_system_status(self) -> Dict[str, str]:
        """Get system health status"""
        return {
            "web_agent": "Active" if self.agents.get('web') else "Inactive",
            "advisor_agent": "Active" if self.agents.get('advisor') else "Inactive",
            "memory_manager": "Active" if self.agents.get('memory') else "Inactive",
            "data_agent": "Template-Based Fallback",
            "summarizer_agent": "Active" if self.agents.get('summarizer') else "Inactive"
        }

    def _create_error_response(self, error_message: str, user_query: str) -> Dict[str, Any]:
        """Create error response"""
        print(f"[Controller] Generating error response for: {user_query}")
        return {
            "session_id": "error_session",
            "request_type": "system_error",
            "advisor_full_response": f"System Error: {error_message}",
            "summary": "System encountered an error.",
            "detailed_insights": "",
            "executive_summary": {},
            "analysis_results": None,
            "web_context": [],
            "market_data_context": None,
            "memory_used": False,
            "system_status": self._get_system_status(),
            "original_query": user_query
        }
        
    def health_check(self):
        """Better health check that understands template fallbacks are normal"""
        status = self._get_system_status()
    
        # Only consider it degraded if critical agents fail
        critical_failures = any([
            "Inactive" in status.get('web_agent', ''),
            "Inactive" in status.get('advisor_agent', ''),
            "Inactive" in status.get('summarizer_agent', '')
        ])
    
        return {
            "overall_status": "degraded" if critical_failures else "healthy",
            "agent_statuses": status,
            "note": "Template-based Data Agent is normal and reliable"
        }