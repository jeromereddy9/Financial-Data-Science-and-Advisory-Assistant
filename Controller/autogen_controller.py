# Controller/autogen_controller.py

import sys
import os
import json
import re
from typing import Dict, Any, List, Optional
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.advisor_model import AdvisorAgent
from Model.data_model import DataAgent
from Model.web_model import WebSupplementationAgent
from Model.summarizer_model import SummarizerAgent
from .memory_manager import MemoryManager
from Model.embeddings_model import EmbeddingAgent

class FinancialAdvisoryController:
    def __init__(self):
        """Initialize all agents with proper error handling"""
        self.agents = {}
        self._initialize_agents()
        
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

        self.conceptual_keywords = [
            'what is', 'what are', 'explain', 'define', 'who is', 'tell me about'
        ]

        # **FIX**: Add a ticker list here for the router to use.
        # This decouples the controller's routing logic from the Web Agent.
        self.jse_ticker_list = [
            'AGL', 'AMS', 'ANG', 'APN', 'ARI', 'BHP', 'BID', 'BTI', 'BVT', 'CFR', 
            'CLS', 'CPI', 'DSY', 'EXX', 'FSR', 'GFI', 'GLN', 'GRT', 'HAR', 'IMP', 
            'INL', 'INP', 'MCG', 'MNP', 'MRP', 'MTN', 'NED', 'NPH', 'NPN', 'NRP', 
            'OMU', 'OUT', 'PIK', 'PPH', 'PRX', 'REM', 'RLO', 'SBK', 'SHP', 'SLM', 
            'SOL', 'SSW', 'TFG', 'TRU', 'VOD', 'WHL'
        ]
    
    def _initialize_agents(self):
        """Initialize all agents with proper error handling"""
        try:
            print("[Controller] Initializing Financial Advisory System...")
            
            self.agents['web'] = WebSupplementationAgent()
            print("[Controller] ? Web Supplementation Agent loaded")
            
            self.agents['embedding'] = EmbeddingAgent()
            print("[Controller] ? Embedding Agent loaded")
            
            self.agents['memory'] = MemoryManager()
            print("[Controller] ? Memory Manager loaded")
            
            self.agents['advisor'] = AdvisorAgent()
            print("[Controller] ? Advisor Agent loaded")
            
            self.agents['data'] = DataAgent()
            print("[Controller] ? Data Science Agent loaded")
            
            self.agents['summarizer'] = SummarizerAgent()
            print("[Controller] ? Summarizer Agent loaded")
            
            print(f"[Controller] System initialized with {sum(1 for agent in self.agents.values() if agent is not None)}/6 agents")
            
        except Exception as e:
            print(f"[Controller] Critical error during initialization: {e}")
            raise

    def _is_analytical_query(self, query: str) -> bool:
        """
        **NEW**: Helper method to gracefully determine if a query is analytical.
        It checks for the presence of any known JSE tickers as whole words.
        """
        # Using regex with word boundaries (\b) is the most robust way to find tickers
        # and avoid matching substrings (e.g., 'IMP' in 'IMPORTANT').
        return any(re.search(r'\b' + ticker + r'\b', query, re.IGNORECASE) for ticker in self.jse_ticker_list)

    def process_user_query(self, user_query: str, user_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main processing pipeline with the upgraded, more graceful Query Router.
        """
        try:
            print(f"[Controller] Processing query: {user_query[:100]}...")
            
            # **FIX**: Upgraded Query Router Logic
            is_analytical = self._is_analytical_query(user_query)
            starts_with_conceptual_keyword = any(user_query.lower().strip().startswith(key) for key in self.conceptual_keywords)

            # A query is conceptual ONLY if it starts with a keyword AND is NOT analytical.
            if starts_with_conceptual_keyword and not is_analytical:
                print("[Controller] Conceptual query detected. Routing to explain_concept.")
                
                concept_explanation = self.agents['advisor'].explain_concept(user_query)
                
                return {
                    "session_id": "conceptual_session",
                    "request_type": "conceptual_explanation",
                    "advisor_full_response": concept_explanation,
                    "summary": concept_explanation[:200] + "...",
                    "detailed_insights": "",
                    "executive_summary": {},
                    "analysis_results": None,
                    "web_context": [],
                    "market_data_context": None,
                    "memory_used": False,
                    "system_status": self._get_system_status()
                }

            # Default to the analytical pipeline for all other cases.
            print("[Controller] Analytical query detected. Proceeding with full pipeline.")
            request_type = self._classify_request(user_query)
            
            external_context = self._get_external_context(user_query)
            memory_context = self._get_memory_context(user_query)
            
            combined_context = {
                "web_context": external_context,
                "market_data": external_context.get("market_data"),
                "memory_context": memory_context
            }

            advisor_response = self._get_advisor_response(user_query, combined_context)
            
            analysis_results = self._generate_data_analysis(user_query, advisor_response, request_type)
            
            summary_results = self._create_summary(advisor_response, analysis_results, external_context.get('articles', []), request_type)
            
            session_id = self._store_session(user_query, advisor_response, external_context.get('articles', []), 
                                           analysis_results, summary_results, user_metadata)
            
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
                "memory_used": bool(memory_context),
                "system_status": self._get_system_status()
            }
            
            print(f"[Controller] Query processed successfully (session: {session_id})")
            return response
            
        except Exception as e:
            print(f"[Controller] Error processing query: {e}")
            traceback.print_exc()
            return self._create_error_response(str(e), user_query)
    
    def _classify_request(self, query: str) -> str:
        """Classify the type of financial request"""
        query_lower = query.lower()
        
        for request_type, keywords in self.request_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return request_type
        
        return "general_advice"
    
    def _get_external_context(self, user_query: str) -> Dict[str, Any]:
        """
        Fetch all external context using the WebSupplementationAgent Facade.
        """
        if not self.agents.get('web'):
            return {"articles": [], "market_data": "Web agent not available."}
        
        try:
            return self.agents['web'].get_relevant_info(user_query, max_articles=3)
        except Exception as e:
            print(f"[Controller] Error fetching external context: {e}")
            return {"articles": [], "market_data": "Error fetching market data."}

    def _get_memory_context(self, user_query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from previous sessions"""
        if not self.agents.get('memory'):
            return []
        
        try:
            return self.agents['memory'].search_memory(user_query, top_k=2)
        except Exception as e:
            print(f"[Controller] Error fetching memory context: {e}")
            return []
    
    def _get_advisor_response(self, user_query: str, combined_context: Dict) -> str:
        """Generate financial advice using the Advisor Agent with all context."""
        if not self.agents.get('advisor'):
            return "Advisor agent not available."
        
        try:
            return self.agents['advisor'].get_financial_advice(user_query, context=combined_context)
        except Exception as e:
            print(f"[Controller] Error generating advisor response: {e}")
            return f"An error occurred in the advisor agent: {str(e)}"

    def _generate_data_analysis(self, user_query: str, advisor_response: str, 
                              request_type: str) -> Optional[Dict[str, Any]]:
        """Generate data analysis and visualizations if needed"""
        if not self.agents.get('data'):
            return None
        
        try:
            needs_visualization = (
                request_type == "data_visualization" or
                any(keyword in advisor_response.lower() for keyword in self.visualization_keywords) or
                any(keyword in user_query.lower() for keyword in self.visualization_keywords)
            )
            
            if not needs_visualization:
                return None
            
            stocks = self._extract_stock_symbols(user_query + " " + advisor_response)
            
            analysis_type = self._determine_analysis_type(user_query, advisor_response)
            
            analysis_request = {
                'task_type': analysis_type,
                'stocks': stocks if stocks else ['AGL', 'SBK', 'NPN'],
                'time_period': self._extract_time_period(user_query),
                'query_context': user_query
            }
            
            analysis_code = self.agents['data'].generate_analysis_code(analysis_request)
            
            data_insights = self.agents['data'].generate_data_insights(
                f"Query: {user_query} | Response: {advisor_response[:200]}...",
                analysis_type
            )
            
            return {
                'analysis_code': analysis_code,
                'data_insights': data_insights,
                'analysis_request': analysis_request,
                'chart_type': analysis_type
            }
            
        except Exception as e:
            print(f"[Controller] Error generating data analysis: {e}")
            return {
                'error': str(e),
                'fallback_message': 'Data visualization unavailable.'
            }
    
    def _create_summary(self, advisor_response: str, analysis_results: Optional[Dict],
                       web_context: List[Dict], request_type: str) -> Dict[str, str]:
        """Create comprehensive summary using the Summarizer Agent"""
        if not self.agents.get('summarizer'):
            return {
                "summary": advisor_response[:200] + "...",
                "detailed_insights": "Please refer to the full advisor response.",
                "executive_summary": {}
            }
        
        try:
            summary, detailed_insights = self.agents['summarizer'].create_insights(
                advisor_response, 
                advice_type=request_type
            )
            
            data_insights_text = analysis_results.get('data_insights', '') if analysis_results else ''
            
            web_context_text = ' '.join([article.get('summary', '') for article in web_context[:3]]) if web_context else ''
            
            executive_summary = self.agents['summarizer'].create_executive_summary(
                advisor_response=advisor_response,
                data_insights=data_insights_text,
                context_info=web_context_text
            )
            
            return {
                "summary": summary,
                "detailed_insights": detailed_insights,
                "executive_summary": executive_summary
            }
            
        except Exception as e:
            print(f"[Controller] Error creating summary: {e}")
            return {
                "summary": advisor_response[:150] + "...",
                "detailed_insights": "Summary generation failed.",
                "executive_summary": {"error": str(e)}
            }
    
    def _store_session(self, user_query: str, advisor_response: str, web_context: List[Dict],
                      analysis_results: Optional[Dict], summary_results: Dict,
                      user_metadata: Optional[Dict]) -> str:
        """Store the complete session in memory"""
        if not self.agents.get('memory'):
            return "memory_unavailable"
        
        try:
            return self.agents['memory'].add_session(
                query=user_query,
                advisor_response=advisor_response,
                web_context=web_context,
                data_insights=analysis_results.get('data_insights', '') if analysis_results else '',
                summary=summary_results.get('summary', ''),
                user_metadata=user_metadata
            )
        except Exception as e:
            print(f"[Controller] Error storing session: {e}")
            return "storage_error"
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract JSE stock symbols from text"""
        # This is a simplified version for the controller. The Web Agent has the primary one.
        mentioned_stocks = []
        text_upper = text.upper()
        
        for stock in self.jse_ticker_list:
            if re.search(r'\b' + stock + r'\b', text_upper):
                mentioned_stocks.append(stock)
        
        return list(set(mentioned_stocks))[:5]
    
    def _determine_analysis_type(self, query: str, response: str) -> str:
        """Determine the type of data analysis needed"""
        combined_text = (query + " " + response).lower()
        
        if any(word in combined_text for word in ['portfolio', 'allocation', 'diversification']):
            return 'portfolio_allocation'
        elif any(word in combined_text for word in ['candlestick', 'technical']):
            return 'candlestick_chart'
        elif any(word in combined_text for word in ['volume', 'trading']):
            return 'volume_analysis'
        elif any(word in combined_text for word in ['sector', 'compare', 'comparison']):
            return 'sector_comparison'
        elif any(word in combined_text for word in ['correlation', 'relationship']):
            return 'correlation_matrix'
        elif any(word in combined_text for word in ['moving average', 'trend']):
            return 'moving_averages'
        else:
            return 'stock_price_trend'
    
    def _extract_time_period(self, query: str) -> str:
        """Extract time period from query text"""
        query_lower = query.lower()
        
        if any(period in query_lower for period in ['1 month', 'month', '1m']):
            return '1mo'
        elif any(period in query_lower for period in ['3 month', '3m', 'quarter']):
            return '3mo'
        elif any(period in query_lower for period in ['6 month', '6m', 'half year']):
            return '6mo'
        elif any(period in query_lower for period in ['2 year', '2y', 'two year']):
            return '2y'
        elif any(period in query_lower for period in ['5 year', '5y', 'five year']):
            return '5y'
        else:
            return '1y'
    
    def _get_system_status(self) -> Dict[str, bool]:
        """Get status of all system components"""
        return {
            "web_agent": self.agents.get('web') is not None,
            "embedding_agent": self.agents.get('embedding') is not None,
            "memory_manager": self.agents.get('memory') is not None,
            "advisor_agent": self.agents.get('advisor') is not None,
            "data_agent": self.agents.get('data') is not None,
            "summarizer_agent": self.agents.get('summarizer') is not None
        }
    
    def _create_error_response(self, error_message: str, user_query: str) -> Dict[str, Any]:
        """Create error response when system fails"""
        return {
            "session_id": "error_session",
            "request_type": "error",
            "advisor_full_response": f"I apologize, but I encountered an error processing your query: '{user_query}'. Please try again or contact support.",
            "summary": "System error occurred",
            "detailed_insights": f"Error details: {error_message}",
            "executive_summary": {"error": error_message},
            "analysis_results": None,
            "web_context": [],
            "memory_used": False,
            "system_status": self._get_system_status(),
            "error": True
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "system_name": "Financial Data Science & Advisory Assistant",
            "version": "1.0.0",
            "target_market": "Johannesburg Stock Exchange (JSE)",
            "agents": {},
            "system_status": self._get_system_status()
        }
        
        for agent_name, agent in self.agents.items():
            if agent and hasattr(agent, 'get_model_info'):
                try:
                    info["agents"][agent_name] = agent.get_model_info()
                except:
                    info["agents"][agent_name] = {"status": "loaded", "info": "unavailable"}
            else:
                info["agents"][agent_name] = {"status": "not_loaded"}
        
        if self.agents.get('memory'):
            try:
                info["memory_stats"] = self.agents['memory'].get_memory_stats()
            except:
                info["memory_stats"] = {"error": "unable to retrieve stats"}
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            "overall_status": "healthy",
            "agent_health": {},
            "critical_issues": [],
            "warnings": []
        }
        
        for agent_name, agent in self.agents.items():
            if agent is None:
                health["agent_health"][agent_name] = "failed"
                health["critical_issues"].append(f"{agent_name} failed to load")
            else:
                health["agent_health"][agent_name] = "healthy"
        
        if health["critical_issues"]:
            health["overall_status"] = "degraded" if len(health["critical_issues"]) < 3 else "critical"
        
        essential_agents = ['advisor', 'memory']
        for agent in essential_agents:
            if health["agent_health"].get(agent) != "healthy":
                health["overall_status"] = "critical"
                health["critical_issues"].append(f"Essential agent {agent} is not healthy")
        
        return health