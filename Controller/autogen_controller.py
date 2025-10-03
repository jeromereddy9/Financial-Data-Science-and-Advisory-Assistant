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
    """
    Main controller orchestrating the Financial Data Science & Advisory Assistant.
    Integrates all specialized agents to provide comprehensive JSE investment guidance.
    """
    
    def __init__(self):
        """Initialize all agents with proper error handling"""
        self.agents = {}
        self._initialize_agents()
        
        # Analysis keywords for triggering data visualizations
        self.visualization_keywords = [
            "chart", "graph", "plot", "visualize", "show trends", "compare performance",
            "analyze data", "correlation", "moving average", "candlestick", "volume",
            "portfolio allocation", "sector comparison", "historical data"
        ]
        
        # Request type classification
        self.request_types = {
            "portfolio_analysis": ["portfolio", "diversification", "allocation", "holdings", "balance"],
            "stock_analysis": ["stock", "share", "company", "equity", "price", "valuation"],
            "market_analysis": ["market", "sector", "economy", "trend", "outlook", "conditions"],
            "data_visualization": ["chart", "graph", "plot", "visualize", "data", "trends"]
        }
    
    def _initialize_agents(self):
        """Initialize all agents with proper error handling"""
        try:
            print("[Controller] Initializing Financial Advisory System...")
            
            # Initialize Web Supplementation Agent
            try:
                self.agents['web'] = WebSupplementationAgent()
                print("[Controller] ? Web Supplementation Agent loaded")
            except Exception as e:
                print(f"[Controller] ? Failed to load Web Agent: {e}")
                self.agents['web'] = None
            
            # Initialize Embedding Agent
            try:
                self.agents['embedding'] = EmbeddingAgent()
                print("[Controller] ? Embedding Agent loaded")
            except Exception as e:
                print(f"[Controller] ? Failed to load Embedding Agent: {e}")
                self.agents['embedding'] = None
            
            # Initialize Memory Manager (depends on embedding agent)
            try:
                self.agents['memory'] = MemoryManager()
                print("[Controller] ? Memory Manager loaded")
            except Exception as e:
                print(f"[Controller] ? Failed to load Memory Manager: {e}")
                self.agents['memory'] = None
            
            # Initialize Advisor Agent
            try:
                self.agents['advisor'] = AdvisorAgent()
                print("[Controller] ? Advisor Agent loaded")
            except Exception as e:
                print(f"[Controller] ? Failed to load Advisor Agent: {e}")
                self.agents['advisor'] = None
            
            # Initialize Data Agent
            try:
                self.agents['data'] = DataAgent()
                print("[Controller] ? Data Science Agent loaded")
            except Exception as e:
                print(f"[Controller] ? Failed to load Data Agent: {e}")
                self.agents['data'] = None
            
            # Initialize Summarizer Agent
            try:
                self.agents['summarizer'] = SummarizerAgent()
                print("[Controller] ? Summarizer Agent loaded")
            except Exception as e:
                print(f"[Controller] ? Failed to load Summarizer Agent: {e}")
                self.agents['summarizer'] = None
            
            print(f"[Controller] System initialized with {sum(1 for agent in self.agents.values() if agent is not None)}/6 agents")
            
        except Exception as e:
            print(f"[Controller] Critical error during initialization: {e}")
            raise
    
    def process_user_query(self, user_query: str, user_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main processing pipeline for user financial queries.
        
        Args:
            user_query: User's financial question or request
            user_metadata: Optional metadata about the user/session
        
        Returns:
            Dictionary containing full response, summary, insights, and analysis
        """
        try:
            print(f"[Controller] Processing query: {user_query[:100]}...")
            
            # Classify request type
            request_type = self._classify_request(user_query)
            
            # Step 1: Fetch relevant web context
            web_context = self._get_web_context(user_query)
            
            # Step 2: Retrieve relevant memory context
            memory_context = self._get_memory_context(user_query)
            
            # Step 3: Generate financial advice using combined context
            advisor_response = self._get_advisor_response(user_query, web_context, memory_context)
            
            # Step 4: Generate data analysis if needed
            analysis_results = self._generate_data_analysis(user_query, advisor_response, request_type)
            
            # Step 5: Create comprehensive summary and insights
            summary_results = self._create_summary(advisor_response, analysis_results, web_context, request_type)
            
            # Step 6: Store session in memory
            session_id = self._store_session(user_query, advisor_response, web_context, 
                                           analysis_results, summary_results, user_metadata)
            
            # Prepare final response
            response = {
                "session_id": session_id,
                "request_type": request_type,
                "advisor_full_response": advisor_response,
                "summary": summary_results.get("summary", ""),
                "detailed_insights": summary_results.get("detailed_insights", ""),
                "executive_summary": summary_results.get("executive_summary", {}),
                "analysis_results": analysis_results,
                "web_context": web_context[:3] if web_context else [],  # Limit for response size
                "memory_used": len(memory_context) > 0,
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
    
    def _get_web_context(self, user_query: str) -> List[Dict[str, Any]]:
        """Fetch relevant web context from financial news sources"""
        if not self.agents.get('web'):
            return []
        
        try:
            web_articles = self.agents['web'].get_relevant_info(user_query, max_articles=5)
            return web_articles if web_articles else []
        except Exception as e:
            print(f"[Controller] Error fetching web context: {e}")
            return []
    
    def _get_memory_context(self, user_query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context from previous sessions"""
        if not self.agents.get('memory'):
            return []
        
        try:
            memory_sessions = self.agents['memory'].search_memory(
                user_query, 
                top_k=3, 
                similarity_threshold=0.3
            )
            return memory_sessions
        except Exception as e:
            print(f"[Controller] Error fetching memory context: {e}")
            return []
    
    def _get_advisor_response(self, user_query: str, web_context: List[Dict], 
                            memory_context: List[Dict]) -> str:
        """Generate financial advice using the Advisor Agent"""
        if not self.agents.get('advisor'):
            return "Advisor agent not available. Please contact system administrator."
        
        try:
            # Structure context for the improved Advisor Agent
            structured_context = {
                'web_context': web_context,
                'memory_context': memory_context
            }
            
            advisor_response = self.agents['advisor'].get_financial_advice(
                user_query, 
                context=structured_context
            )
            
            return advisor_response
            
        except Exception as e:
            print(f"[Controller] Error generating advisor response: {e}")
            return f"I apologize, but I encountered an error while processing your financial query. Please try rephrasing your question or contact support. Error details: {str(e)}"
    
    def _generate_data_analysis(self, user_query: str, advisor_response: str, 
                              request_type: str) -> Optional[Dict[str, Any]]:
        """Generate data analysis and visualizations if needed"""
        if not self.agents.get('data'):
            return None
        
        try:
            # Check if visualization is needed
            needs_visualization = (
                request_type == "data_visualization" or
                any(keyword in advisor_response.lower() for keyword in self.visualization_keywords) or
                any(keyword in user_query.lower() for keyword in self.visualization_keywords)
            )
            
            if not needs_visualization:
                return None
            
            # Extract stocks mentioned in query/response
            stocks = self._extract_stock_symbols(user_query + " " + advisor_response)
            
            # Determine analysis type
            analysis_type = self._determine_analysis_type(user_query, advisor_response)
            
            # Create analysis request
            analysis_request = {
                'task_type': analysis_type,
                'stocks': stocks if stocks else ['AGL', 'SBK', 'NPN'],  # Default JSE stocks
                'time_period': self._extract_time_period(user_query),
                'query_context': user_query
            }
            
            # Generate analysis code
            analysis_code = self.agents['data'].generate_analysis_code(analysis_request)
            
            # Generate data insights
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
                'fallback_message': 'Data visualization unavailable. Please refer to the written analysis above.'
            }
    
    def _create_summary(self, advisor_response: str, analysis_results: Optional[Dict],
                       web_context: List[Dict], request_type: str) -> Dict[str, str]:
        """Create comprehensive summary using the Summarizer Agent"""
        if not self.agents.get('summarizer'):
            # Fallback summary
            return {
                "summary": advisor_response[:200] + "..." if len(advisor_response) > 200 else advisor_response,
                "detailed_insights": "Please refer to the full advisor response above.",
                "executive_summary": {}
            }
        
        try:
            # Create basic summary and insights
            summary, detailed_insights = self.agents['summarizer'].create_insights(
                advisor_response, 
                advice_type=request_type,
                short_max_length=150,
                detailed_max_length=400
            )
            
            # Create executive summary combining all components
            data_insights_text = ""
            if analysis_results:
                data_insights_text = analysis_results.get('data_insights', '')
            
            web_context_text = ""
            if web_context:
                web_summaries = [article.get('summary', '') for article in web_context[:3]]
                web_context_text = ' '.join(web_summaries)
            
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
                "detailed_insights": "Summary generation failed. Please refer to full response.",
                "executive_summary": {"error": str(e)}
            }
    
    def _store_session(self, user_query: str, advisor_response: str, web_context: List[Dict],
                      analysis_results: Optional[Dict], summary_results: Dict,
                      user_metadata: Optional[Dict]) -> str:
        """Store the complete session in memory"""
        if not self.agents.get('memory'):
            return "memory_unavailable"
        
        try:
            session_id = self.agents['memory'].add_session(
                query=user_query,
                advisor_response=advisor_response,
                web_context=web_context,
                data_insights=analysis_results.get('data_insights', '') if analysis_results else '',
                summary=summary_results.get('summary', ''),
                user_metadata=user_metadata
            )
            return session_id
        except Exception as e:
            print(f"[Controller] Error storing session: {e}")
            return "storage_error"
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract JSE stock symbols from text"""
        # Common JSE stock patterns
        jse_stocks = [
            'AGL', 'SBK', 'FSR', 'NED', 'BIL', 'GFI', 'NPN', 'MTN', 'VOD',
            'SHP', 'TRU', 'WHL', 'CLS', 'REM', 'BTI', 'CFR', 'IMP', 'BVT'
        ]
        
        # Find mentioned stocks
        mentioned_stocks = []
        text_upper = text.upper()
        
        for stock in jse_stocks:
            if stock in text_upper:
                mentioned_stocks.append(stock)
        
        # Also look for 3-letter patterns that might be stock codes
        stock_pattern = re.findall(r'\b[A-Z]{3}\b', text.upper())
        for match in stock_pattern:
            if match not in mentioned_stocks and match in jse_stocks:
                mentioned_stocks.append(match)
        
        return mentioned_stocks[:5]  # Limit to 5 stocks
    
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
            return '1y'  # Default to 1 year
    
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
        
        # Get info from each agent
        for agent_name, agent in self.agents.items():
            if agent and hasattr(agent, 'get_model_info'):
                try:
                    info["agents"][agent_name] = agent.get_model_info()
                except:
                    info["agents"][agent_name] = {"status": "loaded", "info": "unavailable"}
            else:
                info["agents"][agent_name] = {"status": "not_loaded"}
        
        # Get memory statistics
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
        
        # Check each agent
        for agent_name, agent in self.agents.items():
            if agent is None:
                health["agent_health"][agent_name] = "failed"
                health["critical_issues"].append(f"{agent_name} failed to load")
            else:
                health["agent_health"][agent_name] = "healthy"
        
        # Overall status
        if health["critical_issues"]:
            if len(health["critical_issues"]) >= 3:
                health["overall_status"] = "critical"
            else:
                health["overall_status"] = "degraded"
        
        # Essential agents check
        essential_agents = ['advisor', 'memory']
        for agent in essential_agents:
            if health["agent_health"].get(agent) != "healthy":
                health["overall_status"] = "critical"
                health["critical_issues"].append(f"Essential agent {agent} is not healthy")
        
        return health