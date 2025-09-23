import os
import warnings
# Suppress TensorFlow and Hugging Face warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow info/warning messages
warnings.filterwarnings("ignore", message="Xet Storage is enabled")  # HF XET warning

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any, Optional
import re

class AdvisorAgent:
    def __init__(self):
        self.model_name = "tiiuae/falcon-rw-1b"  # open alternative to Falcon-1B-Instruct
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto" if torch.cuda.is_available() else None
            ).to(self.device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"[AdvisorAgent] Successfully loaded model: {self.model_name}")
        except Exception as e:
            print(f"[AdvisorAgent] Error loading model: {e}")
            raise
    
    def _process_context(self, context: Dict[str, Any]) -> str:
        """
        Process and structure context from web articles and memory.
        
        Args:
            context: Dictionary containing 'web_context' and 'memory_context'
        
        Returns:
            Formatted context string for the prompt
        """
        if not context:
            return ""
        
        formatted_context = []
        
        # Process web context (recent news/articles)
        if context.get('web_context'):
            web_articles = context['web_context']
            if web_articles:
                formatted_context.append("RECENT MARKET NEWS:")
                for i, article in enumerate(web_articles[:3], 1):  # Limit to top 3
                    summary = article.get('summary', '')
                    source = article.get('source', 'Unknown')
                    if summary:
                        formatted_context.append(f"{i}. {summary} (Source: {source})")
                formatted_context.append("")
        
        # Process memory context (historical sessions)
        if context.get('memory_context'):
            memory_sessions = context['memory_context']
            if memory_sessions:
                formatted_context.append("RELEVANT PREVIOUS ADVICE:")
                for i, session in enumerate(memory_sessions[:2], 1):  # Limit to top 2
                    if isinstance(session, dict):
                        query = session.get('query', '')
                        advice = session.get('advice', '')
                        if query and advice:
                            formatted_context.append(f"{i}. Previous Query: {query}")
                            formatted_context.append(f"   Previous Advice: {advice[:200]}...")
                    elif isinstance(session, str):
                        formatted_context.append(f"{i}. {session[:200]}...")
                formatted_context.append("")
        
        return "\n".join(formatted_context)
    
    def _create_jse_financial_prompt(self, query: str, context: str) -> str:
        """Create a JSE-specific financial advisory prompt"""
        
        prompt = f"""You are an expert financial advisor specializing in the Johannesburg Stock Exchange (JSE) and South African markets. 

IMPORTANT GUIDELINES:
- Provide practical, actionable advice specific to JSE trading and South African investments
- Consider South African economic conditions, rand volatility, and local market dynamics
- Mention relevant JSE sectors (mining, banking, telecommunications, etc.) when appropriate
- Include risk warnings and remind users this is not personalized financial advice
- Consider tax implications for South African residents (capital gains tax, dividend withholding tax)
- Be aware of JSE trading hours (9:00 AM - 5:00 PM SAST) and settlement periods

{context}

USER QUESTION: {query}

FINANCIAL ADVICE (provide structured, clear guidance):"""
        
        return prompt
    
    def _clean_response(self, response: str, prompt: str) -> str:
        """Clean and format the model response"""
        # Remove the original prompt
        response = response.replace(prompt, "").strip()
        
        # Handle common response patterns
        if "FINANCIAL ADVICE" in response:
            response = response.split("FINANCIAL ADVICE")[-1].strip()
            if response.startswith("(") or response.startswith(":"):
                response = response[1:].strip()
        
        # Remove any remaining prompt artifacts
        response = re.sub(r'^[:\(\)\-\s]+', '', response)
        
        # Add standard disclaimer if not present
        if "not personalized financial advice" not in response.lower() and "disclaimer" not in response.lower():
            disclaimer = "\n\nDISCLAIMER: This is general information only and not personalized financial advice. Please consult with a qualified financial advisor before making investment decisions."
            response += disclaimer
        
        return response
    
    def get_financial_advice(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Provide contextual financial advice based on user query.
        
        Args:
            query: User's financial question
            context: Dictionary with 'web_context' and 'memory_context'
        
        Returns:
            Financial advice string
        """
        try:
            # Process context into formatted string
            context_str = ""
            if context:
                context_str = self._process_context(context)
            
            # Create JSE-specific prompt
            prompt = self._create_jse_financial_prompt(query, context_str)
            
            # Tokenize with proper length management
            inputs = self._safe_tokenize(prompt, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=800,        # Reasonable limit for advice
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    top_p=0.9,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_response(response, prompt)
            
        except Exception as e:
            print(f"[AdvisorAgent] Error generating financial advice: {e}")
            return f"I apologize, but I encountered an error while processing your query: {query}. Please try rephrasing your question or contact support."
    
    def explain_concept(self, concept: str, user_level: str = "beginner", 
                       jse_context: bool = True) -> str:
        """
        Explain financial concepts at appropriate level with JSE context.
        
        Args:
            concept: Financial concept to explain
            user_level: User expertise level (beginner, intermediate, advanced)
            jse_context: Whether to include JSE-specific context
        
        Returns:
            Explanation string
        """
        try:
            jse_note = ""
            if jse_context:
                jse_note = "Provide examples relevant to the JSE and South African market context where applicable."
            
            prompt = f"""You are a financial education expert. Explain the following concept clearly for a {user_level}-level audience.

GUIDELINES:
- Use simple, clear language appropriate for {user_level} level
- Provide practical examples and real-world applications
- {jse_note}
- Include any risks or important considerations

CONCEPT TO EXPLAIN: {concept}

EXPLANATION:"""
            
            inputs = self._safe_tokenize(prompt, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=600,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    top_p=0.9,
                    top_k=50
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_response(response, prompt)
            
        except Exception as e:
            print(f"[AdvisorAgent] Error explaining concept: {e}")
            return f"I apologize, but I encountered an error while explaining the concept '{concept}'. Please try again."
    
    def _safe_tokenize(self, text: str, max_length: int = 1024):
        """Safely tokenize text with proper error handling"""
        try:
            return self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.model.device)
        except Exception as e:
            print(f"[AdvisorAgent] Tokenization error: {e}")
            # Fallback with shorter text
            shortened_text = text[:max_length * 3]  # Rough character-to-token ratio
            return self.tokenizer(
                shortened_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.model.device)
    
    def analyze_portfolio_diversification(self, portfolio_info: Dict[str, Any], 
                                        context: Optional[Dict[str, Any]] = None) -> str:
        """
        Specialized method for portfolio diversification analysis.
        
        Args:
            portfolio_info: Dictionary with portfolio details (sectors, holdings, etc.)
            context: Additional context from web/memory
        
        Returns:
            Diversification analysis and recommendations
        """
        try:
            portfolio_summary = self._summarize_portfolio(portfolio_info)
            context_str = ""
            if context:
                context_str = self._process_context(context)
            
            query = f"Analyze this portfolio diversification and provide recommendations: {portfolio_summary}"
            return self.get_financial_advice(query, context)
            
        except Exception as e:
            print(f"[AdvisorAgent] Error in portfolio analysis: {e}")
            return "I apologize, but I encountered an error while analyzing your portfolio. Please provide your portfolio details in a clear format."
    
    def _summarize_portfolio(self, portfolio_info: Dict[str, Any]) -> str:
        """Convert portfolio dictionary to readable summary"""
        if not portfolio_info:
            return "No portfolio information provided."
        
        summary_parts = []
        
        if 'holdings' in portfolio_info:
            summary_parts.append(f"Holdings: {portfolio_info['holdings']}")
        
        if 'sectors' in portfolio_info:
            summary_parts.append(f"Sectors: {portfolio_info['sectors']}")
        
        if 'total_value' in portfolio_info:
            summary_parts.append(f"Total Value: {portfolio_info['total_value']}")
        
        return "; ".join(summary_parts) if summary_parts else str(portfolio_info)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else "Unknown",
                "model_parameters": f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M" if self.model else "Unknown"
            }
        except Exception as e:
            return {"model_name": self.model_name, "error": str(e)}