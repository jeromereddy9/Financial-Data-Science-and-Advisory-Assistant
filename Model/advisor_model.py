# Model/advisor_model.py - FIXED VERSION

import os
import warnings
# Suppress TensorFlow and Hugging Face warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow info/warning messages
warnings.filterwarnings("ignore", message="Xet Storage is enabled")  # HF XET warning

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple, Optional, Any
import re

class AdvisorAgent:
    def __init__(self):
        self.model_name =  "Qwen/Qwen2.5-3B-Instruct"   # open alternative to Falcon-1B-Instruct
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # FIXED: Remove device_map to avoid offloading conflicts
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
                # Removed: device_map="auto" - this causes the offloading issue
            )
            
            # Manually move to device after loading
            self.model = self.model.to(self.device)
            
            # FIXED: Ensure pad token is properly set with attention mask compatibility
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer))
                    
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
                        advice = session.get('advisor_response', session.get('summary', ''))
                        if query and advice:
                            formatted_context.append(f"{i}. Previous Query: {query}")
                            formatted_context.append(f"   Previous Advice: {advice[:200]}...")
                    elif isinstance(session, str):
                        formatted_context.append(f"{i}. {session[:200]}...")
                formatted_context.append("")
        
        return "\n".join(formatted_context)
    
    def _create_jse_financial_prompt(self, query: str, context: str) -> str:
        """Create a JSE-specific financial advisory prompt with strict constraints"""
    
        prompt = f"""You are a professional JSE financial analyst. Provide ONLY factual, specific analysis.

        STRICT REQUIREMENTS:
        - Answer the specific question asked
        - Focus on JSE-listed companies and South African market conditions  
        - Provide concrete financial metrics and sector analysis
        - Keep response under 200 words
        - No marketing language or promotional content
        - No generic investment philosophy
        - Include specific JSE stock codes when relevant

        QUESTION: {query}

        ANALYSIS (factual, specific, under 200 words):"""
    
        return prompt
    
    def _safe_tokenize(self, text: str, max_length: int = 1024):
        """FIXED: Safely tokenize text with proper attention mask handling"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                add_special_tokens=True
            )
            
            # FIXED: Ensure attention mask is always present
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            print(f"[AdvisorAgent] Tokenization error: {e}")
            # Fallback with shorter text
            shortened_text = text[:max_length * 2]  # Rough character-to-token ratio
            inputs = self.tokenizer(
                shortened_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                add_special_tokens=True
            )
            
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
    
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
            
            # FIXED: Use improved tokenization
            inputs = self._safe_tokenize(prompt, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # FIXED: Explicitly pass attention mask
                    max_new_tokens=600,        # Reduced to avoid hanging
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,    # FIXED: Add repetition penalty
                    length_penalty=1.0         # FIXED: Add length penalty
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean response
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
            
            return response if response else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            print(f"[AdvisorAgent] Error generating financial advice: {e}")
            return f"I apologize, but I encountered an error while processing your financial query. Please try rephrasing your question or contact support. Error details: {str(e)}"
    
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
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=400,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # Clean up response
            if "EXPLANATION:" in response:
                response = response.split("EXPLANATION:")[-1].strip()
            
            return response if response else f"I apologize, but I encountered an error while explaining the concept '{concept}'. Please try again."
            
        except Exception as e:
            print(f"[AdvisorAgent] Error explaining concept: {e}")
            return f"I apologize, but I encountered an error while explaining the concept '{concept}'. Please try again."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "dtype": str(self.dtype),
                "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else "Unknown",
                "model_parameters": f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M" if self.model else "Unknown",
                "pad_token": self.tokenizer.pad_token if self.tokenizer else "None"
            }
        except Exception as e:
            return {"model_name": self.model_name, "error": str(e)}