# Model/advisor_model.py - FINAL ROBUST AND VERIFIED VERSION

import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Any, Optional, List
import re

# Suppress TensorFlow and Hugging Face warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", message="Xet Storage is enabled")

class AdvisorAgent:
    class PromptBuilder:
        def __init__(self):
            self.parts: List[str] = []

        def add_persona(self) -> 'AdvisorAgent.PromptBuilder':
            self.parts.append("You are a professional JSE financial analyst. Your task is to provide a single, concise paragraph of analysis based *only* on the provided data.")
            return self

        def add_context(self, context_str: str) -> 'AdvisorAgent.PromptBuilder':
            if context_str and context_str.strip():
                self.parts.append(f"**CONTEXTUAL INFORMATION:**\n{context_str}")
            return self

        def add_analytical_framework(self) -> 'AdvisorAgent.PromptBuilder':
            # *** MINIMAL CHANGE 1: Added a critical rule to prevent price calculation ***
            self.parts.append(
                "**ANALYTICAL FRAMEWORK (Follow these rules strictly):**\n"
                "1.  **Synthesize, Don't List**: Combine the market data and news headlines into a single, flowing analysis.\n"
                "2.  **Use Exact Prices**: You MUST use the exact 'Price' value provided in the context. Do not perform any calculations to create new prices.\n"
                "3.  **Cite Sources Inline**: When you mention a piece of data, you **must** cite its source in parentheses, for example: '(Source: Yahoo Finance)' or '(Source: Moneyweb)'.\n"
                "4.  **Be Concise and Singular**: Your entire response must be a single, focused paragraph. Do not repeat information or use any headers.\n"
                "5.  **Conclude with Insight**: You MUST end your analysis with a concluding sentence that offers a forward-looking advisory insight based on the synthesized data."
            )
            return self
        
        def add_user_query(self, query: str) -> 'AdvisorAgent.PromptBuilder':
            self.parts.append(f'**USER QUESTION:** "{query}"')
            return self

        def add_report_header(self) -> 'AdvisorAgent.PromptBuilder':
            self.parts.append("**JSE Analyst's Report:**")
            return self

        def build(self) -> str:
            return "\n\n".join(self.parts)

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            self._load_model_with_fallback()
            self._configure_tokenizer()
            
            if self.device == "cuda":
                print("[AdvisorAgent] Applying torch.compile() for optimized inference.")
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

            print(f"[AdvisorAgent] Successfully loaded model: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"[AdvisorAgent] Error loading model: {e}")
            raise

    def _load_model_with_fallback(self):
        # This method is correct and requires no changes
        attn_implementation = "sdpa"
        if self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb > 7:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=self.dtype, trust_remote_code=True,
                    attn_implementation=attn_implementation
                ).to(self.device)
            else:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, quantization_config=quantization_config, trust_remote_code=True,
                    device_map="auto", attn_implementation=attn_implementation
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device)

    def _configure_tokenizer(self):
        # This method is correct and requires no changes
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_context(self, context: Dict[str, Any]) -> str:
        # This method is correct and requires no changes
        if not context: return ""
        # The web_context is nested inside the main context dictionary
        web_context = context.get('web_context', {})
        if not web_context: return ""
        parts = []
        if web_context.get('market_data'):
            parts.append(web_context['market_data'])
        articles = web_context.get('articles')
        if articles:
            for article in articles:
                parts.append(f"- [{article.get('ticker', 'N/A')}] '{article.get('headline', 'N/A')}' (Source: {article.get('source', 'N/A')})")
        return "\n".join(parts)
    
    def _safe_tokenize(self, text: str, max_length: int = 2048):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}
    
    def get_financial_advice(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        try:
            context_str = self._process_context(context)
            
            prompt = (
                self.PromptBuilder()
                .add_persona()
                .add_context(context_str)
                .add_analytical_framework()
                .add_user_query(query)
                .add_report_header()
                .build()
            )
            
            inputs = self._safe_tokenize(prompt)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.15
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response.split("**JSE Analyst's Report:**")[-1].strip()
            
            response = self._sanitize_analytical_response(response)

            disclaimer = "\n\n*This analysis is based on data from Yahoo Finance and various news outlets. It is not personalized financial advice.*"
            if "disclaimer" not in response.lower():
                response += disclaimer
            
            return response or "Could not generate a valid response. Please try again."
            
        except Exception as e:
            print(f"[AdvisorAgent] Error generating financial advice: {e}")
            return f"An error occurred while processing your query."

    def explain_concept(self, concept: str) -> str:
        # This method is correct and requires no changes
        try:
            prompt = f"""You are a financial educator. Explain the concept "{concept}" for a beginner investor in South Africa.
            - Start with a simple definition.
            - Provide a real-world JSE example if possible.
            - Keep it concise (150-250 words).
            **EXPLANATION:**"""
            
            inputs = self._safe_tokenize(prompt, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=300, temperature=0.5, do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("**EXPLANATION:**")[-1].strip()
        except Exception as e:
            print(f"[AdvisorAgent] Error in explain_concept: {e}")
            return f"I apologize, but an error occurred while explaining '{concept}'."

    def _sanitize_analytical_response(self, text: str) -> str:
        """
        *** MINIMAL CHANGE 2: A more aggressive sanitizer to remove repetition and garble. ***
        """
        if not text: return ""

        # 1. Define phrases that mark the start of the repetitive second paragraph or hallucinations and cut them off.
        stop_phrases = [
            "based on the given data and insights:",
            "the recent performance comparison between"
        ]
        stop_index = len(text)
        for phrase in stop_phrases:
            found_index = text.lower().find(phrase)
            if found_index != -1:
                stop_index = min(stop_index, found_index)
        
        # Take everything before the repetitive part.
        clean_text = text[:stop_index].strip()
        
        # 2. Ensure the result ends with a complete sentence.
        if clean_text and not clean_text.endswith(('.', '!', '?')):
            last_sentence_end = max(clean_text.rfind('.'), clean_text.rfind('!'), clean_text.rfind('?'))
            if last_sentence_end != -1:
                clean_text = clean_text[:last_sentence_end + 1]
        
        return clean_text

    def get_model_info(self) -> Dict[str, Any]:
        # This method is correct and requires no changes
        try:
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "model_parameters": f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M"
            }
        except Exception as e:
            return {"model_name": self.model_name, "error": str(e)}