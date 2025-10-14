# Model/advisor_model.py - FIXED VERSION

import os
import warnings
import json
import datetime
from typing import List, Optional, Tuple, Dict, Any
import torch
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

class AdvisorAgent:
    class PromptBuilder:
        def __init__(self):
            self.parts = []

        def add_persona(self):
            self.parts.append("You are a professional JSE financial analyst with expertise in South African markets.")
            return self

        def add_conversation_history(self, history):
            if history:
                history_text = "**RECENT CONVERSATION:**\n"
                for i, (prev_query, prev_response) in enumerate(history[-3:]):  # Last 3 exchanges
                    history_text += f"User: {prev_query}\nAnalyst: {prev_response}\n\n"
                self.parts.append(history_text)
            return self

        def add_context(self, context_str):
            if context_str and context_str.strip():
                self.parts.append(f"**MARKET CONTEXT:**\n{context_str}")
            return self

        def add_analytical_framework(self):
            self.parts.append(
                "**ANALYTICAL FRAMEWORK:**\n"
                "1. Provide comprehensive JSE stock analysis\n"
                "2. Use current market data and news\n"
                "3. Compare stocks when multiple are mentioned\n"
                "4. Include sector insights\n"
                "5. Provide forward-looking perspective\n"
                "6. Focus on JSE-specific factors\n"
                "7. Keep response professional but accessible"
            )
            return self

        def add_user_query(self, query):
            self.parts.append(f'**QUERY:** "{query}"')
            return self

        def add_report_header(self):
            self.parts.append("**JSE FINANCIAL ANALYSIS REPORT**")
            return self

        def build(self):
            return "\n\n".join(self.parts)

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Try 8-bit loading first
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            print("[AdvisorAgent] Model loaded in 8-bit mode")
        except Exception as e:
            print(f"[AdvisorAgent] 8-bit load failed: {e}, trying full precision...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                print("[AdvisorAgent] Model loaded in full precision")
            except Exception as e2:
                print(f"[AdvisorAgent] Critical error: {e2}")
                raise

        if self.tokenizer:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def _format_context(self, context):
        """Enhanced context formatting."""
        if not context:
            return "No market data available."

        parts = []
        
        # Market summary
        market_summary = context.get('market_summary')
        if market_summary:
            parts.append(f"📈 MARKET OVERVIEW:\n{market_summary}")

        # Market data
        market_data = context.get('market_data')
        if market_data and 'unavailable' not in market_data.lower():
            parts.append(f"📊 STOCK DATA:\n{market_data}")

        # News articles
        articles = context.get('web_context', [])
        if articles:
            parts.append("📰 RECENT NEWS:")
            for article in articles[:5]:  # Top 5 articles
                parts.append(f"• {article.get('headline', 'No headline')} (Source: {article.get('source', 'Unknown')})")

        # Memory context
        memory_context = context.get('memory_context', [])
        if memory_context:
            parts.append("💡 HISTORICAL CONTEXT:")
            for memory in memory_context[:2]:  # Top 2 memories
                parts.append(f"• {memory.get('text', 'No context')}")

        return "\n\n".join(parts) if parts else "Limited market data available."

    def get_financial_advice(self, query, context=None, history=None):
        """Enhanced advice generation."""
        try:
            context_str = self._format_context(context) if context else "No additional context."
            
            prompt = self.PromptBuilder() \
                .add_report_header() \
                .add_persona() \
                .add_conversation_history(history) \
                .add_context(context_str) \
                .add_analytical_framework() \
                .add_user_query(query) \
                .build()

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the new generated text
            prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
            generated_response = response[prompt_length:].strip()
            
            return generated_response if generated_response else "I apologize, but I couldn't generate a response. Please try rephrasing your question."

        except Exception as e:
            print(f"[AdvisorAgent] Error generating response: {e}")
            return f"I encountered an error while processing your request: {str(e)}. Please try again."

    def explain_concept(self, concept):
        """Enhanced concept explanation."""
        return self.get_financial_advice(f"Explain this financial concept: {concept}")