# Model/advisor_model.py - COMPLETE CORRECTED VERSION

import os
import warnings
import json
import datetime
from typing import List, Optional, Tuple, Dict, Any
import torch
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

# Suppress TensorFlow and general warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

class AdvisorAgent:
    """
    Financial advisor agent that uses Qwen2.5-3B-Instruct LLM for JSE market analysis.
    Supports two modes: analytical (data-driven stock analysis) and conceptual (educational explanations).
    """
    
    class AnalyticalPromptBuilder:
        """
        Builder pattern for constructing structured prompts for analytical stock analysis.
        Ensures consistent, high-quality prompts with proper context and instructions.
        """
        
        def __init__(self):
            # List of prompt components that will be joined together
            self.parts = []

        def add_persona(self):
            """Set the AI's role and expertise domain."""
            self.parts.append(
                "You are a professional JSE financial analyst with deep expertise in South African markets."
            )
            return self

        def add_conversation_history(self, history):
            """
            Add recent conversation context for continuity.
            Only includes last 3 exchanges to avoid context overflow.
            """
            if history:
                history_text = "**RECENT CONVERSATION:**\n"
                for i, (prev_query, prev_response) in enumerate(history[-3:]):  # Last 3 exchanges
                    history_text += f"User: {prev_query}\nAnalyst: {prev_response}\n\n"
                self.parts.append(history_text)
            return self

        def add_context(self, context_str):
            """Add market data, news, and other contextual information."""
            if context_str and context_str.strip():
                self.parts.append(f"**MARKET CONTEXT:**\n{context_str}")
            return self

        def add_analytical_framework(self):
            """
            Sets the core analytical instructions for the model.
            Defines the structure and content requirements for stock analysis.
            """
            self.parts.append(
                "**ANALYTICAL FRAMEWORK:**\n"
                "1. Provide a complete JSE stock analysis using ONLY the CONTEXT data.\n"
                "2. Always refer to each stock by its full name and ticker (e.g., 'Standard Bank Limited (SBK.JO)').\n"
                "3. Describe each stock's 'Closing Price', 'Rand Change', and 'Percentage Change' explicitly.\n"
                "4. Mention the JSE market context (from MARKET OVERVIEW) before discussing individual stocks.\n"
                "5. Briefly mention the key headline for EACH stock from the RECENT NEWS section to provide color.\n"
                "6. Include all numerical figures EXACTLY as they appear in CONTEXT — copy them verbatim.\n"
                "7. Keep analysis in coherent sentences (no bullet points, no markdown bolding).\n"
                "8. If multiple stocks are present, use parallel comparative phrasing (e.g., 'MTN.JO rose by X whereas VOD.JO fell by Y'). If only one stock is present, provide a direct summary of its performance.\n"
                "9. Maintain a professional, factual tone; avoid repetition."
            )
            return self

        def add_internal_process(self):
            """
            Adds instructions for the model's internal thought process (Chain of Thought).
            This is not output to the user but guides the model's reasoning.
            """
            self.parts.append(
                "**INTERNAL PROCESS (do NOT output):**\n"
                "- Verify that every price, change, and percentage figure appears in CONTEXT exactly.\n"
                "- If any figure is missing, write 'Data unavailable' instead of estimating.\n"
                "- Ensure all stocks, market overview, and news headlines are covered."
            )
            return self

        def add_output_format(self):
            """
            Defines the exact output structure for REASONING and FINAL CONCLUSION.
            Critical for consistent parsing of model responses.
            """
            self.parts.append(
                "**OUTPUT FORMAT (must follow exactly):**\n"
                "REASONING:\n"
                "The reasoning section begins with a one-sentence market overview. It is followed by a separate paragraph for each stock, analyzing its price data and relevant news.\n\n"
                "FINAL CONCLUSION:\n"
                "The final conclusion is a well-developed advisory paragraph (2-3 sentences). It interprets the findings from the reasoning to provide a forward-looking takeaway. It must not restate any numbers (prices, percentages); it should focus on the strategic implications of the news and performance and explain what the conflict between the data and news means for an investor."
            )
            return self

        def add_strict_rules(self):
            """
            Adds a list of strict, non-negotiable rules for the model to follow.
            Prevents common LLM errors like incomplete responses or formatting issues.
            """
            self.parts.append(
                "**STRICT RULES:**\n"
                "- The final output MUST contain both the REASONING and the FINAL CONCLUSION sections. Do not stop after the reasoning.\n"
                "- Use no markdown or bold formatting in headings.\n"
                "- Output must include the plain text headings exactly as shown: 'REASONING:' and 'FINAL CONCLUSION:'.\n"
                "- Pay close attention to positive (+) and negative (-) signs and copy them exactly.\n"
                "- Do not add any other sections like 'INVESTMENT TAKEAWAY' or repeat sections.\n"
                "- Mention every stock symbol present in the CONTEXT.\n"
                "- Do not invent or infer any figure that is not explicitly given."
            )
            return self
        
        def add_conceptual_prompt(self):
            """Adds a simple instruction for explaining concepts (legacy method)."""
            self.parts.append(
                "**TASK:**\n"
                "Explain the following financial concept clearly and concisely, as a professional financial analyst would to a client."
            )
            return self

        def add_user_query(self, query):
            """Add the user's query at the end of the prompt."""
            self.parts.append(f'**QUERY:** "{query}"')
            return self

        def add_report_header(self):
            """Add a professional header to frame the analysis."""
            self.parts.append("**JSE FINANCIAL ANALYSIS REPORT**")
            return self

        def build(self):
            """Combine all parts into final prompt string."""
            return "\n\n".join(self.parts)

    class ConceptPromptBuilder:
        """
        Builder pattern for constructing prompts for conceptual/educational explanations.
        Focuses on clarity and accuracy without complex formatting or data analysis.
        """
        
        def __init__(self):
            self.parts = []

        def add_persona(self):
            """Set the AI's role as an educational financial explainer."""
            self.parts.append(
                "You are a financial educator specializing in clear, accurate explanations of financial concepts for JSE investors. You provide simple, correct explanations without complex formatting."
            )
            return self

        def add_conversation_history(self, history):
            """
            Add minimal conversation history for context.
            Only last 2 exchanges to keep focus on current concept.
            """
            if history:
                history_text = "**RECENT CONVERSATION:**\n"
                for i, (prev_query, prev_response) in enumerate(history[-2:]):  # Reduced to last 2
                    history_text += f"User: {prev_query}\nAnalyst: {prev_response}\n\n"
                self.parts.append(history_text)
            return self

        def add_explanation_framework(self):
            """
            Sets the framework for clear, accurate concept explanations.
            Emphasizes simplicity and correctness over complexity.
            """
            self.parts.append(
                "**EXPLANATION FRAMEWORK - FOLLOW EXACTLY:**\n"
                "1. Start with a simple one-sentence definition\n"
                "2. Provide a clear example with CORRECT numbers and calculations\n"
                "3. Explain why this matters to JSE investors\n"
                "4. Keep it to 3-4 paragraphs maximum\n"
                "5. Use only plain text - NO markdown, NO LaTeX, NO complex formatting\n"
                "6. Ensure all mathematical calculations are simple and accurate\n"
                "7. Write in complete sentences with proper punctuation\n"
                "8. Do not use bullet points, numbered lists, or special characters"
            )
            return self

        def add_strict_rules(self):
            """
            Adds strict rules for accurate explanations.
            Critical for preventing mathematical errors and formatting issues.
            """
            self.parts.append(
                "**CRITICAL RULES - MUST FOLLOW:**\n"
                "- ALL mathematical examples must be factually correct\n"
                "- Use only basic arithmetic that can be easily verified\n"
                "- If you provide numbers, keep them simple and realistic\n"
                "- NEVER use LaTeX, markdown, or special formatting\n"
                "- Write in continuous paragraphs without line breaks\n"
                "- Double-check that your calculations make sense\n"
                "- If unsure about accuracy, simplify the example\n"
                "- End your explanation with a complete sentence\n"
                "- Maximum 400 words total"
            )
            return self

        def add_user_query(self, query):
            """Add the concept query with clear instruction."""
            self.parts.append(f'**CONCEPT TO EXPLAIN:** "{query}"\n\n**YOUR EXPLANATION:**')
            return self

        def build(self):
            """Combine all parts into final prompt string."""
            return "\n\n".join(self.parts)

    def __init__(self):
        """
        Initialize the advisor agent with Qwen2.5-3B-Instruct model.
        Handles GPU/CPU detection and appropriate quantization.
        """
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            # Load model with fallback logic for different VRAM configurations
            self._load_model_with_fallback()
            print("[AdvisorAgent] Model loaded successfully.")
            
            # Load tokenizer for text encoding/decoding
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            print("[AdvisorAgent] Tokenizer loaded successfully.")

        except Exception as e:
            print(f"[AdvisorAgent] Critical error during model or tokenizer loading: {e}")
            raise

        # Configure tokenizer padding settings
        if self.tokenizer:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"  # Left padding for decoder-only models

    def _load_model_with_fallback(self):
        """
        Load model with appropriate configuration based on available VRAM.
        Uses three strategies: full precision (>7GB), 4-bit quantization (<7GB), or CPU.
        """
        attn_implementation = "sdpa"  # Scaled Dot Product Attention for efficiency
        
        if self.device == "cuda":
            # Check available VRAM to determine loading strategy
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if vram_gb > 7:
                # High VRAM: Load full model in FP16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=self.dtype, trust_remote_code=True,
                    attn_implementation=attn_implementation
                ).to(self.device)
            else:
                # Low VRAM: Use 4-bit quantization to reduce memory footprint
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, quantization_config=quantization_config, trust_remote_code=True,
                    device_map="auto", attn_implementation=attn_implementation
                )
        else:
            # CPU fallback: Load in FP32 for compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Formats context by parsing market data into a structured, readable format for the LLM.
        Extracts market overview, individual stock data, news articles, and historical context.
        """
        if not context:
            return "No market data available."

        parts = []
        market_data_str = context.get('market_data', '')
        # Split and clean lines for processing
        lines = [line.strip() for line in market_data_str.split('\n') if line.strip()]

        # Extract market-level summary (JSE All Share Index info)
        market_summary_line = next((line for line in lines if "JSE Market" in line), None)
        # Extract individual stock lines (contain .JO ticker suffix)
        stock_lines = [line for line in lines if ".JO" in line]

        # Add market overview section if available
        if market_summary_line:
            parts.append(f"📈 MARKET OVERVIEW:\n{market_summary_line.replace('🏛️', '').strip()}")

        # Parse and format individual stock data
        if stock_lines:
            parts.append("📊 DETAILED STOCK DATA:")
            for line in stock_lines:
                try:
                    # Regex to robustly parse the complex stock data string
                    # Format: "Company Name (TICKER): Price: X | Change: Y (Z%)"
                    match = re.search(r'(.+ \((.+)\)): Price: (.+) \| Change: (.+) \((.+)\)', line)
                    if match:
                        full_name, ticker, price, rand_change, pct_change = match.groups()
                        # Format as structured list for LLM clarity
                        parts.append(f"- Stock: {full_name.replace('📊', '').strip()}")
                        parts.append(f"  - Closing Price: {price.strip()}")
                        parts.append(f"  - Rand Change: {rand_change.strip()}")
                        parts.append(f"  - Percentage Change: {pct_change.strip()}")
                    else:
                        parts.append(f"- {line}") # Fallback for non-matching lines
                except Exception:
                    parts.append(f"- {line}") # Fallback if regex fails

        # Extract and organize news articles by stock ticker
        articles = context.get('web_context', [])
        if articles:
            parts.append("\n📰 RECENT NEWS:")
            news_by_stock = {}
            # Group news by ticker for better organization
            for article in articles[:10]:  # Limit to 10 most relevant articles
                ticker = article.get('ticker', 'UNKNOWN')
                news_by_stock.setdefault(ticker, []).append(article)

            # Format news by stock
            for ticker, ticker_articles in news_by_stock.items():
                parts.append(f"{ticker}:")
                for art in ticker_articles:
                    headline = art.get('headline', 'No headline')
                    source = art.get('source', 'Unknown')
                    parts.append(f"- {headline} (Source: {source})")

        # Add memory context from past sessions if available
        memory_context = context.get('memory_context', [])
        if memory_context:
            parts.append("\n💡 HISTORICAL CONTEXT:")
            for memory in memory_context[:2]:  # Limit to 2 most relevant memories
                parts.append(f"• {memory.get('text', 'No context')}")

        return "\n".join(parts) if parts else "Limited market data available."

    def get_financial_advice(self, query, context=None, history=None):
        """
        Enhanced advice generation with robust parsing to prevent repetition and format errors.
        Generates analytical stock analysis using market data and news context.
        """
        try:
            # Format all context data into structured text
            context_str = self._format_context(context) if context else "No additional context."

            # Build comprehensive analytical prompt
            prompt = self.AnalyticalPromptBuilder() \
                .add_report_header() \
                .add_persona() \
                .add_conversation_history(history) \
                .add_context(context_str) \
                .add_analytical_framework() \
                .add_internal_process() \
                .add_output_format() \
                .add_strict_rules() \
                .add_user_query(query) \
                .build()

            # Tokenize prompt with truncation for long contexts
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response with deterministic settings (temperature=0.0)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,  # Deterministic generation for consistency
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Slight penalty to reduce repetition
                )

            # Decode full response and extract only generated portion
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
            generated_response = response[prompt_length:].strip()

            # --- FINAL ROBUST PARSING LOGIC ---
            # Define section markers for parsing structured output
            reasoning_marker = "REASONING:"
            conclusion_marker = "FINAL CONCLUSION:"
            query_marker = "**QUERY:**"

            # Default values if parsing fails
            reasoning = "Analysis could not be reliably parsed."
            conclusion = "Conclusion could not be reliably parsed."

            # Clean any repeated query from the end of the entire response first
            if query_marker in generated_response:
                generated_response = generated_response.split(query_marker)[0].strip()

            # Parse response into reasoning and conclusion sections
            if conclusion_marker in generated_response:
                # Split by conclusion marker
                parts = generated_response.split(conclusion_marker, 1)
                reasoning_block = parts[0]
                conclusion_block = parts[1]

                # Extract reasoning (remove marker if present)
                if reasoning_marker in reasoning_block:
                    reasoning = reasoning_block.split(reasoning_marker, 1)[1].strip()
                else:
                    reasoning = reasoning_block.strip()
                
                # Extract first paragraph of conclusion (avoid repetition)
                conclusion = conclusion_block.strip().split('\n\n')[0].strip()

            elif reasoning_marker in generated_response:
                # Only reasoning found, no conclusion
                reasoning = generated_response.split(reasoning_marker, 1)[1].strip()
                conclusion = "No explicit final conclusion was found in the response."

            # Combine the parsed sections without the headings for the final output
            final_response = f"{reasoning}\n\n{conclusion}"
            return final_response

        except Exception as e:
            print(f"[AdvisorAgent] Error generating response: {e}")
            return f"I encountered an error while processing your request: {str(e)}. Please try again."

    def _clean_concept_response(self, response: str, concept: str) -> str:
        """
        Clean and validate concept explanation responses.
        Fixes formatting issues, removes markdown, and ensures proper sentence structure.
        """
        if not response:
            return f"I cannot provide an explanation for '{concept}' at the moment."
        
        # Fix line break issues in words (prevents "strat\negy" -> "strategy")
        response = re.sub(r'(\w)\s*\n\s*(\w)', r'\1\2', response)
        
        # Remove any markdown formatting (bold, italic, code)
        response = re.sub(r'[*_`#]', '', response)
        
        # Fix mathematical formatting issues
        response = re.sub(r'\\\[.*?\\\]', '', response)  # Remove LaTeX block math
        response = re.sub(r'\$.*?\$', '', response)      # Remove inline math symbols
        
        # Ensure the response ends with proper punctuation
        if response and not response.endswith(('.', '!', '?')):
            # Find the last complete sentence
            sentences = re.split(r'[.!?]', response)
            if len(sentences) > 1:
                # Rebuild with complete sentences only
                complete_sentences = [s.strip() for s in sentences[:-1] if s.strip()]
                if complete_sentences:
                    response = '. '.join(complete_sentences) + '.'
                else:
                    response = response.strip() + '.'
            else:
                response = response.strip() + '.'
        
        # Remove any duplicate whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Validate response length and quality (must have at least 10 words)
        if len(response.split()) < 10:
            return f"I don't have a detailed explanation for '{concept}' readily available. Please try rephrasing your question or ask about a different financial concept."
        
        return response

    def explain_concept(self, concept: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Generates an explanation for a financial concept using the ConceptPromptBuilder.
        Uses more creative generation settings (sampling, temperature) for natural explanations.
        """
        try:
            # Clean the concept query first (remove "what is" prefix if present)
            clean_concept = concept.strip()
            if clean_concept.lower().startswith('what is '):
                clean_concept = clean_concept[8:].strip()
            
            # Build conceptual explanation prompt
            prompt = self.ConceptPromptBuilder() \
                .add_persona() \
                .add_conversation_history(history) \
                .add_explanation_framework() \
                .add_strict_rules() \
                .add_user_query(f"Explain this financial concept clearly and correctly: {clean_concept}") \
                .build()

            # Tokenize with same truncation settings
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with sampling for more natural explanations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,  # More tokens for detailed explanations
                    do_sample=True,  # Enable sampling for variety
                    temperature=0.5,  # Moderate temperature for balance
                    top_p=0.85,  # Nucleus sampling cutoff
                    top_k=40,  # Top-k sampling for diversity
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,  # Higher penalty for conceptual explanations
                    no_repeat_ngram_size=2,  # Prevent 2-gram repetition
                    early_stopping=True,  # Stop at natural endpoints
                    num_return_sequences=1
                )

            # Decode and extract generated portion
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
            generated_response = response[prompt_length:].strip()

            # Clean up formatting issues and validate quality
            generated_response = self._clean_concept_response(generated_response, clean_concept)
            
            return generated_response

        except Exception as e:
            print(f"[AdvisorAgent] Error explaining concept: {e}")
            return f"I encountered an error while explaining that concept. Please try asking again more specifically."

    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None, 
                     history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Smart query router that detects whether it's an analytical or conceptual query.
        Routes to appropriate method based on keyword analysis.
        """
        # Keywords that indicate conceptual/educational questions
        concept_keywords = [
            'what is', 'explain', 'define', 'meaning of', 'how does', 'what are',
            'tell me about', 'describe', 'concept of', 'understanding'
        ]
        
        # Keywords that indicate analytical/data-driven questions  
        analytical_keywords = [
            'analyze', 'analysis', 'compare', 'performance', 'price', 'stock',
            'market', 'trend', 'forecast', 'predict', 'recommend', 'advice',
            'should i', 'buy', 'sell', 'hold', 'investment'
        ]
        
        query_lower = query.lower()
        
        # Check for both types of keywords
        is_conceptual = any(keyword in query_lower for keyword in concept_keywords)
        is_analytical = any(keyword in query_lower for keyword in analytical_keywords)
        
        # Route based on keyword detection priority
        # If clearly conceptual and not analytical, use concept explanation
        if is_conceptual and not is_analytical:
            print("[AdvisorAgent] Detected conceptual query, using explain_concept")
            return self.explain_concept(query, history)
        else:
            # Default to analytical for stock/market queries
            print("[AdvisorAgent] Detected analytical query, using get_financial_advice")
            return self.get_financial_advice(query, context, history)