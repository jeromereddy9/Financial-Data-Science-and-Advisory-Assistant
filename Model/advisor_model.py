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
        """
        An inner class that uses the Builder pattern to construct complex prompts.
        This makes prompts modular, readable, and easy to extend.
        """
        def __init__(self):
            self.parts: List[str] = []

        def add_persona(self) -> 'AdvisorAgent.PromptBuilder':
            self.parts.append("You are a professional JSE financial analyst. Your primary directive is to provide analysis based *only* on the data provided.")
            return self

        def add_context(self, context_str: str) -> 'AdvisorAgent.PromptBuilder':
            if context_str and context_str.strip():
                self.parts.append(f"**CONTEXTUAL INFORMATION:**\n{context_str}")
            return self

        def add_grounding_rules(self) -> 'AdvisorAgent.PromptBuilder':
            """This is the 'advanced' part that builds on the base."""
            # **FIX**: Re-engineered rules for a more integrated and advisory response.
            self.parts.append(
                "**ANALYTICAL FRAMEWORK (Follow these rules strictly):**\n"
                "1.  **Data Grounding**: Every analytical statement you make **must** be directly supported by the information in the 'CONTEXTUAL INFORMATION' section.\n"
                "2.  **Acknowledge Gaps**: If historical data is missing, state it, but still provide an interpretation of the single-day data provided.\n"
                "3.  **Synthesize, Don't Invent**: Your role is to synthesize the provided data, not to add external knowledge.\n"
                "4.  **Cite the Source**: When referencing market data, you **must** mention the source provided in the context (e.g., 'According to Yahoo Finance...').\n"
                "5.  **Quantitative Comparison**: When comparing stocks, first state the values for each stock, then explicitly declare which is higher or lower before drawing a conclusion.\n"
                "6.  **Interpret and Conclude**: After presenting the data, explain what it indicates about recent performance and momentum. You **must** conclude your analysis with a single, concise advisory sentence based on this interpretation. For example: 'This stronger daily momentum for MTN may be of interest to short-term traders.'"
            )
            return self

        def add_strict_requirements(self) -> 'AdvisorAgent.PromptBuilder':
            # **FIX**: Added a critical rule to enforce a single paragraph and forbid headers.
            self.parts.append(
                "**STRICT REQUIREMENTS:**\n"
                "-   Directly address the user's question below.\n"
                "-   Maintain a professional, factual, and impartial tone.\n"
                "-   Do not repeat any headers or instructions from this prompt in your response.\n"
                "-   **CRITICAL**: The entire response must be a single, coherent paragraph. Do not use section headers like 'Summary' or 'Advisory Insight'."
            )
            return self
        
        def add_user_query(self, query: str) -> 'AdvisorAgent.PromptBuilder':
            self.parts.append(f'**USER QUESTION:** "{query}"')
            return self

        def add_report_header(self) -> 'AdvisorAgent.PromptBuilder':
            """
            Provides a header for the model to write under, rather than an
            instruction to copy. This prevents prompt leakage.
            """
            self.parts.append("**JSE Analyst's Report:**")
            return self

        def build(self) -> str:
            """Assembles the final prompt string."""
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

            print(f"[AdvisorAgent] Successfully loaded and optimized model: {self.model_name} on {self.device}")
        except Exception as e:
            print(f"[AdvisorAgent] Error loading model: {e}")
            raise

    def _load_model_with_fallback(self):
        """Loads the model with a fallback strategy."""
        attn_implementation = "sdpa"
        print(f"[AdvisorAgent] Using '{attn_implementation}' attention mechanism.")

        if self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[AdvisorAgent] Detected {vram_gb:.2f}GB of VRAM.")

            if vram_gb > 7:
                print(f"[AdvisorAgent] VRAM > 7GB. Loading full precision model.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    torch_dtype=self.dtype, 
                    trust_remote_code=True,
                    attn_implementation=attn_implementation
                ).to(self.device)
            else:
                print(f"[AdvisorAgent] VRAM < 7GB. Loading 4-bit quantized model.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    quantization_config=quantization_config, 
                    trust_remote_code=True, 
                    device_map="auto",
                    attn_implementation=attn_implementation
                )
        else:
            print("[AdvisorAgent] No CUDA detected. Loading model on CPU.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device)

    def _configure_tokenizer(self):
        """Configures the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_context(self, context: Dict[str, Any]) -> str:
        """Process and structure all context for the prompt."""
        if not context: return ""
        parts = []
        if context.get('market_data'):
            parts.append("REAL-TIME JSE DATA (Limited to latest trading day):")
            parts.append(context['market_data'])
        if context.get('web_context', {}).get('articles'):
            parts.append("\nRECENT MARKET NEWS:")
            for i, article in enumerate(context['web_context']['articles'][:3], 1):
                parts.append(f"{i}. {article.get('summary', 'N/A')} (Source: {article.get('source', 'N/A')})")
        if context.get('memory_context'):
            parts.append("\nRELEVANT PREVIOUS ADVICE:")
            for i, session in enumerate(context['memory_context'][:2], 1):
                parts.append(f"{i}. Regarding '{session.get('query', '')}', the advice was: {session.get('advisor_response', '')[:150]}...")
        return "\n".join(parts)
    
    def _safe_tokenize(self, text: str, max_length: int = 1536):
        """Safely tokenize text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}
    
    def get_financial_advice(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Provide contextual financial advice based on user query and pre-fetched context."""
        try:
            context_str = self._process_context(context)
            
            prompt = (
                self.PromptBuilder()
                .add_persona()
                .add_context(context_str)
                .add_grounding_rules()
                .add_strict_requirements()
                .add_user_query(query)
                .add_report_header()
                .build()
            )
            
            inputs = self._safe_tokenize(prompt)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=350,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response = full_response.split("**JSE Analyst's Report:**")[-1].strip()
            
            # **FIX**: Call the new, robust sanitizer for analytical responses.
            response = self._sanitize_analytical_response(response)

            disclaimer = "\n\nDISCLAIMER: This is general information only and not personalized financial advice. Consult a qualified financial advisor."
            if "disclaimer" not in response.lower():
                response += disclaimer
            
            return response or "Could not generate a response. Please rephrase your question."
            
        except Exception as e:
            print(f"[AdvisorAgent] Error generating financial advice: {e}")
            return f"An error occurred while processing your query. Details: {str(e)}"

    def explain_concept(self, concept: str, user_level: str = "beginner", 
                       jse_context: bool = True) -> str:
        """
        Explain financial concepts at appropriate level with JSE context.
        """
        try:
            jse_note = ""
            if jse_context:
                jse_note = "Provide examples relevant to the JSE (Johannesburg Stock Exchange) and the South African market."
            
            prompt = f"""You are a financial education expert. Your task is to explain a financial concept clearly and accurately.

**AUDIENCE**: A novice investor who is intelligent but new to financial topics.
**TONE**: Clear, professional, and educational. Use simple analogies where helpful.

**GUIDELINES**:
1.  Use the full name "Johannesburg Stock Exchange (JSE)" on the first mention, then the acronym "JSE".
2.  Provide one single, complete, and coherent answer. Do not repeat yourself or add conversational filler.

**CONCEPT TO EXPLAIN:** {concept}

**EXPLANATION:**"""
            
            inputs = self._safe_tokenize(prompt, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("**EXPLANATION:**")[-1].strip()
            
            response = self._sanitize_conceptual_explanation(response)
            
            return response if response else f"I apologize, but I encountered an error while explaining the concept '{concept}'. Please try again."
            
        except Exception as e:
            print(f"[AdvisorAgent] Error explaining concept: {e}")
            return f"I apologize, but I encountered an error while explaining the concept '{concept}'. Please try again."
    
    def _sanitize_conceptual_explanation(self, text: str) -> str:
        """
        A definitive "First Clean Answer" extractor for conceptual questions.
        """
        if not text:
            return ""

        text = re.sub(r'[^\x00-\x7F]+', '', text)

        paragraphs = text.split('\n\n')
        
        if not paragraphs:
            return ""
        first_para = paragraphs[0]

        final_text = first_para
        if len(paragraphs) > 1:
            second_para_start = paragraphs[1].strip().lower()
            continuation_markers = ["key takeaways:", "in summary:", "in conclusion:"]
            if any(second_para_start.startswith(marker) for marker in continuation_markers):
                final_text += "\n\n" + paragraphs[1]

        if not final_text.endswith(('.', '!', '?')):
            last_period = final_text.rfind('.')
            if last_period != -1:
                final_text = final_text[:last_period + 1]

        return final_text

    def _sanitize_analytical_response(self, text: str) -> str:
        """
        **REVISED**: A more aggressive sanitizer to enforce a single paragraph by
        removing common section headers and other noise.
        """
        if not text:
            return ""

        # **FIX**: Added more stop phrases to catch model violations of the single-paragraph rule.
        stop_phrases = [
            "\n\nbased on the provided data", 
            "end of analysis",
            "disclaimer:",
            "in summary,",
            "advisory insight:",
            "human:",
            "assistant:"
        ]

        # Find the earliest occurrence of any stop phrase.
        stop_index = len(text)
        for phrase in stop_phrases:
            found_index = text.lower().find(phrase.lower())
            if found_index != -1:
                stop_index = min(stop_index, found_index)

        # Take only the content before the first stop phrase.
        clean_text = text[:stop_index].strip()

        # Now, ensure the result ends with a complete sentence to avoid abrupt cutoffs.
        if clean_text and not clean_text.endswith(('.', '!', '?')):
            last_sentence_end = max(clean_text.rfind('.'), clean_text.rfind('!'), clean_text.rfind('?'))
            if last_sentence_end != -1:
                clean_text = clean_text[:last_sentence_end + 1]
        
        return clean_text

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