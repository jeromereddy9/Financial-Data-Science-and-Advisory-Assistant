# Model/advisor_model.py - COMPLETE FIXED VERSION WITH ANTI-HALLUCINATION

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", message="Xet Storage is enabled")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Optional, Any, List, Tuple
import re
from datetime import datetime


class AdvancedPromptEngine:
    """Advanced prompt engineering that builds on the base prompt"""
    
    def __init__(self):
        self.techniques = {
            "cot": self._apply_chain_of_thought,
            "critique": self._apply_self_critique, 
            "stepwise": self._apply_step_by_step,
            "factual": self._apply_factual_mode
        }
    
    def enhance_prompt(self, base_prompt: str, technique: str = "factual") -> str:
        """Apply advanced prompt engineering techniques to base prompt"""
        if technique in self.techniques:
            return self.techniques[technique](base_prompt)
        return base_prompt
    
    def _apply_chain_of_thought(self, base_prompt: str) -> str:
        """Chain-of-thought reasoning"""
        enhanced_prompt = f"""{base_prompt}

THINKING PROCESS (internal):
1. Extract key metrics from API data
2. Analyze current price trends
3. Assess sector performance
4. Formulate balanced analysis

FINAL ANALYSIS (external):"""
        
        return enhanced_prompt
    
    def _apply_self_critique(self, base_prompt: str) -> str:
        """Self-critique with API data validation"""
        enhanced_prompt = f"""{base_prompt}

QUALITY VERIFICATION:
- Use only verified API data points
- Report exact symbols and prices from data
- Acknowledge data limitations when present
- Ensure JSE-specific context

FINAL RESPONSE:"""
        
        return enhanced_prompt
    
    def _apply_step_by_step(self, base_prompt: str) -> str:
        """Step-by-step reasoning with API focus"""
        enhanced_prompt = f"""{base_prompt}

ANALYSIS STEPS:
1. Review API-provided stock data
2. Identify key metrics and trends
3. Provide JSE-focused insights
4. Note data freshness and limitations

ANALYSIS OUTPUT:"""
        
        return enhanced_prompt
    
    def _apply_factual_mode(self, base_prompt: str) -> str:
        """Factual mode that strictly uses API data"""
        enhanced_prompt = f"""{base_prompt}

DATA-DRIVEN REQUIREMENTS:
✓ Use only the exact data provided by API
✓ Report prices and changes as shown in data
✓ If data is unavailable, state this clearly
✓ Never invent symbols, prices, or metrics
✓ Focus on factual reporting over speculation

FACTUAL ANALYSIS:"""

        return enhanced_prompt


class AdvisorAgent:
    def __init__(self, web_agent=None):
        self.web_agent = web_agent
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Advanced prompt engine
        self.prompt_engine = AdvancedPromptEngine()
        self.prompt_technique = "factual"  # Best for API data

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            if self.device == "cuda":
                vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_total_gb >= 7:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    ).to("cuda")
                else:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to("cpu")

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            print(f"[AdvisorAgent] Loaded with advanced prompting")
        except Exception as e:
            print(f"[AdvisorAgent] Error loading model: {e}")
            raise

    def set_web_agent(self, web_agent):
        """Set web agent for API data"""
        self.web_agent = web_agent
        print("[AdvisorAgent] Web agent configured")

    def set_prompt_technique(self, technique: str):
        """Set the prompt technique to use"""
        if technique in self.prompt_engine.techniques:
            self.prompt_technique = technique
            print(f"[AdvisorAgent] Using prompt technique: {technique}")

    def _get_base_prompt(self) -> str:
        """Return the required base prompt structure"""
        return """You are a professional JSE financial analyst. Provide ONLY factual, specific analysis.

STRICT REQUIREMENTS:
- Answer the specific question asked
- Focus on JSE-listed companies and South African market conditions  
- Provide concrete financial metrics and sector analysis
- Keep response under 200 words
- No marketing language or promotional content
- No generic investment philosophy
- Include specific JSE stock codes when relevant

API DATA CONTEXT:
{api_context}

USER QUESTION:
{user_query}

PROFESSIONAL ANALYSIS:"""

    def _create_api_context_string(self, api_context: Dict) -> str:
        """FIXED - Create formatted API context that prevents hallucinations"""
        if not api_context or not api_context.get('stock_data'):
            current_date = api_context.get('current_date', 'today')
            return f"Market data is being retrieved for {current_date}. Analysis based on general JSE conditions."
        
        stock_data = api_context.get('stock_data', [])
        context_lines = []
        
        # Add timestamp FIRST to anchor the model in reality
        current_date = api_context.get('current_date', '')
        current_time = api_context.get('current_time', '')
        market_day = api_context.get('market_day', '')
        
        context_prefix = f"LIVE JSE DATA ({current_date} {current_time}, {market_day}): "
        
        for stock in stock_data[:3]:  # Limit to 3 stocks
            symbol = stock.get('symbol', '').upper()  # Force uppercase
            price = stock.get('current_price', 'N/A')
            daily_change = stock.get('daily_change', 'N/A')
            company = stock.get('company_name', '')
            sector = stock.get('sector', '')
            data_date = stock.get('data_date', '')
            
            # Skip stocks with errors
            if stock.get('error') or price == "N/A":
                continue
            
            # Format with EXACT symbol and data
            line_parts = [f"{symbol}"]
            if company and company != symbol:
                line_parts.append(f"({company})")
            line_parts.append(f"R{price}")
            
            if daily_change != "N/A":
                change_sign = "+" if float(daily_change) >= 0 else ""
                line_parts.append(f"({change_sign}{daily_change}%)")
            
            if sector:
                line_parts.append(f"[{sector}]")
            
            if data_date:
                line_parts.append(f"as of {data_date}")
            
            context_lines.append(" ".join(line_parts))
        
        if context_lines:
            # Add explicit instruction to prevent hallucination
            valid_symbols = api_context.get('symbols_used', [])
            symbols_instruction = f"\nVALID SYMBOLS ONLY: {', '.join(valid_symbols)}. DO NOT use other symbols."
            
            return context_prefix + " | ".join(context_lines) + symbols_instruction
        else:
            return f"JSE market data ({current_date}): Currently updating. Use general market analysis."

    def _create_enhanced_prompt(self, query: str, api_context: Dict) -> str:
        """Create complete prompt with base + advanced techniques"""
        base_prompt_template = self._get_base_prompt()
        api_context_str = self._create_api_context_string(api_context)
        
        # Fill in the base template
        base_prompt = base_prompt_template.format(
            api_context=api_context_str,
            user_query=query
        )
        
        # Apply advanced techniques
        enhanced_prompt = self.prompt_engine.enhance_prompt(base_prompt, self.prompt_technique)
        return enhanced_prompt

    def _clean_response(self, response: str, prompt: str) -> str:
        """Clean response while preserving actual content"""
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Remove any lines that look like prompt instructions or are in ALL CAPS
        response = re.sub(r'(STRICT REQUIREMENTS:|API DATA CONTEXT:|USER QUESTION:|PROFESSIONAL ANALYSIS:).*', '', response, flags=re.IGNORECASE)
        response = '\n'.join([line for line in response.split('\n') if not line.strip().isupper()])
        
        # Remove unrelated content (e.g., HR, contracts)
        response = re.sub(r'Human resources.*?contracts.*', '', response, flags=re.IGNORECASE)
        
        # Remove internal thinking sections but keep the analysis
        sections_to_remove = [
            r'THINKING PROCESS.*?FINAL ANALYSIS',
            r'QUALITY VERIFICATION.*?FINAL RESPONSE', 
            r'ANALYSIS STEPS.*?ANALYSIS OUTPUT',
            r'DATA-DRIVEN REQUIREMENTS.*?FACTUAL ANALYSIS',
            r'Thinking Process.*?Final Analysis',
            r'Quality Verification.*?Final Response'
        ]
        
        for pattern in sections_to_remove:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
        
        # Extract content after final markers
        final_markers = [
            'FINAL ANALYSIS:', 'FINAL RESPONSE:', 'ANALYSIS OUTPUT:', 
            'FACTUAL ANALYSIS:', 'PROFESSIONAL ANALYSIS:'
        ]
        
        for marker in final_markers:
            if marker in response:
                parts = response.split(marker, 1)
                if len(parts) > 1:
                    response = parts[1].strip()
                    break
        
        # Clean up formatting
        response = re.sub(r'\s+', ' ', response)
        response = response.strip()
        
        # Remove any remaining internal markers
        internal_terms = ['Internal:', 'Thinking Process', 'Quality Verification', 'Analysis Steps']
        for term in internal_terms:
            response = response.replace(term, '')
        
        return response if response and len(response) > 20 else "Based on current JSE market analysis."

    def _validate_response_strictly(self, text: str, api_context: Dict) -> str:
        """STRICT validation to prevent hallucinations"""
        valid_symbols = [s.upper() for s in api_context.get('symbols_used', [])]
        valid_prices = {}
        
        # Build valid price map from API data
        for stock in api_context.get('stock_data', []):
            symbol = stock.get('symbol', '').upper()
            price = stock.get('current_price')
            if symbol and price and price != "N/A":
                valid_prices[symbol] = price
        
        # Remove any mention of invalid symbols
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Check if it looks like a stock symbol (2-5 uppercase letters)
            if re.match(r'^[A-Z]{2,5}$', word):
                if word in valid_symbols:
                    cleaned_words.append(word)
                else:
                    # Replace invalid symbols with valid ones if context suggests
                    replacement = self._get_symbol_replacement(word, valid_symbols)
                    if replacement:
                        cleaned_words.append(replacement)
                    # Otherwise skip the invalid symbol
            else:
                cleaned_words.append(word)
        
        # Validate prices - remove any prices not matching our API data
        cleaned_text = " ".join(cleaned_words)
        
        # Remove price mentions that don't match our data
        for symbol, correct_price in valid_prices.items():
            # Find price patterns like "R123.45" or "ZAR 123.45"
            price_patterns = [
                rf'{symbol}[^0-9]*R?([0-9]+\.?[0-9]*)',
                rf'R([0-9]+\.?[0-9]*)[^0-9]*{symbol}',
            ]
            
            for pattern in price_patterns:
                matches = re.finditer(pattern, cleaned_text, re.IGNORECASE)
                for match in matches:
                    try:
                        mentioned_price = float(match.group(1))
                        correct_price_float = float(correct_price)
                        
                        # If mentioned price is very different from correct price, replace it
                        if abs(mentioned_price - correct_price_float) / correct_price_float > 0.1:  # 10% tolerance
                            old_text = match.group(0)
                            new_text = old_text.replace(match.group(1), str(correct_price))
                            cleaned_text = cleaned_text.replace(old_text, new_text)
                    except (ValueError, ZeroDivisionError):
                        continue
        
        return cleaned_text

    def _get_symbol_replacement(self, invalid_symbol: str, valid_symbols: List[str]) -> Optional[str]:
        """Get replacement for invalid symbol based on similarity"""
        replacements = {
            'MTM': 'MTN', 'MTG': 'MTN', 'MTB': 'MTN', 'MTX': 'MTN',
            'ABL': 'AGL', 'AGB': 'AGL', 'AGO': 'AGL', 'ABO': 'AGL',
            'SBG': 'SBK', 'FSB': 'FSR', 'NED': 'NED'
        }
        
        if invalid_symbol in replacements and replacements[invalid_symbol] in valid_symbols:
            return replacements[invalid_symbol]
        
        return None

    def get_financial_advice(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """FIXED - Generate financial advice with strict anti-hallucination measures"""
        try:
            if not self.web_agent:
                return "Financial data service unavailable. Please try again later."

            # Get API context from web agent
            api_context = self.web_agent.get_api_context(query)
            
            # CRITICAL: Verify we have valid data before proceeding
            if not api_context.get('stock_data'):
                print("[AdvisorAgent] No stock data, using general analysis mode")
                # Continue with general analysis instead of stopping
                api_context = {
                    "stock_data": [],
                    "current_date": datetime.now().strftime("%Y-%m-%d"),
                    "current_time": datetime.now().strftime("%H:%M"),
                    "market_day": datetime.now().strftime("%A"),
                    "data_status": "general"
                }
            
            # Create enhanced prompt with factual mode
            prompt = self._create_enhanced_prompt(query, api_context)
            
            # Generate with conservative parameters to reduce hallucination
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=200,  # REDUCED from 300 to limit hallucination
                    temperature=0.3,     # REDUCED from 0.7 to reduce creativity
                    do_sample=True,
                    top_p=0.8,          # ADD top_p for better control
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1
                )
            
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Multi-stage cleaning
            clean_response = self._clean_response(raw_response, prompt)
            clean_response = self._validate_response_strictly(clean_response, api_context)
            
            # Add data freshness note
            data_status = api_context.get('data_status', 'unknown')
            current_date = api_context.get('current_date', '')
            current_time = api_context.get('current_time', '')
            
            if current_date and current_time:
                clean_response += f"\n\nData as of: {current_date} {current_time}"
            
            # Add disclaimer
            clean_response += "\n\nDisclaimer: This is general market information. Consult a qualified financial advisor."

            return clean_response

        except Exception as e:
            print(f"[AdvisorAgent] Error: {e}")
            return "Error processing query. Financial data service may be temporarily unavailable."

    def explain_concept(self, concept: str, user_level: str = "beginner", 
                       jse_context: bool = True) -> str:
        """Explain financial concepts with JSE context"""
        try:
            jse_note = " with JSE examples" if jse_context else ""
            
            prompt = f"""You are a financial education expert. Explain the following concept clearly for a {user_level}-level audience.

Concept: {concept}
Audience: {user_level} level{jse_note}

Provide a clear, practical explanation:"""
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=250,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(prompt, "").strip()
            
        except Exception as e:
            return f"I apologize, but I couldn't explain the concept '{concept}' at this time. Please try again later."

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "prompt_technique": self.prompt_technique,
            "base_prompt": "Professional JSE financial analyst",
            "web_agent_integration": "Active" if self.web_agent else "Inactive",
            "features": [
                "Proper base prompt structure",
                "Advanced prompting techniques", 
                "API data integration",
                "JSE-focused analysis",
                "Symbol validation",
                "Anti-hallucination measures"
            ]
        }
