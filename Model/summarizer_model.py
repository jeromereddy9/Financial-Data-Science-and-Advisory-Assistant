# Model/summarizer_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Tuple, Optional, Any
import re

class SummarizerAgent:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the Financial Summarizer Agent with DistilBART model.
        
        Args:
            model_name: Name of the summarization model to use
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            print(f"[SummarizerAgent] Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"[SummarizerAgent] Error loading model: {e}")
            raise
        
        # Financial advice structure templates
        self.advice_templates = {
            "portfolio": ["Investment Strategy", "Risk Assessment", "Recommendations", "Next Steps"],
            "stock_analysis": ["Stock Overview", "Key Factors", "Risk Considerations", "Investment Outlook"],
            "market_insight": ["Market Conditions", "Key Trends", "Opportunities", "Risks"],
            "diversification": ["Current Allocation", "Diversification Gaps", "Recommendations", "Implementation"],
            "general": ["Key Points", "Analysis", "Recommendations", "Important Notes"]
        }
    
    def create_insights(self, text: str, advice_type: str = "general", 
                       short_max_length: int = 120, detailed_max_length: int = 400) -> Tuple[str, str]:
        """
        Create both short summary and detailed insights for financial advice.
        
        Args:
            text: Input financial advice text
            advice_type: Type of advice (portfolio, stock_analysis, market_insight, etc.)
            short_max_length: Maximum length for short summary
            detailed_max_length: Maximum length for detailed insights
        
        Returns:
            Tuple of (short_summary, detailed_insights)
        """
        try:
            if not text or not text.strip():
                return "No content to summarize.", "No detailed analysis available."
            
            # Pre-process text for financial context
            processed_text = self._preprocess_financial_text(text)
            
            # Generate short summary
            short_summary = self._generate_summary(
                processed_text, 
                max_length=short_max_length,
                min_length=30,
                summary_type="concise"
            )
            
            # Generate detailed insights with structure
            detailed_insights = self._generate_structured_insights(
                processed_text, 
                advice_type,
                max_length=detailed_max_length
            )
            
            # Post-process summaries for financial appropriateness
            short_summary = self._postprocess_summary(short_summary, "short")
            detailed_insights = self._postprocess_summary(detailed_insights, "detailed")
            
            return short_summary, detailed_insights
            
        except Exception as e:
            print(f"[SummarizerAgent] Error creating insights: {e}")
            return self._fallback_summary(text, "short"), self._fallback_summary(text, "detailed")
    
    def create_executive_summary(self, advisor_response: str, data_insights: str = "", 
                               context_info: str = "") -> Dict[str, str]:
        """
        Create an executive summary combining advisor response, data insights, and context.
        
        Args:
            advisor_response: Main financial advice from the Advisor Agent
            data_insights: Insights from Data Agent analysis
            context_info: Context from web/memory sources
        
        Returns:
            Dictionary with structured summary components
        """
        try:
            # Combine all inputs
            combined_text = self._combine_inputs(advisor_response, data_insights, context_info)
            
            # Generate different summary components
            executive_summary = {
                "key_takeaways": self._generate_key_takeaways(combined_text),
                "investment_recommendation": self._extract_investment_recommendation(advisor_response),
                "risk_assessment": self._extract_risk_assessment(combined_text),
                "market_context": self._summarize_market_context(context_info),
                "action_items": self._extract_action_items(advisor_response),
                "disclaimer": self._add_jse_disclaimer()
            }
            
            return executive_summary
            
        except Exception as e:
            print(f"[SummarizerAgent] Error creating executive summary: {e}")
            return {
                "key_takeaways": "Error generating summary",
                "investment_recommendation": "Please review full advisor response",
                "risk_assessment": "Standard investment risks apply",
                "market_context": "Current market conditions considered",
                "action_items": "Consult with financial advisor",
                "disclaimer": self._add_jse_disclaimer()
            }
    
    def _preprocess_financial_text(self, text: str) -> str:
        """Preprocess text to highlight financial keywords and JSE context"""
        # Add context cues for the summarization model
        financial_keywords = [
            "JSE", "Johannesburg Stock Exchange", "portfolio", "diversification",
            "investment", "stocks", "shares", "bonds", "risk", "return",
            "ZAR", "rand", "volatility", "market", "sector", "allocation"
        ]
        
        # Prepare text with financial context
        context_prefix = "Financial Advisory Summary for JSE Investment: "
        return context_prefix + text
    
    def _generate_summary(self, text: str, max_length: int, min_length: int, 
                         summary_type: str = "general") -> str:
        """Generate summary using the model"""
        try:
            # Adjust prompt based on summary type
            if summary_type == "concise":
                prompt_prefix = "Summarize the key financial advice points: "
            else:
                prompt_prefix = "Provide detailed financial analysis summary: "
            
            full_text = prompt_prefix + text
            
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True,
                    do_sample=False
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Remove prompt prefix from output
            summary = summary.replace(prompt_prefix, "").strip()
            
            return summary if summary else self._fallback_summary(text, summary_type)
            
        except Exception as e:
            print(f"[SummarizerAgent] Error in model generation: {e}")
            return self._fallback_summary(text, summary_type)
    
    def _generate_structured_insights(self, text: str, advice_type: str, 
                                    max_length: int) -> str:
        """Generate structured insights based on advice type"""
        try:
            # Get structure template
            structure = self.advice_templates.get(advice_type, self.advice_templates["general"])
            
            # Create structured prompt
            prompt = f"Create structured financial insights with sections for {', '.join(structure)}: {text}"
            
            detailed_insights = self._generate_summary(
                prompt,
                max_length=max_length,
                min_length=60,
                summary_type="detailed"
            )
            
            # Add structure if model didn't include it
            if not any(section.lower() in detailed_insights.lower() for section in structure):
                detailed_insights = self._add_structure_to_insights(detailed_insights, structure)
            
            return detailed_insights
            
        except Exception as e:
            print(f"[SummarizerAgent] Error generating structured insights: {e}")
            return self._fallback_summary(text, "detailed")
    
    def _add_structure_to_insights(self, insights: str, structure: list) -> str:
        """Add structure to unstructured insights"""
        sentences = insights.split('. ')
        structured_output = []
        
        sentences_per_section = max(1, len(sentences) // len(structure))
        
        for i, section in enumerate(structure):
            start_idx = i * sentences_per_section
            end_idx = start_idx + sentences_per_section if i < len(structure) - 1 else len(sentences)
            section_content = '. '.join(sentences[start_idx:end_idx])
            
            if section_content:
                structured_output.append(f"**{section}**: {section_content}")
        
        return '\n\n'.join(structured_output)
    
    def _combine_inputs(self, advisor_response: str, data_insights: str, context_info: str) -> str:
        """Combine different input sources"""
        combined = []
        
        if advisor_response:
            combined.append(f"FINANCIAL ADVICE: {advisor_response}")
        
        if data_insights:
            combined.append(f"DATA ANALYSIS: {data_insights}")
        
        if context_info:
            combined.append(f"MARKET CONTEXT: {context_info}")
        
        return " | ".join(combined)
    
    def _generate_key_takeaways(self, text: str) -> str:
        """Extract key takeaways from combined text"""
        return self._generate_summary(
            f"Extract the most important takeaways from this financial advice: {text}",
            max_length=150,
            min_length=40,
            summary_type="concise"
        )
    
    def _extract_investment_recommendation(self, advisor_response: str) -> str:
        """Extract specific investment recommendations"""
        # Look for recommendation keywords
        recommendation_keywords = [
            "recommend", "suggest", "consider", "should", "invest", 
            "buy", "sell", "hold", "allocate", "diversify"
        ]
        
        sentences = advisor_response.split('.')
        recommendations = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in recommendation_keywords):
                recommendations.append(sentence.strip())
        
        if recommendations:
            return '. '.join(recommendations[:3])  # Top 3 recommendations
        else:
            return self._generate_summary(
                f"Extract investment recommendations from: {advisor_response}",
                max_length=120,
                min_length=30
            )
    
    def _extract_risk_assessment(self, text: str) -> str:
        """Extract risk-related information"""
        risk_keywords = ["risk", "volatile", "uncertainty", "caution", "careful", "consider"]
        
        sentences = text.split('.')
        risk_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in risk_keywords):
                risk_sentences.append(sentence.strip())
        
        if risk_sentences:
            return '. '.join(risk_sentences[:2])
        else:
            return "Standard investment risks apply. Past performance does not guarantee future results."
    
    def _summarize_market_context(self, context_info: str) -> str:
        """Summarize market context information"""
        if not context_info:
            return "Current market conditions considered in analysis."
        
        return self._generate_summary(
            f"Summarize current market conditions: {context_info}",
            max_length=100,
            min_length=20,
            summary_type="concise"
        )
    
    def _extract_action_items(self, advisor_response: str) -> str:
        """Extract actionable items from advisor response"""
        action_keywords = [
            "review", "monitor", "research", "consult", "implement", 
            "consider", "evaluate", "contact", "track", "rebalance"
        ]
        
        sentences = advisor_response.split('.')
        actions = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in action_keywords):
                actions.append(sentence.strip())
        
        if actions:
            return '. '.join(actions[:3])
        else:
            return "Review recommendations with a qualified financial advisor."
    
    def _add_jse_disclaimer(self) -> str:
        """Add appropriate JSE/South African disclaimer"""
        return ("This is general financial information only, not personalized investment advice. "
                "JSE investments carry risks including currency fluctuation. "
                "Consult with a qualified financial advisor before making investment decisions. "
                "Consider your risk tolerance and investment objectives.")
    
    def _postprocess_summary(self, summary: str, summary_type: str) -> str:
        """Post-process summary for financial appropriateness"""
        if not summary:
            return "Summary not available."
        
        # Ensure proper financial language
        summary = summary.strip()
        
        # Add period if missing
        if not summary.endswith('.'):
            summary += '.'
        
        # Ensure it doesn't sound like personalized advice
        overconfident_phrases = [
            "you should definitely", "guaranteed to", "will certainly",
            "absolutely must", "without doubt"
        ]
        
        for phrase in overconfident_phrases:
            summary = summary.replace(phrase, "consider")
        
        return summary
    
    def _fallback_summary(self, text: str, summary_type: str) -> str:
        """Provide fallback summary when model fails"""
        if summary_type == "short":
            # Extract first two sentences
            sentences = text.split('.')[:2]
            return '. '.join(sentences).strip() + '.' if sentences else "Investment analysis provided."
        
        else:  # detailed
            # Extract key sentences based on financial keywords
            financial_keywords = ["investment", "portfolio", "risk", "return", "JSE", "market", "recommend"]
            sentences = text.split('.')
            
            relevant_sentences = []
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in financial_keywords):
                    relevant_sentences.append(sentence.strip())
            
            if relevant_sentences:
                return '. '.join(relevant_sentences[:4]) + '.'
            else:
                return "Detailed financial analysis considers market conditions, risk factors, and investment objectives for JSE investments."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the summarizer model"""
        try:
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "supported_advice_types": list(self.advice_templates.keys()),
                "max_input_length": 1024,
                "specialization": "Financial advice summarization for JSE investments"
            }
        except Exception as e:
            return {"model_name": self.model_name, "error": str(e)}