# summarizer_model.py - FIXED VERSION
import torch
from transformers import pipeline
from typing import Dict, Any

class SummarizerAgent:
    """
    Summarization agent that condenses long financial analysis texts into concise summaries.
    Uses DistilBART model for abstractive summarization (generates new sentences, not just extraction).
    Provides graceful fallback to sentence-truncation if model unavailable.
    """
    
    def __init__(self):
        """
        Initialize the summarization pipeline with DistilBART model.
        DistilBART is a distilled version of BART - smaller, faster, but still high quality.
        Model size: ~306MB, suitable for production deployment.
        """
        try:
            # Load pre-trained summarization model
            # sshleifer/distilbart-cnn-12-6 is fine-tuned on CNN/DailyMail dataset
            # Good for financial/news content summarization
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",  # 306MB distilled BART model
                tokenizer="sshleifer/distilbart-cnn-12-6",
                framework="pt"  # Use PyTorch framework
            )
            print("[SummarizerAgent] Successfully loaded model: sshleifer/distilbart-cnn-12-6")
        except Exception as e:
            # Graceful degradation: system continues without summarization
            print(f"[SummarizerAgent] Error loading model: {e}")
            self.summarizer = None
    
    def summarize_text(self, text: str, summary_type: str = 'brief') -> str:
        """
        Summarize text with proper error handling and multiple fallback strategies.
        Abstractive summarization: generates new sentences that capture key information.
        
        Args:
            text: Input text to summarize (e.g., advisor report + data insights + news)
            summary_type: 'brief' (30-100 words) or 'detailed' (50-200 words)
                         Brief = quick overview for busy users
                         Detailed = comprehensive summary with more context
        
        Returns:
            Summarized text string
            - Model summary if available (abstractive, high quality)
            - Sentence-truncation fallback if model fails (extractive, lower quality)
            - Original text truncated if very short
        """
        # Input validation: empty or whitespace-only text
        if not text or not text.strip():
            return "No text to summarize."
            
        # Fallback strategy 1: Model not loaded
        if self.summarizer is None:
            # Simple extractive summarization: take first N sentences
            # Not ideal but better than nothing
            sentences = text.split('.')
            if len(sentences) > 3:
                return '. '.join(sentences[:3]) + '.'
            # If text already short, just truncate to 200 chars
            return text[:200] + "..." if len(text) > 200 else text
        
        try:
            # Clean and prepare text for model
            # Remove newlines (model handles spaces better) and strip whitespace
            clean_text = text.replace('\n', ' ').strip()
            
            # Skip summarization if text already very short (< 50 chars)
            # Summarizing short text often produces worse output than original
            if len(clean_text) < 50:
                return clean_text
            
            # Set length parameters based on summary type
            if summary_type == 'brief':
                max_length = 100  # Maximum 100 tokens (~75 words) for quick overview
                min_length = 30   # Minimum 30 tokens (~20 words) to ensure substance
            else:  # detailed
                max_length = 200  # Maximum 200 tokens (~150 words) for comprehensive summary
                min_length = 50   # Minimum 50 tokens (~35 words) for adequate detail
            
            # Truncate text if exceeds model's context limit
            # DistilBART has 1024 token limit (~750-850 words depending on vocabulary)
            if len(clean_text) > 1024:
                clean_text = clean_text[:1024]
            
            # Generate summary using the model
            # do_sample=False ensures deterministic output (same input = same output)
            summary = self.summarizer(
                clean_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False  # Deterministic generation for consistency
            )
            
            # Extract summary text from pipeline output
            # Pipeline returns list of dicts: [{'summary_text': '...'}]
            return summary[0]['summary_text'] if summary else clean_text[:150] + "..."
            
        except Exception as e:
            # Fallback strategy 2: Model loaded but summarization failed
            # Possible reasons: OOM, invalid input, tokenization error
            print(f"[SummarizerAgent] Summarization failed: {e}")
            
            # Simple extractive fallback: take first 2 sentences
            sentences = text.split('.')
            if len(sentences) > 2:
                return '. '.join(sentences[:2]) + '.'
            
            # Final fallback: truncate to 150 characters
            return text[:150] + "..." if len(text) > 150 else text