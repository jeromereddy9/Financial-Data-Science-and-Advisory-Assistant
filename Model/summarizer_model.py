# summarizer_model.py - FIXED VERSION
import torch
from transformers import pipeline
from typing import Dict, Any

class SummarizerAgent:
    def __init__(self):
        try:
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                tokenizer="sshleifer/distilbart-cnn-12-6",
                framework="pt"
            )
            print("[SummarizerAgent] Successfully loaded model: sshleifer/distilbart-cnn-12-6")
        except Exception as e:
            print(f"[SummarizerAgent] Error loading model: {e}")
            self.summarizer = None

    def summarize_text(self, text: str, summary_type: str = 'brief') -> str:
        """Summarize text with proper error handling"""
        if not text or not text.strip():
            return "No text to summarize."
            
        if self.summarizer is None:
            # Fallback summarization
            sentences = text.split('.')
            if len(sentences) > 3:
                return '. '.join(sentences[:3]) + '.'
            return text[:200] + "..." if len(text) > 200 else text

        try:
            # Clean and prepare text
            clean_text = text.replace('\n', ' ').strip()
            if len(clean_text) < 50:
                return clean_text

            # Set max length based on summary type
            if summary_type == 'brief':
                max_length = 100
                min_length = 30
            else:  # detailed
                max_length = 200
                min_length = 50

            # Ensure text isn't too long for the model
            if len(clean_text) > 1024:
                clean_text = clean_text[:1024]

            summary = self.summarizer(
                clean_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            return summary[0]['summary_text'] if summary else clean_text[:150] + "..."
            
        except Exception as e:
            print(f"[SummarizerAgent] Summarization failed: {e}")
            # Simple fallback
            sentences = text.split('.')
            if len(sentences) > 2:
                return '. '.join(sentences[:2]) + '.'
            return text[:150] + "..." if len(text) > 150 else text