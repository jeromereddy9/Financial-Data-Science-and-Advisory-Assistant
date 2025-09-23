from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch  

class SummarizerAgent:
    def __init__(self):
        self.model_name = "sshleifer/distilbart-cnn-12-6"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def _deduplicate_text(self, text):
        """
        Remove repeated sentences and extra whitespace
        """
        # Split into sentences (rough split by punctuation)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen = set()
        deduped = []
        for sentence in sentences:
            s_clean = sentence.strip()
            if s_clean and s_clean.lower() not in seen:
                deduped.append(s_clean)
                seen.add(s_clean.lower())
        return " ".join(deduped)

    def summarize_news(self, article_text, max_length=100):
        """
        Summarize financial news articles
        """
        article_text = self._deduplicate_text(article_text)
        inputs = self.tokenizer(
            article_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def create_insights(self, articles_list):
        """
        Create insights from multiple articles
        """
        insights = []
        for article in articles_list:
            article = self._deduplicate_text(article)
            summary = self.summarize_news(article, max_length=80)
            insights.append(summary)
        
        # Combine insights and deduplicate before final summarization
        combined_text = self._deduplicate_text(" ".join(insights))
        final_insight = self.summarize_news(combined_text, max_length=120)
        return final_insight, insights
    
    def extract_key_points(self, text):
        """
        Extract key financial points from text
        """
        text = self._deduplicate_text(text)
        financial_text = f"Financial summary: {text}"
        
        inputs = self.tokenizer(
            financial_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        import torch
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids,
                max_length=80,
                min_length=20,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        key_points = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return key_points
