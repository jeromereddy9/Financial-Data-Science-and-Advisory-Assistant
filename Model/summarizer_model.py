# Model/summarizer_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SummarizerAgent:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def create_insights(self, text, short_max_length=120, detailed_max_length=400):
        """
        Returns both a short summary and a more detailed insight of the input text.
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(self.device)

        # Generate short summary
        with torch.no_grad():
            short_ids = self.model.generate(
                inputs.input_ids,
                max_length=short_max_length,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            short_summary = self.tokenizer.decode(short_ids[0], skip_special_tokens=True)

        # Generate detailed insights
        with torch.no_grad():
            detailed_ids = self.model.generate(
                inputs.input_ids,
                max_length=detailed_max_length,
                min_length=60,
                num_beams=4,
                length_penalty=1.5,
                early_stopping=True
            )
            detailed_insights = self.tokenizer.decode(detailed_ids[0], skip_special_tokens=True)

        return short_summary, detailed_insights
