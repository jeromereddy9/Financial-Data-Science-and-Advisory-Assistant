import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class WebSupplementationAgent:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize_article(self, article_text, max_length=120):
        inputs = self.tokenizer(
            article_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
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

