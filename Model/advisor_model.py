import os
import warnings

# Suppress TensorFlow and Hugging Face warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow info/warning messages
warnings.filterwarnings("ignore", message="Xet Storage is enabled")  # HF XET warning

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AdvisorAgent:
    def __init__(self):
        self.model_name = "tiiuae/falcon-rw-1b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto" if torch.cuda.is_available() else None
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_financial_advice(self, query, context=""):
        """Provide contextual financial advice based on user query"""
        prompt = f"""You are a helpful financial advisor. Provide clear, actionable advice.

Context: {context}

User Question: {query}

Financial Advice:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=1000,        # high limit to avoid truncation
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,  # stops naturally when model predicts EOS
                no_repeat_ngram_size=3,
                top_p=0.9,
                top_k=50
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        if "Financial Advice:" in response:
            return response.split("Financial Advice:")[-1].strip()
        return response

    def explain_concept(self, concept, user_level="beginner"):
        """Explain financial concepts at appropriate level"""
        prompt = f"""Explain the following financial concept for a {user_level} level audience:

Concept: {concept}

Explanation:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=1000,        # high limit to avoid truncation
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,  # stops naturally when model predicts EOS
                no_repeat_ngram_size=3,
                top_p=0.9,
                top_k=50
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        if "Explanation:" in response:
            return response.split("Explanation:")[-1].strip()
        return response
