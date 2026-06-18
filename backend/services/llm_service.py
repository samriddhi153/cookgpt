import os
from typing import Optional

groq_client = None
hf_pipeline = None


class LLMService:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACE_API_KEY")

    # -------------------------
    # GROQ INIT (Lazy)
    # -------------------------
    def _init_groq(self):
        global groq_client

        if groq_client is not None:
            return groq_client

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")

        try:
            from groq import Groq
            groq_client = Groq(api_key=self.groq_api_key)
            return groq_client
        except Exception as e:
            raise RuntimeError(f"Groq init failed: {str(e)}")

    # -------------------------
    # HF INIT (Lazy) - LIGHT MODEL
    # -------------------------
    def _init_hf(self):
        global hf_pipeline

        if hf_pipeline is not None:
            return hf_pipeline

        try:
            from transformers import pipeline

            # ⚠️ Use SMALL model (no GPU needed)
            hf_pipeline = pipeline(
                "text-generation",
                model="distilgpt2"
            )

            return hf_pipeline

        except Exception as e:
            raise RuntimeError(f"HF init failed: {str(e)}")

    # -------------------------
    # GROQ GENERATE
    # -------------------------
    def _generate_groq(self, prompt: str) -> str:
        client = self._init_groq()

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )

        return response.choices[0].message.content

    # -------------------------
    # HF GENERATE (SAFE)
    # -------------------------
    def _generate_hf(self, prompt: str) -> str:
        pipe = self._init_hf()

        result = pipe(
            prompt,
            max_new_tokens=200,
            do_sample=True
        )

        return result[0]["generated_text"]

    # -------------------------
    # MAIN GENERATE
    # -------------------------
    def generate(self, prompt: str) -> str:
        """
        Try Groq → fallback HF → never crash backend
        """

        # ---- Try GROQ ----
        try:
            print("Using Groq...")
            return self._generate_groq(prompt)

        except Exception as e:
            print(f"Groq failed: {str(e)}")

        # ---- Try HF ----
        try:
            print("Falling back to HF...")
            return self._generate_hf(prompt)

        except Exception as e:
            print(f"HF failed: {str(e)}")

        # ---- FINAL SAFE RESPONSE ----
        return "⚠️ AI service is temporarily unavailable. Please try again."

    def generate_text(self, prompt: str) -> str:
        return self.generate(prompt)