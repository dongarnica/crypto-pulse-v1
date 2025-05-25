# llm_access.py: Unified interface for Perplexity (Sonar Pro) and OpenAI (GPT-4) LLMs
import os
import openai
from openai import OpenAI

class LLMClient:
    """
    Unified interface for Perplexity (Sonar Pro) and OpenAI (GPT-4) LLMs.
    """
    def __init__(self, openai_api_key=None, perplexity_api_key=None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.perplexity_api_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")

    def query_openai_gpt4(self, prompt, model="gpt-4-turbo", **kwargs):
        """
        Query OpenAI GPT-4 model.
        """
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    def query_perplexity_sonarpro(self, prompt, model="sonar-pro", **kwargs):
        """
        Query Perplexity Sonar Pro model.
        """
        client = OpenAI(api_key=self.perplexity_api_key, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    def query(self, prompt, provider="openai", model=None, **kwargs):
        """
        Query the selected LLM provider and model.
        provider: 'openai' or 'perplexity'
        model: model name (default: gpt-4-turbo for openai, sonar-pro for perplexity)
        """
        if provider == "openai":
            return self.query_openai_gpt4(prompt, model=model or "gpt-4-turbo", **kwargs)
        elif provider == "perplexity":
            return self.query_perplexity_sonarpro(prompt, model=model or "sonar-pro", **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
