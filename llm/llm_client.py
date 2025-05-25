# llm_client.py: Unified interface for Perplexity (Sonar Pro) and OpenAI (GPT-4) LLMs
import os
import sys
import openai
from openai import OpenAI

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import AppConfig

class LLMClient:
    """
    Unified interface for Perplexity (Sonar Pro) and OpenAI (GPT-4) LLMs.
    Integrates with AppConfig for centralized configuration management.
    """
    def __init__(self, openai_api_key=None, perplexity_api_key=None, config=None):
        """
        Initialize LLM client with API keys.
        
        Args:
            openai_api_key: OpenAI API key (optional, will use config if not provided)
            perplexity_api_key: Perplexity API key (optional, will use config if not provided)
            config: AppConfig instance (optional, will create new if not provided)
        """
        # Use provided config or create new one
        if config is None:
            config = AppConfig()
        
        # Get LLM configuration
        llm_config = config.get_llm_config()
        
        self.openai_api_key = openai_api_key or llm_config.get('api_key')
        self.perplexity_api_key = perplexity_api_key or llm_config.get('perplexity_key')
        self.default_model = llm_config.get('model', 'gpt-4-turbo')
        
        # Validate API keys
        if not self.openai_api_key:
            print("Warning: OpenAI API key not found. OpenAI queries will fail.")
        if not self.perplexity_api_key:
            print("Warning: Perplexity API key not found. Perplexity queries will fail.")

    def query_openai_gpt4(self, prompt, model=None, **kwargs):
        """
        Query OpenAI GPT-4 model.
        
        Args:
            prompt: The prompt to send to the model
            model: Model name (defaults to configured model)
            **kwargs: Additional parameters for the API call
            
        Returns:
            str: The model's response content
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    def query_perplexity_sonarpro(self, prompt, model="sonar-pro", **kwargs):
        """
        Query Perplexity Sonar Pro model.
        
        Args:
            prompt: The prompt to send to the model
            model: Model name (defaults to sonar-pro)
            **kwargs: Additional parameters for the API call
            
        Returns:
            str: The model's response content
        """
        if not self.perplexity_api_key:
            raise ValueError("Perplexity API key not configured. Please set PERPLEXITY_API_KEY in your .env file.")
        
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
        
        Args:
            prompt: The prompt to send to the model
            provider: 'openai' or 'perplexity' (defaults to 'openai')
            model: model name (defaults to configured model for the provider)
            **kwargs: Additional parameters for the API call
            
        Returns:
            str: The model's response content
        """
        if provider == "openai":
            return self.query_openai_gpt4(prompt, model=model, **kwargs)
        elif provider == "perplexity":
            return self.query_perplexity_sonarpro(prompt, model=model or "sonar-pro", **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported providers: 'openai', 'perplexity'")

# Convenience function for backward compatibility
def create_llm_client(config=None):
    """
    Factory function to create an LLMClient with proper configuration.
    
    Args:
        config: AppConfig instance (optional)
        
    Returns:
        LLMClient: Configured LLM client instance
    """
    return LLMClient(config=config)
