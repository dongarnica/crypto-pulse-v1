# llm_client.py: Unified interface for Perplexity (Sonar Pro) and OpenAI (GPT-4) LLMs
import os
import sys
import logging
import time
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
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing LLM client")
        
        # Use provided config or create new one
        if config is None:
            config = AppConfig()
            self.logger.debug("Created new AppConfig instance")
        else:
            self.logger.debug("Using provided AppConfig instance")
        
        # Get LLM configuration
        llm_config = config.get_llm_config()
        self.logger.debug(f"Retrieved LLM config: model={llm_config.get('model', 'gpt-4-turbo')}")
        
        self.openai_api_key = openai_api_key or llm_config.get('api_key')
        self.perplexity_api_key = perplexity_api_key or llm_config.get('perplexity_key')
        self.default_model = llm_config.get('model', 'gpt-4-turbo')
        
        # Validate API keys
        if not self.openai_api_key:
            self.logger.warning("OpenAI API key not found. OpenAI queries will fail.")
        else:
            self.logger.info("OpenAI API key configured successfully")
            
        if not self.perplexity_api_key:
            self.logger.warning("Perplexity API key not found. Perplexity queries will fail.")
        else:
            self.logger.info("Perplexity API key configured successfully")
            
        self.logger.info(f"LLM client initialized with default model: {self.default_model}")

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
            self.logger.error("OpenAI API key not configured")
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        
        model_name = model or self.default_model
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        self.logger.info(f"Querying OpenAI model '{model_name}' with prompt: {prompt_preview}")
        
        start_time = time.time()
        try:
            client = OpenAI(api_key=self.openai_api_key)
            self.logger.debug(f"Created OpenAI client, sending request with kwargs: {kwargs}")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            response_time = time.time() - start_time
            response_content = response.choices[0].message.content
            response_preview = response_content[:100] + "..." if len(response_content) > 100 else response_content
            
            self.logger.info(f"OpenAI query completed in {response_time:.2f}s, response: {response_preview}")
            self.logger.debug(f"Full response metadata: usage={getattr(response, 'usage', 'N/A')}")
            
            return response_content
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"OpenAI query failed after {response_time:.2f}s: {str(e)}")
            raise

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
            self.logger.error("Perplexity API key not configured")
            raise ValueError("Perplexity API key not configured. Please set PERPLEXITY_API_KEY in your .env file.")
        
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        self.logger.info(f"Querying Perplexity model '{model}' with prompt: {prompt_preview}")
        
        start_time = time.time()
        try:
            client = OpenAI(api_key=self.perplexity_api_key, base_url="https://api.perplexity.ai")
            self.logger.debug(f"Created Perplexity client, sending request with kwargs: {kwargs}")
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            response_time = time.time() - start_time
            response_content = response.choices[0].message.content
            response_preview = response_content[:100] + "..." if len(response_content) > 100 else response_content
            
            self.logger.info(f"Perplexity query completed in {response_time:.2f}s, response: {response_preview}")
            self.logger.debug(f"Full response metadata: usage={getattr(response, 'usage', 'N/A')}")
            
            return response_content
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Perplexity query failed after {response_time:.2f}s: {str(e)}")
            raise

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
        self.logger.info(f"Routing query to provider '{provider}' with model '{model}'")
        
        try:
            if provider == "openai":
                return self.query_openai_gpt4(prompt, model=model, **kwargs)
            elif provider == "perplexity":
                return self.query_perplexity_sonarpro(prompt, model=model or "sonar-pro", **kwargs)
            else:
                self.logger.error(f"Unknown provider: {provider}")
                raise ValueError(f"Unknown provider: {provider}. Supported providers: 'openai', 'perplexity'")
        except Exception as e:
            self.logger.error(f"Query routing failed for provider '{provider}': {str(e)}")
            raise

# Convenience function for backward compatibility
def create_llm_client(config=None):
    """
    Factory function to create an LLMClient with proper configuration.
    
    Args:
        config: AppConfig instance (optional)
        
    Returns:
        LLMClient: Configured LLM client instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating LLM client via factory function")
    return LLMClient(config=config)
