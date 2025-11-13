"""
Unified LLM provider interface supporting multiple AI model APIs.
Supports: Hugging Face, OpenAI, Anthropic, Ollama, Groq
"""
from typing import Optional, Dict, Any, Protocol
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is properly configured and available"""
        pass


class HuggingFaceProvider(LLMProvider):
    """Hugging Face Transformers provider for local models"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of model"""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            logger.info(f"Loading Hugging Face model: {self.config.llm.hf_model_id}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm.hf_model_id)

            if self.config.llm.hf_use_quantization and self.config.llm.hf_load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.llm.hf_model_id,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.llm.hf_model_id,
                    device_map="auto"
                )

            self._initialized = True
            logger.info("Hugging Face model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face model: {e}")
            raise

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Generate text using Hugging Face model"""
        self._initialize()

        import torch

        max_tokens = max_tokens or self.config.llm.max_tokens
        temperature = temperature or self.config.llm.temperature

        try:
            device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    num_beams=4 if temperature == 0 else 1,
                    early_stopping=True
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Hugging Face is available"""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, config):
        self.config = config
        self.client = None

    def _initialize(self):
        """Initialize OpenAI client"""
        if self.client is not None:
            return

        try:
            from openai import OpenAI

            if not self.config.llm.openai_api_key:
                raise ValueError("OpenAI API key not configured")

            kwargs = {"api_key": self.config.llm.openai_api_key}
            if self.config.llm.openai_base_url:
                kwargs["base_url"] = self.config.llm.openai_base_url

            self.client = OpenAI(**kwargs)
            logger.info("OpenAI client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Generate text using OpenAI API"""
        self._initialize()

        max_tokens = max_tokens or self.config.llm.max_tokens
        temperature = temperature or self.config.llm.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        try:
            import openai
            return bool(self.config.llm.openai_api_key)
        except ImportError:
            return False


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""

    def __init__(self, config):
        self.config = config
        self.client = None

    def _initialize(self):
        """Initialize Anthropic client"""
        if self.client is not None:
            return

        try:
            from anthropic import Anthropic

            if not self.config.llm.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")

            self.client = Anthropic(api_key=self.config.llm.anthropic_api_key)
            logger.info("Anthropic client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Generate text using Anthropic API"""
        self._initialize()

        max_tokens = max_tokens or self.config.llm.max_tokens
        temperature = temperature or self.config.llm.temperature

        try:
            response = self.client.messages.create(
                model=self.config.llm.anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        try:
            import anthropic
            return bool(self.config.llm.anthropic_api_key)
        except ImportError:
            return False


class OllamaProvider(LLMProvider):
    """Ollama local API provider"""

    def __init__(self, config):
        self.config = config

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Generate text using Ollama API"""
        import requests

        max_tokens = max_tokens or self.config.llm.max_tokens
        temperature = temperature or self.config.llm.temperature

        try:
            response = requests.post(
                f"{self.config.llm.ollama_base_url}/api/generate",
                json={
                    "model": self.config.llm.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=300
            )
            response.raise_for_status()

            return response.json()["response"]

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            response = requests.get(
                f"{self.config.llm.ollama_base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False


class GroqProvider(LLMProvider):
    """Groq API provider"""

    def __init__(self, config):
        self.config = config
        self.client = None

    def _initialize(self):
        """Initialize Groq client"""
        if self.client is not None:
            return

        try:
            from groq import Groq

            if not self.config.llm.groq_api_key:
                raise ValueError("Groq API key not configured")

            self.client = Groq(api_key=self.config.llm.groq_api_key)
            logger.info("Groq client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Generate text using Groq API"""
        self._initialize()

        max_tokens = max_tokens or self.config.llm.max_tokens
        temperature = temperature or self.config.llm.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Groq is available"""
        try:
            import groq
            return bool(self.config.llm.groq_api_key)
        except ImportError:
            return False


class LLMProviderFactory:
    """Factory for creating LLM providers"""

    _providers: Dict[str, type[LLMProvider]] = {
        "huggingface": HuggingFaceProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "groq": GroqProvider,
    }

    @classmethod
    def create(cls, config) -> LLMProvider:
        """
        Create LLM provider based on configuration.

        Args:
            config: Application configuration

        Returns:
            Initialized LLM provider

        Raises:
            ValueError: If provider is not supported or not configured
        """
        provider_name = config.llm.provider.lower()

        if provider_name not in cls._providers:
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_name]
        provider = provider_class(config)

        if not provider.is_available():
            raise ValueError(
                f"LLM provider '{provider_name}' is not available. "
                f"Please check configuration and dependencies."
            )

        return provider

    @classmethod
    def get_available_providers(cls, config) -> list[str]:
        """Get list of available (configured) providers"""
        available = []
        for name, provider_class in cls._providers.items():
            try:
                provider = provider_class(config)
                if provider.is_available():
                    available.append(name)
            except Exception:
                continue
        return available


# Global provider cache
_provider_cache: Optional[LLMProvider] = None


def get_llm_provider(config, force_reload: bool = False) -> LLMProvider:
    """
    Get or create LLM provider instance (with caching).

    Args:
        config: Application configuration
        force_reload: Force recreation of provider

    Returns:
        LLM provider instance
    """
    global _provider_cache

    if force_reload or _provider_cache is None:
        _provider_cache = LLMProviderFactory.create(config)

    return _provider_cache
