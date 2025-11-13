"""
Configuration management for Video Processor application.
Supports environment variables, default values, and validation.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class PathConfig:
    """Path configuration"""
    vosk_model_path: Path = field(
        default_factory=lambda: Path(os.getenv("VOSK_MODEL_PATH", "./models/vosk-model-en-us-0.22-lgraph"))
    )
    output_directory: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIRECTORY", "./output_directory"))
    )
    temp_directory: Path = field(
        default_factory=lambda: Path(os.getenv("TEMP_DIRECTORY", "./streamlit"))
    )

    def __post_init__(self):
        """Create directories if they don't exist"""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.temp_directory.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMConfig:
    """Large Language Model configuration"""
    # Model provider: "huggingface", "openai", "anthropic", "ollama", "groq"
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "huggingface"))

    # Hugging Face settings
    hf_model_id: str = field(
        default_factory=lambda: os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    )
    hf_use_quantization: bool = field(
        default_factory=lambda: os.getenv("HF_USE_QUANTIZATION", "true").lower() == "true"
    )
    hf_load_in_4bit: bool = field(
        default_factory=lambda: os.getenv("HF_LOAD_IN_4BIT", "true").lower() == "true"
    )

    # OpenAI settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))
    openai_base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))

    # Anthropic settings
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    )

    # Ollama settings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))

    # Groq settings
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"))

    # General LLM settings
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "200"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("LLM_CHUNK_SIZE", "1000"))
    )


@dataclass
class UpscalingConfig:
    """Video upscaling configuration"""
    model_id: str = field(
        default_factory=lambda: os.getenv("UPSCALE_MODEL_ID", "ai-forever/Real-ESRGAN")
    )
    model_filename: str = field(
        default_factory=lambda: os.getenv("UPSCALE_MODEL_FILENAME", "RealESRGAN_x4.pth")
    )
    scale: int = field(
        default_factory=lambda: int(os.getenv("UPSCALE_SCALE", "4"))
    )
    max_duration: Optional[int] = field(
        default_factory=lambda: int(os.getenv("UPSCALE_MAX_DURATION", "15"))
        if os.getenv("UPSCALE_MAX_DURATION") else None
    )
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("UPSCALE_MAX_WORKERS", "4"))
    )


@dataclass
class TranslationConfig:
    """Translation model configuration"""
    model_id: str = field(
        default_factory=lambda: os.getenv("TRANSLATION_MODEL_ID", "facebook/hf-seamless-m4t-medium")
    )
    supported_languages: list = field(
        default_factory=lambda: os.getenv(
            "SUPPORTED_LANGUAGES",
            "eng,fra,deu,spa,ita,por,rus,zho,jpn,kor"
        ).split(",")
    )


@dataclass
class AppConfig:
    """Main application configuration"""
    # Enable/disable features
    enable_caching: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CACHING", "true").lower() == "true"
    )
    enable_logging: bool = field(
        default_factory=lambda: os.getenv("ENABLE_LOGGING", "true").lower() == "true"
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Performance settings
    use_gpu: bool = field(
        default_factory=lambda: os.getenv("USE_GPU", "true").lower() == "true"
    )

    # UI settings
    page_title: str = field(
        default_factory=lambda: os.getenv("PAGE_TITLE", "AI Video Processor")
    )
    page_icon: str = field(default_factory=lambda: os.getenv("PAGE_ICON", "🎬"))

    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    upscaling: UpscalingConfig = field(default_factory=UpscalingConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration settings.
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        # Validate LLM provider settings
        if self.llm.provider == "openai" and not self.llm.openai_api_key:
            errors.append("OpenAI API key is required when using OpenAI provider")

        if self.llm.provider == "anthropic" and not self.llm.anthropic_api_key:
            errors.append("Anthropic API key is required when using Anthropic provider")

        if self.llm.provider == "groq" and not self.llm.groq_api_key:
            errors.append("Groq API key is required when using Groq provider")

        # Validate paths for local models
        if self.llm.provider == "huggingface":
            # Will download from HF if not exists, so just a warning
            pass

        # Validate numeric ranges
        if self.llm.temperature < 0 or self.llm.temperature > 2:
            errors.append("LLM temperature must be between 0 and 2")

        if self.upscaling.scale not in [2, 4, 8]:
            errors.append("Upscaling scale must be 2, 4, or 8")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for logging/debugging)"""
        return {
            "app": {
                "enable_caching": self.enable_caching,
                "enable_logging": self.enable_logging,
                "log_level": self.log_level,
                "use_gpu": self.use_gpu,
                "page_title": self.page_title,
            },
            "paths": {
                "vosk_model_path": str(self.paths.vosk_model_path),
                "output_directory": str(self.paths.output_directory),
                "temp_directory": str(self.paths.temp_directory),
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self._get_active_model_name(),
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
            },
            "upscaling": {
                "model_id": self.upscaling.model_id,
                "scale": self.upscaling.scale,
                "max_duration": self.upscaling.max_duration,
            },
        }

    def _get_active_model_name(self) -> str:
        """Get the active model name based on provider"""
        provider_models = {
            "huggingface": self.llm.hf_model_id,
            "openai": self.llm.openai_model,
            "anthropic": self.llm.anthropic_model,
            "ollama": self.llm.ollama_model,
            "groq": self.groq_model,
        }
        return provider_models.get(self.llm.provider, "unknown")


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create global configuration instance"""
    global _config
    if _config is None:
        _config = AppConfig()
        is_valid, errors = _config.validate()
        if not is_valid:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    return _config


def reload_config() -> AppConfig:
    """Reload configuration from environment"""
    global _config
    load_dotenv(override=True)
    _config = None
    return get_config()
