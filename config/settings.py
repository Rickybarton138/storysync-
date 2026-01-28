"""Application settings and configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==========================================================================
    # API Keys
    # ==========================================================================
    openai_api_key: str = Field(default="", description="OpenAI API key for Whisper")
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude")
    kling_api_key: str = Field(default="", description="Kling API key")
    minimax_api_key: str = Field(default="", description="Minimax API key")
    runway_api_key: str = Field(default="", description="Runway API key")
    fal_api_key: str = Field(default="", alias="FAL_KEY", description="fal.ai API key")

    # ==========================================================================
    # Video Generation
    # ==========================================================================
    default_video_model: Literal["kling", "minimax", "runway", "fal-kling", "fal-minimax"] = Field(
        default="fal-kling",
        description="Default video generation model",
    )
    default_video_quality: Literal["draft", "standard", "premium"] = Field(
        default="standard",
        description="Default video quality setting",
    )
    max_parallel_generations: int = Field(
        default=3,
        description="Maximum parallel video generations",
    )
    default_scene_duration: float = Field(
        default=5.0,
        description="Default scene duration in seconds",
    )
    min_scene_duration: float = Field(
        default=3.0,
        description="Minimum scene duration in seconds",
    )
    max_scene_duration: float = Field(
        default=15.0,
        description="Maximum scene duration in seconds",
    )

    # ==========================================================================
    # Storage (Phase 3)
    # ==========================================================================
    aws_access_key_id: str = Field(default="", description="AWS access key")
    aws_secret_access_key: str = Field(default="", description="AWS secret key")
    aws_region: str = Field(default="eu-west-2", description="AWS region")
    s3_bucket_name: str = Field(default="storysync-media", description="S3 bucket name")

    # ==========================================================================
    # Database (Phase 3)
    # ==========================================================================
    database_url: str = Field(
        default="postgresql://localhost:5432/storysync",
        description="Database connection URL",
    )

    # ==========================================================================
    # Redis (Phase 3)
    # ==========================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # ==========================================================================
    # Application
    # ==========================================================================
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    temp_dir: Path = Field(
        default=Path("/tmp/storysync"),
        description="Temporary file directory",
    )
    output_dir: Path = Field(
        default=Path("./output"),
        description="Output directory for generated videos",
    )

    # ==========================================================================
    # LLM Settings
    # ==========================================================================
    claude_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model for story analysis",
    )
    max_tokens_story_analysis: int = Field(
        default=4096,
        description="Max tokens for story analysis response",
    )
    max_tokens_prompt_generation: int = Field(
        default=1024,
        description="Max tokens for prompt generation response",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_video_api_key(self) -> bool:
        return bool(
            self.kling_api_key or self.minimax_api_key or self.runway_api_key or self.fal_api_key
        )

    @property
    def has_fal_key(self) -> bool:
        return bool(self.fal_api_key)

    def get_video_api_key(self, model: str | None = None) -> str:
        """Get API key for specified video model."""
        model = model or self.default_video_model
        keys = {
            "kling": self.kling_api_key,
            "minimax": self.minimax_api_key,
            "runway": self.runway_api_key,
            "fal-kling": self.fal_api_key,
            "fal-minimax": self.fal_api_key,
        }
        return keys.get(model, "")


# Global settings instance
settings = Settings()
