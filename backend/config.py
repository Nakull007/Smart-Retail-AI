from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = "Smart Retail AI"
    DATABASE_URL: str = "postgresql+psycopg://retail:retail_password@localhost:5432/smartretail"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SENTIMENT_MODEL: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    MAX_UPLOAD_MB: int = 10
    ARTIFACTS_DIR: Path = Path("artifacts")


settings = Settings()
