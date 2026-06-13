from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # MongoDB Config
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "thought_compression"

    # JWT Config
    SECRET_KEY: str = "YOUR_SUPER_SECRET_KEY_NEVER_SHARE_THIS_1234567890"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # AI Config
    GEMINI_API_KEY: Optional[str] = None

    # App Settings
    PROJECT_NAME: str = "Thought Compression Engine API"
    VERSION: str = "0.1.0"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
# Trigger reload to load new GEMINI_API_KEY from .env

