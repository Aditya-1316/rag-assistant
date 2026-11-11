from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Config()