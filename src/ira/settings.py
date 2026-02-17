from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    log_level: str = "INFO"
    service_name: str = "intelligent-research-assistant"

    # add to your Settings model
    IRA_DATA_DIR: str = "data"
    GITHUB_TOKEN: str | None = None



settings = Settings()
