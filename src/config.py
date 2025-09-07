from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresConfig(BaseSettings):
    USER: str = "postgres"
    PASSWORD: str = "password"
    DB: str = "query_bier"
    HOST: str = "localhost"
    PORT: int = 5432

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        env_prefix="POSTGRES_",
        env_file_encoding="utf-8",
    )


class AppConfig(BaseSettings):
    postgres_config: PostgresConfig


app_config = AppConfig()
