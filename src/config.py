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


class ApiConfig(BaseSettings):
    OPENROUTER_API_KEY: str = ""
    GEMINI_API_KEYS: list[str] = []

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        env_prefix="API_",
        env_file_encoding="utf-8",
    )


class AppConfig(BaseSettings):
    postgres_config: PostgresConfig = PostgresConfig()
    api_config: ApiConfig = ApiConfig()


app_config = AppConfig(postgres_config=PostgresConfig())
