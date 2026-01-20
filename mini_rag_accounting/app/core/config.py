from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "UIT MiniRAG Accounting"
    MONGO_URL: str
    DB_NAME: str = "uit_ai"
    OLLAMA_HOST: str = "http://localhost:11434"
    # OLLAMA_HOST: str = "http://mis_ollama:11434"

    # Model config - tách riêng để tối ưu tốc độ
    CLASSIFIER_MODEL: str = "qwen2.5:0.5b"  # Model nhỏ cho phân loại (nhanh)
    GENERATION_MODEL: str = "qwen2.5:3b"    # Model lớn hơn cho trả lời (chất lượng)

    class Config:
        env_file = ".env"

settings = Settings()