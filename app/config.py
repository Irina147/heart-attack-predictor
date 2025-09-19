import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Настройки приложения"""
    
    # Настройки API
    API_TITLE: str = "Heart Attack Risk Prediction API"
    API_DESCRIPTION: str = "ML API для предсказания риска сердечного приступа"
    API_VERSION: str = "1.0.0"
    
    # Пути к моделям
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_model.pkl")
    THRESHOLD_PATH: str = os.getenv("THRESHOLD_PATH", "models/best_threshold.pkl")
    
    # Параметры предсказания
    DEFAULT_THRESHOLD: float = 0.38
    
    # Настройки сервера
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    class Config:
        env_file = ".env"

# Создаем экземпляр настроек
settings = Settings()