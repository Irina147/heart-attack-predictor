"""
Heart Attack Prediction API Package

Этот пакет содержит FastAPI приложение для предсказания риска сердечного приступа.
"""

__version__ = "1.0.0"
__author__ = "Irina"
__description__ = "ML API для предсказания риска сердечного приступа"

# Импорты для удобства
from .main import app
from .predictor import HeartAttackPredictor
from .models import HealthCheck, PredictionResponse

__all__ = ['app', 'HeartAttackPredictor', 'HealthCheck', 'PredictionResponse']