"""
Tests package for Heart Attack Prediction API

Этот пакет содержит тесты для FastAPI приложения.
"""

__version__ = "1.0.0"

# Импорты для удобства
from .test_api import test_health_check, test_predict_endpoint

__all__ = ['test_health_check', 'test_predict_endpoint']