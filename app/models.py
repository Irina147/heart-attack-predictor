from pydantic import BaseModel
from typing import Optional

# Модели для API схем (не путаем с ML моделью!)
class HealthCheck(BaseModel):
    """Модель для ответа проверки здоровья"""
    status: str
    message: str
    threshold: float
    threshold_source: str

class PredictionResponse(BaseModel):
    """Модель для ответа предсказания"""
    message: str
    output_file: str
    records_processed: int
    high_risk_count: int
    threshold_used: float
    threshold_source: str

class ThresholdInfo(BaseModel):
    """Модель для информации о пороге"""
    threshold: float
    threshold_source: str
    model_loaded: bool

