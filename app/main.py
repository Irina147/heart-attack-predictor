from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import logging
from .predictor import HeartAttackPredictor
from .models import PredictionResponse, HealthCheck, ThresholdInfo

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API для предсказания риска сердечного приступа",
    version="1.0.0"
)

# Инициализация predictor
predictor = HeartAttackPredictor()

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Проверка статуса сервиса"""
    threshold_info = predictor.get_threshold_info()
    return {
        "status": "healthy", 
        "message": "Heart Attack Prediction API is running",
        "threshold": threshold_info["threshold"],
        "threshold_source": threshold_info["threshold_source"]
    }

@app.get("/threshold")
async def get_threshold_info():
    """Получение информации о используемом пороге"""
    return predictor.get_threshold_info()

@app.post("/predict", response_model=PredictionResponse)
async def predict_heart_attack(file: UploadFile = File(...)):
    """
    Предсказание риска сердечного приступа для уже обработанных данных
    Возвращает CSV файл с колонками: id, prediction
    """
    try:
        # Проверка формата файла
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Поддерживаются только CSV файлы")
        
        # Чтение файла
        df = pd.read_csv(file.file)
        logger.info(f"Получен файл с формой: {df.shape}")
        
        # Предсказание с порогом
        predictions = predictor.predict(df)
        
        # Сохранение результатов в требуемом формате
        output_path = "predictions.csv"
        predictions.to_csv(output_path, index=False)
        
        # Статистика
        high_risk_count = (predictions['prediction'] == 1).sum()
        threshold_info = predictor.get_threshold_info()
        
        return PredictionResponse(
            message="Предсказание успешно завершено",
            output_file=output_path,
            records_processed=len(predictions),
            high_risk_count=high_risk_count,
            threshold_used=threshold_info["threshold"],
            threshold_source=threshold_info["threshold_source"]
        )
        
    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_predictions(filename: str):
    """Скачивание файла с предсказаниями"""
    import os
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return FileResponse(
        filename, 
        media_type='text/csv',
        filename=f"heart_attack_predictions.csv"
    )