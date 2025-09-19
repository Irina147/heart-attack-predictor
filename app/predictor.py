import pandas as pd
import numpy as np
import joblib
import logging
import os

class HeartAttackPredictor:
    def __init__(self, model_path: str = 'models/best_model.pkl'):
        self.model = None  # ← Это ваша RandomForest модель из Jupyter!
        self.threshold = None
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.load_model_and_threshold()
    
    def load_model_and_threshold(self):
        """Загрузка ML модели и порога из Jupyter"""
        try:
            # Загружаем обученную модель
            self.model = joblib.load(self.model_path)
            self.logger.info("✅ ML модель успешно загружена")
            self.logger.info(f"Тип модели: {type(self.model).__name__}")
            
            # Загружаем порог из нашего исследования
            threshold_path = 'models/best_threshold.pkl'
            if os.path.exists(threshold_path):
                self.threshold = joblib.load(threshold_path)
                self.logger.info(f"✅ Порог загружен: {self.threshold}")
            else:
                self.threshold = 0.38
                self.logger.warning("⚠️ Используется порог по умолчанию")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки: {e}")
            raise
    
    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Используем нашу ML модель для предсказаний"""
        # Здесь self.model - это наш RandomForestClassifier
        probabilities = self.model.predict_proba(input_data)[:, 1]
        binary_predictions = (probabilities >= self.threshold).astype(int)
        
        return pd.DataFrame({
            'id': range(1, len(input_data) + 1),
            'prediction': binary_predictions
        })
