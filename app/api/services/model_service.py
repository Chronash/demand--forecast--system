import json
import joblib
import numpy as np
from pathlib import Path
from loguru import logger
from xgboost import XGBRegressor

MODELS_DIR = Path("artifacts/models")


class ModelService:
    def __init__(self):
        logger.info("Загрузка модели и артефактов...")

        self.region_encoder = joblib.load(MODELS_DIR / "label_encoder_region.pkl")
        self.store_type_encoder = joblib.load(MODELS_DIR / "label_encoder_store_type.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")

        self.model = XGBRegressor()
        self.model.load_model(str(MODELS_DIR / "xgboost_model.json"))

        with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(
            f"Модель загружена: {self.metadata['best_model']} "
            f"v{self.metadata['version']} | "
            f"R²={self.metadata['metrics']['xgboost']['r2']}"
        )

    def predict(self, payload) -> dict:
        try:
            region_encoded = self.region_encoder.transform([payload.region])[0]
            store_type_encoded = self.store_type_encoder.transform([payload.store_type])[0]
        except ValueError as e:
            logger.error(f"Ошибка энкодинга: {e}")
            raise ValueError(f"Неизвестное значение: {e}")

        row = np.array([[
            payload.store_id,
            region_encoded,
            store_type_encoded,
            payload.month,
            payload.weekday,
            payload.day_of_year,
            int(payload.is_weekend),
            int(payload.is_holiday),
            int(payload.promo_flag),
            payload.temperature,
            payload.price,
        ]])

        row_scaled = self.scaler.transform(row)
        raw_pred = float(self.model.predict(row_scaled)[0])
        predicted_demand = max(0, int(round(raw_pred)))
        predicted_revenue = round(predicted_demand * payload.price, 2)

        return {
            "predicted_demand": predicted_demand,
            "predicted_revenue": predicted_revenue,
            "model_name": self.metadata["best_model"],
            "model_version": self.metadata["version"],
        }