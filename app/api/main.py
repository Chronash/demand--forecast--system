import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "artifacts/models"
DATA_PATH = BASE_DIR / "artifacts/generated/sales_daily.csv"

app = FastAPI(
    title="Borjomi Demand Forecast API",
    version="3.0",
    description="API для прогноза спроса на минеральную воду Borjomi",
)

model = None
metadata = None
region_encoder = None
store_type_encoder = None
sales_df = None


class PredictionRequest(BaseModel):
    store_id: int = Field(..., ge=1, le=400)
    region: str
    store_type: str
    month: int = Field(..., ge=1, le=12)
    weekday: int = Field(..., ge=0, le=6)
    day_of_year: int = Field(..., ge=1, le=366)
    is_weekend: bool
    is_holiday: bool
    promo_flag: bool
    temperature: float = Field(..., ge=-35, le=35)
    price: float = Field(..., ge=70, le=140)


def load_assets():
    global model, metadata, region_encoder, store_type_encoder, sales_df

    metadata_path = MODELS_DIR / "metadata.json"
    model_path = MODELS_DIR / "xgboost_model.json"
    region_encoder_path = MODELS_DIR / "label_encoder_region.pkl"
    store_type_encoder_path = MODELS_DIR / "label_encoder_store_type.pkl"

    if not metadata_path.exists():
        raise FileNotFoundError("Не найден artifacts/models/metadata.json")
    if not model_path.exists():
        raise FileNotFoundError("Не найден artifacts/models/xgboost_model.json")
    if not region_encoder_path.exists():
        raise FileNotFoundError("Не найден artifacts/models/label_encoder_region.pkl")
    if not store_type_encoder_path.exists():
        raise FileNotFoundError("Не найден artifacts/models/label_encoder_store_type.pkl")
    if not DATA_PATH.exists():
        raise FileNotFoundError("Не найден artifacts/generated/sales_daily.csv")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    region_encoder = joblib.load(region_encoder_path)
    store_type_encoder = joblib.load(store_type_encoder_path)

    model = XGBRegressor()
    model.load_model(str(model_path))

    sales_df = pd.read_csv(DATA_PATH)
    sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"])
    sales_df = sales_df.sort_values(["store_id", "sale_date"]).reset_index(drop=True)


@app.on_event("startup")
def startup_event():
    load_assets()


@app.get("/")
def root():
    return {
        "message": "Borjomi Demand Forecast API is running",
        "version": metadata["version"] if metadata else "unknown",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "metadata_loaded": metadata is not None,
        "dataset_loaded": sales_df is not None,
    }


@app.get("/metadata")
def get_metadata():
    if metadata is None:
        raise HTTPException(status_code=500, detail="Metadata не загружена")
    return metadata


def compute_calendar_features(month: int):
    ref_date = pd.Timestamp(year=2024, month=month, day=15)
    week_of_year = int(ref_date.isocalendar().week)
    quarter = int(ref_date.quarter)
    days_to_month_end = int(ref_date.days_in_month - ref_date.day)
    return week_of_year, quarter, days_to_month_end


def get_store_history_features(store_id: int):
    store_hist = sales_df[sales_df["store_id"] == store_id].sort_values("sale_date")

    if store_hist.empty:
        median_demand = float(sales_df["demand_units"].median())
        return {
            "lag_1": median_demand,
            "lag_7": median_demand,
            "lag_14": median_demand,
            "rolling_mean_7": median_demand,
            "rolling_mean_14": median_demand,
            "rolling_std_7": 0.0,
            "last_price": float(sales_df["price"].median()),
            "last_temperature": float(sales_df["temperature"].median()),
        }

    demand_series = store_hist["demand_units"].tolist()
    price_series = store_hist["price"].tolist()
    temp_series = store_hist["temperature"].tolist()

    def safe_lag(values, lag, default):
        return float(values[-lag]) if len(values) >= lag else float(default)

    def safe_mean(values, window, default):
        return float(np.mean(values[-window:])) if len(values) >= 1 else float(default)

    def safe_std(values, window):
        if len(values) <= 1:
            return 0.0
        chunk = values[-window:] if len(values) >= window else values
        return float(np.std(chunk))

    median_demand = float(sales_df["demand_units"].median())

    return {
        "lag_1": safe_lag(demand_series, 1, median_demand),
        "lag_7": safe_lag(demand_series, 7, median_demand),
        "lag_14": safe_lag(demand_series, 14, median_demand),
        "rolling_mean_7": safe_mean(demand_series, 7, median_demand),
        "rolling_mean_14": safe_mean(demand_series, 14, median_demand),
        "rolling_std_7": safe_std(demand_series, 7),
        "last_price": float(price_series[-1]) if len(price_series) else float(sales_df["price"].median()),
        "last_temperature": float(temp_series[-1]) if len(temp_series) else float(sales_df["temperature"].median()),
    }


def build_feature_row(payload: PredictionRequest):
    if payload.region not in metadata["regions"]:
        raise HTTPException(status_code=422, detail=f"Неизвестный регион: {payload.region}")

    if payload.store_type not in metadata["store_types"]:
        raise HTTPException(status_code=422, detail=f"Неизвестный тип магазина: {payload.store_type}")

    week_of_year, quarter, days_to_month_end = compute_calendar_features(payload.month)
    hist = get_store_history_features(payload.store_id)

    row = {
        "store_id": payload.store_id,
        "region": int(region_encoder.transform([payload.region])[0]),
        "store_type": int(store_type_encoder.transform([payload.store_type])[0]),
        "month": payload.month,
        "weekday": payload.weekday,
        "day_of_year": payload.day_of_year,
        "week_of_year": week_of_year,
        "quarter": quarter,
        "days_to_month_end": days_to_month_end,
        "is_weekend": int(payload.is_weekend),
        "is_holiday": int(payload.is_holiday),
        "promo_flag": int(payload.promo_flag),
        "temperature": float(payload.temperature),
        "temp_delta_vs_lag1": float(payload.temperature - hist["last_temperature"]),
        "price": float(payload.price),
        "price_delta_vs_lag1": float(payload.price - hist["last_price"]),
        "lag_1": hist["lag_1"],
        "lag_7": hist["lag_7"],
        "lag_14": hist["lag_14"],
        "rolling_mean_7": hist["rolling_mean_7"],
        "rolling_mean_14": hist["rolling_mean_14"],
        "rolling_std_7": hist["rolling_std_7"],
    }

    feature_columns = metadata["feature_columns"]
    missing = [col for col in feature_columns if col not in row]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось собрать все признаки для модели. Отсутствуют: {missing}",
        )

    X = pd.DataFrame([[row[col] for col in feature_columns]], columns=feature_columns)
    return X


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        X = build_feature_row(request)
        prediction = float(model.predict(X)[0])
        prediction = max(0.0, prediction)

        revenue = prediction * float(request.price)

        return {
            "predicted_demand": int(round(prediction)),
            "predicted_revenue": float(round(revenue, 2)),
            "model_name": metadata["best_model"],
            "model_version": metadata["version"],
            "used_features": metadata["feature_columns"],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")