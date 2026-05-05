import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from ml.preprocess import load_and_prepare

MODELS_DIR = Path("artifacts/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R²   = {r2:.4f}")
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "r2": round(float(r2), 4)}


def train():
    X_train, X_test, y_train, y_test = load_and_prepare()

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    all_metrics = {}

    for name, model in models.items():
        print(f"\nОбучение: {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds, name)
        all_metrics[name] = metrics

        if name == "xgboost":
            model.save_model(str(MODELS_DIR / "xgboost_model.json"))
        else:
            joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    best_model = max(all_metrics, key=lambda x: all_metrics[x]["r2"])
    print(f"\n✅ Лучшая модель: {best_model} (R² = {all_metrics[best_model]['r2']})")

    metadata = {
        "best_model": best_model,
        "version": "1.0.0",
        "feature_columns": [
            "store_id", "region_encoded", "store_type_encoded", "month",
            "weekday", "day_of_year", "is_weekend", "is_holiday",
            "promo_flag", "temperature", "price"
        ],
        "regions": ["Москва", "Санкт-Петербург", "Поволжье", "Урал", "Сибирь", "Юг"],
        "store_types": ["Супермаркет", "Минимаркет", "Гипермаркет", "АЗС"],
        "metrics": all_metrics,
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("\nВсе модели и metadata.json сохранены в artifacts/models/")


if __name__ == "__main__":
    train()