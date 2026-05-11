from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_PATH = Path("artifacts/generated/sales_daily.csv")
MODELS_DIR = Path("artifacts/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values("sale_date").reset_index(drop=True)

    feature_columns = [
        "store_id",
        "region",
        "store_type",
        "month",
        "weekday",
        "day_of_year",
        "week_of_year",
        "quarter",
        "days_to_month_end",
        "is_weekend",
        "is_holiday",
        "promo_flag",
        "temperature",
        "temp_delta_vs_lag1",
        "price",
        "price_delta_vs_lag1",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
    ]
    target_column = "demand_units"

    region_encoder = LabelEncoder()
    store_type_encoder = LabelEncoder()

    df["region"] = region_encoder.fit_transform(df["region"])
    df["store_type"] = store_type_encoder.fit_transform(df["store_type"])

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=450,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics = {}
    trained_models = {}

    for model_name, model in models.items():
        if model_name == "linear_regression":
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        metrics[model_name] = evaluate(y_test, pred)
        trained_models[model_name] = model

    best_model_name = max(metrics, key=lambda k: metrics[k]["r2"])
    best_model = trained_models[best_model_name]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_rows = []
    cv_summary = {}

    for model_name, model in models.items():
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if model_name == "linear_regression":
                local_scaler = StandardScaler()
                X_tr_scaled = local_scaler.fit_transform(X_tr)
                X_val_scaled = local_scaler.transform(X_val)
                model.fit(X_tr_scaled, y_tr)
                pred = model.predict(X_val_scaled)
            else:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)

            score = evaluate(y_val, pred)
            fold_scores.append(score)
            cv_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "mae": score["mae"],
                    "rmse": score["rmse"],
                    "r2": score["r2"],
                }
            )

        cv_summary[model_name] = {
            "cv_avg_mae": float(np.mean([x["mae"] for x in fold_scores])),
            "cv_avg_rmse": float(np.mean([x["rmse"] for x in fold_scores])),
            "cv_avg_r2": float(np.mean([x["r2"] for x in fold_scores])),
        }

    pd.DataFrame(cv_rows).to_csv(MODELS_DIR / "cv_results.csv", index=False)

    joblib.dump(region_encoder, MODELS_DIR / "label_encoder_region.pkl")
    joblib.dump(store_type_encoder, MODELS_DIR / "label_encoder_store_type.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    joblib.dump(trained_models["linear_regression"], MODELS_DIR / "linear_regression.pkl")
    joblib.dump(trained_models["random_forest"], MODELS_DIR / "random_forest.pkl")
    trained_models["xgboost"].save_model(str(MODELS_DIR / "xgboost_model.json"))

    metadata = {
        "version": "3.0",
        "best_model": best_model_name,
        "feature_columns": feature_columns,
        "regions": region_encoder.classes_.tolist(),
        "store_types": store_type_encoder.classes_.tolist(),
        "metrics": metrics,
        "cv_results": cv_summary,
        "dataset": {
            "rows": int(len(df)),
            "price_min": float(df["price"].min()),
            "price_max": float(df["price"].max()),
            "temperature_min": float(df["temperature"].min()),
            "temperature_max": float(df["temperature"].max()),
            "date_min": str(df["sale_date"].min().date()),
            "date_max": str(df["sale_date"].max().date()),
        },
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Training complete")
    print("Best model:", best_model_name)
    print("Metrics:")
    for name, vals in metrics.items():
        print(name, vals)


if __name__ == "__main__":
    main()