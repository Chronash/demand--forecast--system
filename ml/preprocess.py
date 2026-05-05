import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path("artifacts/generated")
MODELS_DIR = Path("artifacts/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "store_id",
    "region_encoded",
    "store_type_encoded",
    "month",
    "weekday",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "promo_flag",
    "temperature",
    "price",
]

TARGET_COLUMN = "demand_units"


def load_and_prepare():
    df = pd.read_csv(DATA_DIR / "sales_daily.csv")
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values("sale_date").reset_index(drop=True)

    region_encoder = LabelEncoder()
    store_type_encoder = LabelEncoder()

    df["region_encoded"] = region_encoder.fit_transform(df["region"])
    df["store_type_encoded"] = store_type_encoder.fit_transform(df["store_type"])

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(region_encoder, MODELS_DIR / "label_encoder_region.pkl")
    joblib.dump(store_type_encoder, MODELS_DIR / "label_encoder_store_type.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    print(f"Train: {len(X_train):,} строк | Test: {len(X_test):,} строк")
    print("Артефакты сохранены в artifacts/models/")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    load_and_prepare()