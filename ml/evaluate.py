import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

MODELS_DIR = Path("artifacts/models")
PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "store_id", "region_encoded", "store_type_encoded", "month",
    "weekday", "day_of_year", "is_weekend", "is_holiday",
    "promo_flag", "temperature", "price"
]

FEATURE_NAMES_RU = [
    "ID магазина", "Регион", "Тип магазина", "Месяц",
    "День недели", "День года", "Выходной", "Праздник",
    "Промо", "Температура", "Цена"
]


def plot_feature_importance():
    model = XGBRegressor()
    model.load_model(str(MODELS_DIR / "xgboost_model.json"))

    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(10, 7))
    plt.barh(
        [FEATURE_NAMES_RU[i] for i in sorted_idx],
        importance[sorted_idx],
        color="#2196F3"
    )
    plt.title("Важность признаков (XGBoost)", fontsize=14)
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.show()
    print("График сохранён в artifacts/plots/feature_importance.png")


if __name__ == "__main__":
    plot_feature_importance()