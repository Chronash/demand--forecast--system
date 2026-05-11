from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("artifacts/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "sales_daily.csv"

RANDOM_SEED = 42
TARGET_ROWS = 800_000
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

REGIONS = ["Москва", "Санкт-Петербург", "Поволжье", "Урал", "Сибирь", "Юг"]
STORE_TYPES = ["Супермаркет", "Минимаркет", "Гипермаркет", "АЗС"]
STORE_IDS = list(range(1, 401))

REGION_TEMP_OFFSETS = {
    "Москва": 0,
    "Санкт-Петербург": -2,
    "Поволжье": 1,
    "Урал": -4,
    "Сибирь": -8,
    "Юг": 6,
}

REGION_DEMAND_MULTIPLIER = {
    "Москва": 1.18,
    "Санкт-Петербург": 1.08,
    "Поволжье": 0.97,
    "Урал": 0.94,
    "Сибирь": 0.92,
    "Юг": 1.12,
}

STORE_TYPE_MULTIPLIER = {
    "Супермаркет": 1.12,
    "Минимаркет": 0.88,
    "Гипермаркет": 1.28,
    "АЗС": 0.72,
}

STORE_TYPE_PRICE_BASE = {
    "Супермаркет": 94,
    "Минимаркет": 108,
    "Гипермаркет": 88,
    "АЗС": 118,
}

REGION_PRICE_ADJ = {
    "Москва": 6,
    "Санкт-Петербург": 5,
    "Поволжье": -2,
    "Урал": 0,
    "Сибирь": 3,
    "Юг": 1,
}

HOLIDAYS = {
    "2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07",
    "2022-02-23", "2022-03-08", "2022-05-01", "2022-05-09", "2022-06-12", "2022-11-04",
    "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-07",
    "2023-02-23", "2023-03-08", "2023-05-01", "2023-05-09", "2023-06-12", "2023-11-04",
    "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07",
    "2024-02-23", "2024-03-08", "2024-05-01", "2024-05-09", "2024-06-12", "2024-11-04",
    "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
    "2025-02-23", "2025-03-08", "2025-05-01", "2025-05-09", "2025-06-12", "2025-11-04",
}


def seasonal_temperature(month: int) -> float:
    monthly_avg = {
        1: -10, 2: -8, 3: -1, 4: 8, 5: 16, 6: 22,
        7: 25, 8: 23, 9: 16, 10: 8, 11: 0, 12: -7,
    }
    return monthly_avg[month]


def seasonal_demand_boost(month: int) -> float:
    if month in [5, 6, 7, 8, 9]:
        return 18
    if month in [12, 1]:
        return -6
    return 4


def temperature_effect(temperature: float) -> float:
    if temperature <= -15:
        return -10
    if temperature <= 0:
        return -6
    if temperature <= 10:
        return 0
    if temperature <= 20:
        return (temperature - 10) * 1.8
    if temperature <= 30:
        return 18 + (temperature - 20) * 2.4
    return 42 + (temperature - 30) * 1.2


def promo_probability(month: int, store_type: str) -> float:
    base = 0.12
    if month in [5, 6, 7, 8]:
        base += 0.04
    if store_type == "Гипермаркет":
        base += 0.03
    if store_type == "АЗС":
        base -= 0.02
    return min(max(base, 0.05), 0.25)


def generate_base_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rows = []
    rows_per_day = int(np.ceil(TARGET_ROWS / len(dates)))

    for current_date in dates:
        month = current_date.month
        weekday = current_date.weekday()
        day_of_year = current_date.dayofyear
        is_weekend = int(weekday >= 5)
        is_holiday = int(str(current_date.date()) in HOLIDAYS)

        for _ in range(rows_per_day):
            region = rng.choice(REGIONS)
            store_type = rng.choice(STORE_TYPES, p=[0.42, 0.24, 0.18, 0.16])
            store_id = int(rng.choice(STORE_IDS))

            base_temp = seasonal_temperature(month)
            regional_temp = base_temp + REGION_TEMP_OFFSETS[region]
            temperature = regional_temp + rng.normal(0, 5.5)
            temperature = float(np.clip(temperature, -35, 35))

            promo_flag = int(rng.random() < promo_probability(month, store_type))
            promo_discount = rng.uniform(4, 12) if promo_flag else 0.0

            price = (
                STORE_TYPE_PRICE_BASE[store_type]
                + REGION_PRICE_ADJ[region]
                + rng.normal(0, 6.5)
                - promo_discount
            )
            price = float(np.clip(price, 70, 140))

            weekend_effect = 7 if is_weekend else 0
            holiday_effect = 12 if is_holiday else 0
            promo_effect = 16 if promo_flag else 0
            season_boost = seasonal_demand_boost(month)
            temp_boost = temperature_effect(temperature)
            price_effect = -(price - 95) * 1.7

            base_demand = 52
            demand = (
                base_demand
                * REGION_DEMAND_MULTIPLIER[region]
                * STORE_TYPE_MULTIPLIER[store_type]
                + season_boost
                + temp_boost
                + weekend_effect
                + holiday_effect
                + promo_effect
                + price_effect
                + rng.normal(0, 5)
            )

            demand = max(5, round(demand))
            revenue = round(demand * price, 2)

            rows.append(
                {
                    "sale_date": current_date.strftime("%Y-%m-%d"),
                    "store_id": store_id,
                    "region": region,
                    "store_type": store_type,
                    "month": month,
                    "weekday": weekday,
                    "day_of_year": day_of_year,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "promo_flag": promo_flag,
                    "temperature": round(temperature, 1),
                    "price": round(price, 2),
                    "demand_units": demand,
                    "revenue": revenue,
                }
            )

            if len(rows) >= TARGET_ROWS:
                break

        if len(rows) >= TARGET_ROWS:
            break

    df = pd.DataFrame(rows).sort_values(["store_id", "sale_date"]).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values(["store_id", "sale_date"]).reset_index(drop=True)

    group = df.groupby("store_id")["demand_units"]

    df["lag_1"] = group.shift(1)
    df["lag_7"] = group.shift(7)
    df["lag_14"] = group.shift(14)
    df["rolling_mean_7"] = group.shift(1).rolling(7).mean()
    df["rolling_mean_14"] = group.shift(1).rolling(14).mean()
    df["rolling_std_7"] = group.shift(1).rolling(7).std()

    df["price_delta_vs_lag1"] = df["price"] - df.groupby("store_id")["price"].shift(1)
    df["temp_delta_vs_lag1"] = df["temperature"] - df.groupby("store_id")["temperature"].shift(1)

    df["days_to_month_end"] = df["sale_date"].dt.days_in_month - df["sale_date"].dt.day
    df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["sale_date"].dt.quarter

    df["lag_1"] = df["lag_1"].fillna(df["demand_units"].median())
    df["lag_7"] = df["lag_7"].fillna(df["demand_units"].median())
    df["lag_14"] = df["lag_14"].fillna(df["demand_units"].median())
    df["rolling_mean_7"] = df["rolling_mean_7"].fillna(df["demand_units"].median())
    df["rolling_mean_14"] = df["rolling_mean_14"].fillna(df["demand_units"].median())
    df["rolling_std_7"] = df["rolling_std_7"].fillna(0)
    df["price_delta_vs_lag1"] = df["price_delta_vs_lag1"].fillna(0)
    df["temp_delta_vs_lag1"] = df["temp_delta_vs_lag1"].fillna(0)

    return df


def main():
    df = generate_base_dataset()
    df = add_time_features(df)
    df = df.sort_values("sale_date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    print("Price min/max:", df["price"].min(), df["price"].max())
    print("Temperature min/max:", df["temperature"].min(), df["temperature"].max())
    print("Demand min/max:", df["demand_units"].min(), df["demand_units"].max())
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()