from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("artifacts/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

dates = pd.date_range(start="2021-01-01", end="2024-12-31", freq="D")
regions = ["Москва", "Санкт-Петербург", "Поволжье", "Урал", "Сибирь", "Юг"]
store_types = ["Супермаркет", "Минимаркет", "Гипермаркет", "АЗС"]

n_stores = 400
stores = []
for i in range(1, n_stores + 1):
    stores.append({
        "store_id": i,
        "region": np.random.choice(regions),
        "store_type": np.random.choice(store_types),
        "base_demand": np.random.randint(80, 260),
        "price_base": np.random.uniform(45, 70),
    })
stores_df = pd.DataFrame(stores)

rows = []
for _, store in stores_df.iterrows():
    for d in dates:
        weekday = d.weekday()
        is_weekend = weekday >= 5
        is_holiday = (
            (d.month == 1 and d.day <= 8)
            or (d.month == 5 and d.day in [1, 9])
            or (d.month == 12 and d.day == 31)
        )
        promo_flag = np.random.rand() < 0.12
        seasonal = 35 * np.sin(2 * np.pi * d.dayofyear / 365)
        temp = (
            8
            + 18 * np.sin(2 * np.pi * (d.dayofyear - 80) / 365)
            + np.random.normal(0, 4)
        )
        weekend_boost = 20 if is_weekend else 0
        holiday_boost = 35 if is_holiday else 0
        promo_boost = 18 if promo_flag else 0
        heat_boost = max(0, temp - 18) * 2.4
        price = store["price_base"] + np.random.normal(0, 2)
        price_penalty = max(0, price - 55) * 2.1
        noise = np.random.normal(0, 10)

        demand = max(
            5,
            int(round(
                store["base_demand"]
                + seasonal
                + weekend_boost
                + holiday_boost
                + promo_boost
                + heat_boost
                - price_penalty
                + noise
            ))
        )
        revenue = round(demand * price, 2)

        rows.append({
            "sale_date": d.date(),
            "store_id": int(store["store_id"]),
            "region": store["region"],
            "store_type": store["store_type"],
            "month": d.month,
            "weekday": weekday,
            "day_of_year": d.dayofyear,
            "is_weekend": int(is_weekend),
            "is_holiday": int(is_holiday),
            "promo_flag": int(promo_flag),
            "temperature": round(temp, 2),
            "price": round(price, 2),
            "demand_units": demand,
            "revenue": revenue,
        })

sales_df = pd.DataFrame(rows)
sales_df.to_csv(OUTPUT_DIR / "sales_daily.csv", index=False)
stores_df[["store_id", "region", "store_type"]].to_csv(
    OUTPUT_DIR / "stores.csv", index=False
)

print(f"Сгенерировано {len(sales_df):,} строк")
print(f"Файлы сохранены в: {OUTPUT_DIR.resolve()}")