import pandas as pd
import numpy as np
import sys
from pathlib import Path
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import BORJOMI_PRODUCTS, PROJECT_CONFIG, RAW_DATA_DIR

print("Запуск генератора данных")

class BorjomiDataGenerator:

    def __init__(self):
        self.products = BORJOMI_PRODUCTS
        self.config = PROJECT_CONFIG
        np.random.seed(42)

    def generate(self):
        print("Параметры генерации")
        print(f"Продуктов {len(self.products)}")
        print(f"Регионов {len(self.config['locations'])}")
        print(f"Дней {self.config['periods']:,}")
        print(f"Масшта/день: {self.config.get('demand_multiplier', 30):,} бутылок")

        data = []

        start_date = pd.to_datetime(self.config['start_date'])
        dates = pd.date_range(start_date, periods=self.config['periods'], freq='D')

        locations = self.config['locations']
        multiplier = self.config.get('demand_multiplier', 30)

        print("Генерация по продуктам...")

        for product_id, info in self.products.items():
            print(f" {info['name']}...")

            for location in locations:
                base_demand = np.random.randint(20, 50)

                for date in dates:
                    days_passed = (date - start_date).days
                    trend = 1 + (days_passed / len(dates)) * 0.25

                    season = 1 + 0.4 * np.sin(2 * np.pi * date.dayofyear / 365)

                    weekday = 1.25 if date.weekday() >= 4 else 1.0

                    holiday = 1.0
                    if date.month == 1 and date.day <= 8: holiday = 1.5
                    elif date.month == 12 and date.day >= 25: holiday = 1.3
                    elif date.month == 9 and date.day == 1: holiday = 1.15

                    region_boost = 1.4 if location == "Moscow" else 1.0

                    noise = np.random.normal(1.0, 0.12)

                    single_unit = max(1, int(base_demand * trend * season * weekday * holiday * region_boost * noise))

                    demand = single_unit * multiplier

                    price = round(100 + info['volume'] * 35 + np.random.uniform(-20,20),2)


                    data.append({
                        'product_id': product_id,
                        'product_name': info['name'],
                        'volume_litres': info['volume'],
                        'category': info['category'],
                        'date': date,
                        'quantity_sold': demand,
                        'price': price,
                        'store_location': location,
                    })
        
        df = pd.DataFrame(data)
        print("Генерация завершена")
        print(f"Всего записей: {len(df):,}")
        print(f"Средний спрос/день: {df['quantity_sold'].mean():.0f} шт")
        print(f"Средняя цена: {df['price'].mean():.1f} Р")

        print(f"\n Спрос по категориям:")
        stats = df.groupby('category')['quantity_sold'].agg(['count', 'mean', 'sum']).round(0)
        print(stats)

        return df
    
if __name__ == '__main__':
    gen = BorjomiDataGenerator()
    df = gen.generate()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"borjomi_{len(df):,}records.csv"
    csv_path = RAW_DATA_DIR / filename
    df.to_csv(csv_path, index=False)

    print(f"Файл сохранен: {csv_path}")
    print(f"размер файла: {csv_path.stat().st_size / 1e6:.1f} MB")
    print(f"\n Первые 10 записей:")
    print(df.head(10).to_string(index=False))




