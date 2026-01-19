import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
print(f"Корень проекта : {BASE_DIR}")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


print(f" Папки данных:")
print(f" Сырые : {RAW_DATA_DIR}")
print(f" Обработаные : {PROCESSED_DATA_DIR}")
print(f" Модели : {MODELS_DIR}")


DB_USER = "postgres"
DB_PASSWORD = "Franch2006"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "borjomi_forecast"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(f" Бд подключения : {DATABASE_URL}")
 
BORJOMI_PRODUCTS = {
  "BOR-001": {"name": "Классическая Borjomi 0.33л", "volume": 0.33, "category": "Classic"},
  "BOR-002": {"name": "Классическая Borjomi 0.5л", "volume": 0.5, "category": "Classic"},
  "BOR-003": {"name": "Классическая Borjomi 0.75л", "volume": 0.75, "category": "Classic"},
  "BOR-004": {"name": "Классическая Borjomi 1.0л", "volume": 1.0, "category": "Classic"},
  "BOR-005": {"name": "Классическая Borjomi 1.2л", "volume": 1.2, "category": "Classic"},
  "BOR-006": {"name": "Негазированная Borjomi 0.5л", "volume": 0.33, "category": "NonCarbonated"},
  "BOR-007": {"name": "Flavored Borjomi Вишня-Гранат 0.33л", "volume": 0.33, "category": "Flavored"},
  "BOR-008": {"name": "Flavored Borjomi Вишня-Гранат 0.5л", "volume": 0.5, "category": "Flavored"},
  "BOR-009": {"name": "Flavored Borjomi Вишня-Гранат 1.0л", "volume": 1.0, "category": "Flavored"},
  "BOR-010": {"name": "Flavored Borjomi Цитрус-Имбирь 0.33л", "volume": 0.33, "category": "Flavored"},
  "BOR-011": {"name": "Flavored Borjomi Цитрус-Имбирь 0.5л", "volume": 0.5, "category": "Flavored"},
  "BOR-012": {"name": "Flavored Borjomi Цитрус-Имбирь 1.0л", "volume": 1.0, "category": "Flavored"},
  "BOR-013": {"name": "Limonati Borjomi Груша 0.33л", "volume": 0.33, "category": "Limonati"},
  "BOR-014": {"name": "Limonati Borjomi Груша 0.5л", "volume": 0.5, "category": "Limonati"},
  "BOR-015": {"name": "Limonati Borjomi Манадарин 0.33л", "volume": 0.33, "category": "Limonati"},
  "BOR-016": {"name": "Limonati Borjomi Мандарин 0.5л", "volume": 0.5, "category": "Limonati"},
  "BOR-017": {"name": "Limonati Borjomi Цитрус 0.33л", "volume": 0.33, "category": "Limonati"},
  "BOR-018": {"name": "Limonati Borjomi Цитрус 0.5л", "volume": 0.5, "category": "Limonati"},
  "BOR-019": {"name": "Limonati Borjomi Тархун 0.33л", "volume": 0.33, "category": "Limonati"},
  "BOR-020": {"name": "Limonati Borjomi Тархун 0.5л", "volume": 0.5, "category": "Limonati"},
}

PROJECT_CONFIG = {
    "start_date": "2021-01-01",
    "periods": 1095,
    "locations": {"Moscow", "SPB","Ekaterinburg","Kazan","Rostov","Krasnodar"},
    "demand_multiplier":30,
    "test_size":0.2,
}

XGB_PARAMS = {
    "max_depth": 7,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "random_state": 7,
    "n_jobs": -1,
}

print("cofig загружен")
print(f"продуктов боржоми: {len(BORJOMI_PRODUCTS)}")
print(f"регионов: {len(PROJECT_CONFIG['locations'])}")
print(f"дней: {PROJECT_CONFIG['periods']:,}")