from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    store_id: int = Field(..., ge=1, le=400, description="ID магазина (1–400)")
    region: str = Field(..., description="Регион магазина")
    store_type: str = Field(..., description="Тип магазина")
    month: int = Field(..., ge=1, le=12, description="Месяц (1–12)")
    weekday: int = Field(..., ge=0, le=6, description="День недели (0=пн, 6=вс)")
    day_of_year: int = Field(..., ge=1, le=366, description="День года")
    is_weekend: bool = Field(..., description="Выходной день")
    is_holiday: bool = Field(..., description="Праздничный день")
    promo_flag: bool = Field(..., description="Активна промо-акция")
    temperature: float = Field(..., ge=-30.0, le=50.0, description="Температура в °C")
    price: float = Field(..., ge=10.0, le=200.0, description="Цена за единицу (₽)")

    class Config:
        json_schema_extra = {
            "example": {
                "store_id": 25,
                "region": "Москва",
                "store_type": "Супермаркет",
                "month": 7,
                "weekday": 5,
                "day_of_year": 200,
                "is_weekend": True,
                "is_holiday": False,
                "promo_flag": False,
                "temperature": 28.5,
                "price": 58.0
            }
        }