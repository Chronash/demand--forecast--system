from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_demand: int
    predicted_revenue: float
    model_name: str
    model_version: str

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_demand": 185,
                "predicted_revenue": 10730.0,
                "model_name": "xgboost",
                "model_version": "1.0.0"
            }
        }