from fastapi import APIRouter, HTTPException
from loguru import logger

from app.api.schemas.prediction import PredictionRequest
from app.api.schemas.response import PredictionResponse
from app.api.services.model_service import ModelService

router = APIRouter(prefix="/predict", tags=["predict"])
model_service = ModelService()


@router.post("", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        result = model_service.predict(payload)
        logger.info(
            f"Прогноз: store_id={payload.store_id}, "
            f"region={payload.region}, "
            f"demand={result['predicted_demand']}"
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка прогноза: {e}")
        raise HTTPException(status_code=500, detail="Ошибка сервера")