from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes.health import router as health_router
from app.api.routes.predict import router as predict_router

app = FastAPI(
    title="Borjomi Demand Forecast API",
    description="Система прогнозирования спроса на минеральную воду",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)


@app.on_event("startup")
async def startup_event():
    logger.info("API запущен. Документация: http://localhost:8000/docs")