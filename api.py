"""FastAPI prediction API for Karachi housing prices."""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ARTIFACTS_PATH
from src.model import load_all_models, predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI(title="Karachi Housing Price Prediction API", version="2.0.0")

pipelines = {}
metadata = None


@app.on_event("startup")
def startup():
    global pipelines, metadata
    if not ARTIFACTS_PATH.exists():
        raise RuntimeError("No trained model found. Run 'python train.py' first.")
    pipelines, metadata = load_all_models()
    logging.info("Loaded %d models, best: %s", len(pipelines), metadata["best_model"])


class PredictionRequest(BaseModel):
    property_type: str = Field(..., examples=["House"])
    location: str = Field(..., examples=["DHA Phase 6"])
    area: float = Field(..., gt=0, description="Area in square yards")
    beds: int = Field(..., ge=0, examples=[3])
    baths: int = Field(..., ge=0, examples=[2])
    model: Optional[str] = Field(None, description="Model to use. If omitted, uses the best model.")


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    currency: str = "PKR"
    model_used: str
    is_recommended_model: bool


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(pipelines)}


@app.get("/model/info")
def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "best_model": metadata["best_model"],
        "available_models": metadata["available_models"],
        "trained_at": metadata["trained_at"],
        "train_size": metadata["train_size"],
        "test_size": metadata["test_size"],
        "all_results": metadata["all_results"],
        "known_types": metadata["known_types"],
        "known_locations": metadata["known_locations"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_price(req: PredictionRequest):
    if not pipelines:
        raise HTTPException(status_code=503, detail="Models not loaded")

    model_name = req.model or metadata["best_model"]

    if model_name not in pipelines:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model_name}'. Available: {list(pipelines.keys())}",
        )
    if req.property_type not in metadata["known_types"]:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown property type '{req.property_type}'. Known: {metadata['known_types']}",
        )
    if req.location not in metadata["known_locations"]:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown location '{req.location}'. Known: {metadata['known_locations']}",
        )

    price = max(0, predict(pipelines[model_name], req.property_type, req.location, req.area, req.beds, req.baths))

    return PredictionResponse(
        predicted_price=round(price, 0),
        predicted_price_formatted=f"PKR {price:,.0f}",
        currency="PKR",
        model_used=model_name,
        is_recommended_model=(model_name == metadata["best_model"]),
    )
