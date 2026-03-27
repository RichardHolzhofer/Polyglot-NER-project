import sys
from typing import List, Dict, Any, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from src.config import NERConfig
from src.predictor import NERPredictor
from src.logger import logging
from src.exception import NERException

# Global predictor instance
predictor: Optional[NERPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads the model once when the server starts.
    """
    global predictor
    try:
        logging.info("Starting up FastAPI server...")
        config = NERConfig()
        predictor = NERPredictor(config)
        logging.info("Model loaded and predictor initialized.")
        yield
    except Exception as e:
        logging.error(f"Startup failed: {e}")
        # We don't want the server to start if the model fails to load
        raise NERException(e, sys)
    finally:
        logging.info("Shutting down FastAPI server...")

app = FastAPI(
    title="Polyglot NER API",
    description="Multilingual Named Entity Recognition serving Hungarian and German.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models ---

class PredictionRequest(BaseModel):
    items: Union[str, List[str]] = Field(
        ..., 
        description="A single text string or a list of strings to analyze.",
        example="Kovács János az OTP Bank igazgatója."
    )

class EntityResponse(BaseModel):
    word: str
    entity_group: str
    score: float
    start: int
    end: int

class PredictionResponse(BaseModel):
    results: Union[List[EntityResponse], List[List[EntityResponse]]]

# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Analyze text and extract named entities.
    Supports both single strings and batches.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor not initialized.")
    
    try:
        prediction_results = predictor.predict(request.items)
        return {"results": prediction_results}
    except NERException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error during API prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
