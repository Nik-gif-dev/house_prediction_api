from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

app = FastAPI(title="House Price Predictor", version="1.0")

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: float

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    arr = np.array(req.features).reshape(1, -1)
    pred = model.predict(arr)[0]
    return PredictResponse(prediction=float(pred))

@app.get("/")
def home():
    return {"message": "House Prediction Model API"}

