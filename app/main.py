from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="House Price Predictor")

# Five important features
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft: float
    age: float
    location_score: float


class PredictionResponse(BaseModel):
    prediction: int  # changed to int


# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("model.pkl")
    except Exception as e:
        raise RuntimeError(f"Could not load model.pkl: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict_price(data: HouseFeatures):

    try:
        features = np.array([
            data.bedrooms,
            data.bathrooms,
            data.sqft,
            data.age,
            data.location_score
        ]).reshape(1, -1)

        price = model.predict(features)[0]
        price = round(float(price))   # <<< THIS IS THE CHANGE

        return PredictionResponse(prediction=price)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running."}

