
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="House Price Prediction API")

# Define input features
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    age: int
    zipcode: int
    condition: int
    grade: int
    waterfront: int
    view: int
    renovated: int

# Load model
model = joblib.load("model.pkl")

@app.post("/predict")
def predict_price(data: HouseFeatures):
    features = [
        data.bedrooms,
        data.bathrooms,
        data.sqft_living,
        data.sqft_lot,
        data.floors,
        data.age,
        data.zipcode,
        data.condition,
        data.grade,
        data.waterfront,
        data.view,
        data.renovated
    ]

    arr = np.array(features).reshape(1, -1)
    prediction = model.predict(arr)[0]

    return {
        "predicted_price": float(prediction),
        "currency": "INR",
        "message": "Price prediction successful!"
    }




