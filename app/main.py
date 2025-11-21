from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="House Price Predictor API",
    description="This API predicts the price of a house using 5 important features.",
    version="1.0"
)

# Input features with descriptions
class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., description="Number of bedrooms in the house")
    bathrooms: int = Field(..., description="Number of bathrooms in the house")
    sqft: float = Field(..., description="Total built-up area in square feet")
    age: float = Field(..., description="Age of the house in years")
    location_score: float = Field(
        ..., 
        description="Location rating from 1 to 10 (10 = best area)"
    )

# Output response
class PredictionResponse(BaseModel):
    message: str


# Load the ML model
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("model.pkl")
    except Exception as e:
        raise RuntimeError(f"Could not load model.pkl: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict_price(data: HouseFeatures):
    """
    Predicts house price based on 5 features.
    """

    try:
        features = np.array([
            data.bedrooms,
            data.bathrooms,
            data.sqft,
            data.age,
            data.location_score
        ]).reshape(1, -1)

        price = model.predict(features)[0]

        # Format message (your requirement)
        formatted_price = f"Estimated house value: ₹{int(price):,}"

        return PredictionResponse(message=formatted_price)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running."}



