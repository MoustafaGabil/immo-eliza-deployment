from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import predict
from typing import Optional

# Initialize FastAPI
app = FastAPI()

# ---------------------------------- Pydantic Models ----------------------------------

class PropertyDetails(BaseModel):
    property_type: Optional[str] = "house"
    locality: Optional[str] = "Gent"
    zip_code: Optional[int]= 9000
    construction_year: Optional[int] = 2020  
    total_area_sqm: Optional[float] = 100  
    surface_land_sqm: Optional[float] = 200
    garden_sqm: Optional[float] = 0
    nbr_frontages: Optional[int] = 2  
    nbr_bedrooms: Optional[int] = 3 
    kitchen_type: Optional[str] = "NOT_INSTALLED"  
    building_state: Optional[str] = "GOOD"  
    epc: Optional[str] = "C"  
    fl_double_glazing: Optional[int] = 1 
    fl_terrace: Optional[int] = 0  
    fl_swimming_pool: Optional[int] = 0  
    fl_floodzone: Optional[int] = 0  
    
class PricePredictionResponse(BaseModel):
    property_type: str
    predicted_price: int
    predicted_price_range: dict

# ---------------------------------- Endpoints ----------------------------------

@app.get("/")
def health_check():
    return {"status": "alive"}
@app.get("/about")
def about():
    return {
        "title": "Property Price Prediction API",
        "description": "This API predicts the price of properties (houses or apartments) based on various features such as construction year, area, location, and more.",
        "usage_instructions": [
            "To get an accurate prediction, make sure to enter the correct location (locality) and zip code.",
            "The location and zip code are used to fetch the latitude and longitude, which are important for accurate price estimation.",
            "Please double-check that the locality name matches exactly, and ensure the zip code corresponds to that locality.",
            "Incorrect or missing location information may lead to inaccurate or failed predictions."
        ],
        "note": "Be careful when entering the location and zip code to ensure accurate results. The more precise your inputs, the better the prediction accuracy.",
    }

@app.post("/predict", response_model=PricePredictionResponse)
def predict_price(property_details: PropertyDetails):
    try:
        # Call the predict function from predict.py
        prediction = predict.get_prediction(property_details)
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
















