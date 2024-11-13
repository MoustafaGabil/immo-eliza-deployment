import joblib
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from fastapi import HTTPException


# Initialize geolocator
geolocator = Nominatim(user_agent="property_price_predictor")

# Load models and encoders
house_model = joblib.load(r"models and encoders/XGB_Regression_HOUSE_without_outliers.pkl")
apartment_model = joblib.load(r"models and encoders/XGB_Regression_APARTMENT_without_outliers.pkl")
locality_encoder = joblib.load(r"models and encoders/locality_encoder.joblib")
kitchen_encoder = joblib.load(r"models and encoders/encoder_kitchen_type.joblib")
building_state_encoder = joblib.load(r"models and encoders/encoder_building_state.joblib")
epc_encoder = joblib.load(r"models and encoders/encoder_epc.joblib")

# MSE values for house and apartment
MSE_VALUES = {
    "house": 50102,
    "apartment": 31681,
}

# Valid categories for kitchen type, building state, and EPC
VALID_KITCHEN_TYPES = ["NOT_INSTALLED", "SEMI_EQUIPPED", "USA_SEMI_EQUIPPED", "INSTALLED", "USA_HYPER_EQUIPPED"]
VALID_BUILDING_STATES = ["TO_RESTORE", "TO_RENOVATE", "TO_BE_DONE_UP", "GOOD", "JUST_RENOVATED", "AS_NEW"]
VALID_EPC_RATINGS = ["G", "F", "E", "D", "C", "B", "A", "A+", "A++"]

# Function to get latitude and longitude based on zip code and locality
def get_lat_lon(zip_code: int, locality: str):
    try:
        location = geolocator.geocode(f"{locality}, {zip_code}")
        if location:
            return location.latitude, location.longitude
        else:
            raise HTTPException(status_code=404, detail="Location not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error fetching location details.")

# Function to validate integer inputs
def validate_int(value, field_name):
    if not isinstance(value, int):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input for {field_name}. Please enter an integer value.",
        )
    if value < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input for {field_name}. Value must be a positive integer.",
        )

# Function to validate binary inputs (1 or 0)
def validate_binary(value, field_name):
    if value not in [0, 1]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input for {field_name}. Please enter either 0 or 1.",
        )

# Function to validate logical relationships between areas
def validate_area_relationships(surface_land_sqm, total_area_sqm, garden_sqm):
    if garden_sqm > surface_land_sqm:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: Garden size cannot be larger than the surface land.",
        )
    if total_area_sqm > surface_land_sqm:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: Total area cannot be larger than the surface land.",
        )
    if garden_sqm > total_area_sqm:
        raise HTTPException(
            status_code=400,
            detail="Invalid input: Garden size cannot be larger than the total area.",
        )

# Function to validate locality input
def validate_locality(locality, locality_encoder):
    valid_localities = locality_encoder.categories_[0].tolist()
    if locality not in valid_localities:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid locality. Please choose a locality from the list: {', '.join(valid_localities)}."
        )

# Function to validate categorical fields
def validate_categorical(value, valid_values, field_name):
    if value not in valid_values:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input for {field_name}. Valid options are: {', '.join(valid_values)}.",
        )

# Prediction function
def get_prediction(property_details) -> dict:
    # Validate inputs
    if property_details.property_type.lower() not in ["house", "apartment"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid property_type. Choose 'house' or 'apartment'.",
        )

    # Validate integers
    validate_int(property_details.construction_year, "Construction Year")
    validate_int(property_details.nbr_bedrooms, "Number of Bedrooms")
    validate_int(property_details.nbr_frontages, "Number of Frontages")

    # Validate binary inputs
    validate_binary(property_details.fl_double_glazing, "Double Glazing")
    validate_binary(property_details.fl_terrace, "Terrace")
    validate_binary(property_details.fl_swimming_pool, "Swimming Pool")
    validate_binary(property_details.fl_floodzone, "Flood Zone")

    # Validate area relationships
    validate_area_relationships(
        property_details.surface_land_sqm,
        property_details.total_area_sqm,
        property_details.garden_sqm,
    )

    # Validate locality
    validate_locality(property_details.locality, locality_encoder)

    # Validate kitchen type, building state, and EPC
    validate_categorical(property_details.kitchen_type, VALID_KITCHEN_TYPES, "Kitchen Type")
    validate_categorical(property_details.building_state, VALID_BUILDING_STATES, "Building State")
    validate_categorical(property_details.epc, VALID_EPC_RATINGS, "EPC Rating")

    # Fetch latitude and longitude
    latitude, longitude = get_lat_lon(property_details.zip_code, property_details.locality)

    # Encode categorical values
    try:
        locality_encoded = locality_encoder.transform([[property_details.locality]])
        kitchen_encoded = kitchen_encoder.transform([[property_details.kitchen_type]])[0][0]
        building_encoded = building_state_encoder.transform([[property_details.building_state]])[0][0]
        epc_encoded = epc_encoder.transform([[property_details.epc]])[0][0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Prepare features for prediction
    features = {
        "construction_year": property_details.construction_year,
        "total_area_sqm": property_details.total_area_sqm,
        "nbr_frontages": property_details.nbr_frontages,
        "nbr_bedrooms": property_details.nbr_bedrooms,
        "kitchen_type_encoded": kitchen_encoded,
        "building_state_encoded": building_encoded,
        "epc_encoded": epc_encoded,
        "garden_sqm": property_details.garden_sqm,
        "surface_land_sqm": property_details.surface_land_sqm,
        "fl_double_glazing": property_details.fl_double_glazing,
        "fl_terrace": property_details.fl_terrace,
        "fl_swimming_pool": property_details.fl_swimming_pool,
        "fl_floodzone": property_details.fl_floodzone,
        "latitude": latitude,
        "longitude": longitude,
        "zip_code": property_details.zip_code,
    }

    # Convert to DataFrame and concatenate locality encoding
    features_df = pd.DataFrame([features])
    input_data = np.concatenate([features_df.values, locality_encoded], axis=1)

    # Select model based on property type
    model = house_model if property_details.property_type.lower() == "house" else apartment_model

    # Make the prediction
    prediction = model.predict(input_data)[0]
    prediction = int(prediction)

    # Calculate the price range
    mse = MSE_VALUES[property_details.property_type.lower()]
    price_range = {"min": prediction - mse, "max": prediction + mse}

    # Return the prediction in the expected format
    return {
        "property_type": property_details.property_type,
        "predicted_price": prediction,
        "predicted_price_range": price_range
    }
