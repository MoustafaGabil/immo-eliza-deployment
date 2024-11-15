# Real Estate Price Prediction Project (immo-eliza-ml) 🏠
![Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![FastAPI](https://img.shields.io/badge/uses-FastAPI-green.svg)
![Scikit-learn](https://img.shields.io/badge/uses-Scikit--learn-orange.svg)



<p align="center">
  <img src="https://qobrix.com/wp-content/uploads/mariatitan/2023/03/API-Blog.png" />
</p>

## 🏢 Description

This API provides real estate price predictions based on a machine learning model trained on a comprehensive dataset of property features. The project includes thorough data preprocessing, feature engineering, and model training using XGBRegressor algorithm.


## Key functionalities of the API:

- Predict property prices with and .
- **Dynamic Feature Selection:** Accepts various property features, such as total area, garden size, energy efficiency, and kitchen type, for customized predictions.
- **Location-Based Precision:** Predictions incorporate detailed geographic features like zip code, locality, and coordinates for accurate regional estimates.
- **Detailed Property Characteristics:** Allows input of specific attributes such as construction year, number of bedrooms, and additional features like swimming pools or terraces.
- **Versatile Predictions:** Works for both houses and apartments, providing tailored estimates for different property types.
- **Visual Range Estimate:** Alongside the predicted price, it provides a range based on the model's Mean Squared Error (MSE) (+/- 50,102) for **Houses** and (+/- 31,681) for **Apartment**to help users assess price variabilit.


Access the API: [https://immo-eliza-deployment-fastapi.onrender.com]


### Issues and update requests
- If you have any questions, run into issues, or have ideas for improvement, please don’t hesitate to open an issue in the repository.
- Contributions to improve the models' functionality or performance of the API are highly encouraged and appreciated.


Find me on [LinkedIn](https://www.linkedin.com/in/moustafa-gabil-8a4a6bab/) for collaboration, feedback, or to connect.

## 📦 Repo structure
```.
├── models and encoders/
│      ├── encoder_building_state.joblib
│      ├── encoder_epc.joblib
│      ├── locality_encoder.joblib
│      ├── encoder_kitchen_type.joblib
│      ├── XGB_Regression_HOUSE_without_outliers.csv
│      └── XGB_Regression_APARTMENT_without_outliers.py
├── Dockerfile 
├── .gitignore
├── app.py
├── Predict.py
├── README.md
└── requirements.txt 

```
## 🚧 Requesrt/Response  

 ### **Base URL**
 [https://immo-eliza-deployment-fastapi.onrender.com/]
  
  ## Endpoints

 1. **Health Check**

    - Endpoint: /
    - Method: GET
    - Description: Verifies the API status.
    - Response: {"status": "alive"}
 2. **About**
    - Endpoint: /about
    - Method: GET
    - Description: Provides API details.
 3. **Predict Price**

    - Endpoint: /predict
    - Method: POST
    - Description: Predicts property price based on user inputs.

    ```python
            {
        "property_type": "house",
        "locality": "Gent",
        "zip_code": 9000,
        "construction_year": 2020,
        "total_area_sqm": 100,
        "nbr_bedrooms": 3
        ... etc
        }
    ```

```
**Notes**
- To get an accurate prediction, make sure to enter the correct location (locality) and zip code.
- Input all relevant property details and submit to receive an estimated price.
  

## 🔧 Updates & Upgrades

The data can be further preprocessed to get more accurate results and more models can be tested. 
```

## ⏱️ Project Timeline:
The initial setup of this project was completed in 6 days.


