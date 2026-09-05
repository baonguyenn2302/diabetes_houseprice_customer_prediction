# ==========================================
# REST API FOR INTELLIGENT SYSTEMS (FASTAPI)
# ==========================================
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Intelligent Systems API - Assignment 02",
    description="API Gateway for Diabetes, House Price, and Customer Behavior Predictions",
    version="1.0.0"
)

# Cấu hình CORS để Web và Mobile App kết nối không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# LOAD SAVED PIPELINES & MODELS
# ------------------------------------------
# 1. Diabetes Models
DIABETES_PREP_PATH = "diabetes/model/preprocessor.joblib"
DIABETES_MODEL_PATH = "diabetes/model/model.joblib"

# 2. House Price Models
HOUSE_PREP_PATH = "house_price/model/preprocessor.joblib"
HOUSE_MODEL_PATH = "house_price/model/model.joblib"

# 3. Customer Behavior Models
CUSTOMER_PREP_PATH = "customer_behavior/model/preprocessor.joblib"
CUSTOMER_MODEL_PATH = "customer_behavior/model/model.joblib"

try:
    diabetes_prep = joblib.load(DIABETES_PREP_PATH)
    diabetes_model = joblib.load(DIABETES_MODEL_PATH)
    
    house_prep = joblib.load(HOUSE_PREP_PATH)
    house_model = joblib.load(HOUSE_MODEL_PATH)
    
    customer_prep = joblib.load(CUSTOMER_PREP_PATH)
    customer_model = joblib.load(CUSTOMER_MODEL_PATH)
    print("All ML Pipelines and Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}. Please ensure all .joblib files exist.")

# ------------------------------------------
# INPUT SCHEMAS (PYDANTIC VALIDATION)
# ------------------------------------------
class DiabetesInput(BaseModel):
    times_pregnant: int = Field(..., example=2)
    glucose_concentration: float = Field(..., example=148.0)
    diastolic_blood_pressure: float = Field(..., example=72.0)
    triceps_sf_thickness: float = Field(..., example=35.0)
    serum_insulin: float = Field(..., example=0.0)
    bmi: float = Field(..., example=33.6)
    d_pedigree_function: float = Field(..., example=0.627)
    years_of_age: int = Field(..., example=50)

class HousePriceInput(BaseModel):
    GrLivArea: float = Field(..., example=1710.0)
    BedroomAbvGr: int = Field(..., example=3)
    FullBath: int = Field(..., example=2)
    YearBuilt: int = Field(..., example=2003)
    GarageCars: int = Field(..., example=2)
    LotArea: float = Field(..., example=8450.0)
    MSZoning: str = Field(..., example="RL")
    Neighborhood: str = Field(..., example="CollgCr")
    HouseStyle: str = Field(..., example="2Story")

class CustomerBehaviorInput(BaseModel):
    num_helpful: int = Field(..., example=2)
    customer_review: str = Field(..., example="This tablet has an amazing display and fast performance.")

# ------------------------------------------
# API ENDPOINTS
# ------------------------------------------
@app.get("/")
def root():
    return {"status": "Online", "message": "Welcome to Intelligent Systems Prediction API"}

# Endpoint 1: Diabetes Prediction
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        df_input = pd.DataFrame([data.model_dump()])
        X_prep = diabetes_prep.transform(df_input)
        pred = diabetes_model.predict(X_prep)[0]
        prob = diabetes_model.predict_proba(X_prep)[0][pred] if hasattr(diabetes_model, "predict_proba") else 1.0
        
        return {
            "application": "Diabetes Prediction",
            "prediction": "Diabetes Positive" if pred == 1 else "Diabetes Negative",
            "class_id": int(pred),
            "confidence": round(float(prob), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: House Price Prediction
@app.post("/predict/house-price")
def predict_house_price(data: HousePriceInput):
    try:
        df_input = pd.DataFrame([data.model_dump()])
        X_prep = house_prep.transform(df_input)
        pred_price = house_model.predict(X_prep)[0]
        
        return {
            "application": "House Price Prediction",
            "predicted_price": round(float(pred_price), 2),
            "currency": "USD"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 3: Customer Behavior Discovery
@app.post("/predict/customer-behavior")
def predict_customer_behavior(data: CustomerBehaviorInput):
    try:
        df_input = pd.DataFrame([data.model_dump()])
        X_prep = customer_prep.transform(df_input)
        pred = customer_model.predict(X_prep)[0]
        prob = customer_model.predict_proba(X_prep)[0][pred] if hasattr(customer_model, "predict_proba") else 1.0
        
        return {
            "application": "E-commerce Customer Behavior",
            "predicted_interest": "High Customer Interest" if pred == 1 else "Low Customer Interest",
            "class_id": int(pred),
            "confidence": round(float(prob), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))