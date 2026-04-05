from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path   

# ── App initialization ──────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description = "Predicts machine failure probability from sensor data using a pre-trained model.",
    version="1.0.0"
)

# ── Load model artifacts ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

try:
    model = joblib.load(BASE_DIR / "model.pkl")
    le = joblib.load(BASE_DIR / "label_encoder.pkl")
    print("Model and label encoder loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ── Input schema ─────────────────────────────────────────────────────────────
class SensorReadings(BaseModel):
    type: str = Field(..., description="Machine type: L, M or H")
    air_temperature: float = Field(..., description="Air temperature in Kelvin")
    process_temperature: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed: float = Field(..., description="Rotational speed in RPM")
    torque: float = Field(..., description="Torque in Nm")
    tool_wear: float = Field(..., description="Tool wear in minutes")

    class Config:
        json_schema_extra = {
            "example": {
            "type": "M",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551.0,
            "torque": 42.8,
            "tool_wear": 0.0
                }
            }
        
# ── Output schema ─────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    failure_probability: float
    prediction: str
    risk_level: str

# ── Helper: feature engineering ───────────────────────────────────────────────
def prepare_features(reading: SensorReadings) -> pd.DataFrame:
    type_encoded = le.transform([reading.type.upper()])[0]
    temp_diff = reading.process_temperature - reading.air_temperature

    features = pd.DataFrame([{
        'Type': type_encoded,
        'Air temperature [K]': reading.air_temperature,
        'Process temperature [K]': reading.process_temperature,
        'Rotational speed [rpm]': reading.rotational_speed,
        'Torque [Nm]': reading.torque,
        'Tool wear [min]': reading.tool_wear,
        'temp_diff': temp_diff
    }])

    # Match training column names
    features.columns = features.columns.str.replace('[', '', regex=False).str.replace(']', '', regex=False)

    return features

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReadings):
    # Validate machine type
    valid_types = ['L', 'M', 'H']
    if reading.type.upper() not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid machine type '{reading.type}'. Must be one of {valid_types}"
        )

    try:
        features = prepare_features(reading)
        prob = float(model.predict_proba(features)[0][1])
        prediction = "FAILURE" if prob >= 0.3 else "NO FAILURE"

        if prob >= 0.7:
            risk_level = "HIGH"
        elif prob >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return PredictionResponse(
            failure_probability=round(prob, 4),
            prediction=prediction,
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")