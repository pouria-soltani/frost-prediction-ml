import yaml
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn

from main import (
    FeatureFactory, 
    ForecastModelTrainer, 
    FrostRiskEngine, 
    FrostClassificationWrapper
)

# === THE FIX FOR JOBLIB PICKLE NAMESPACE CLASH ===
import __main__
__main__.FeatureFactory = FeatureFactory
__main__.ForecastModelTrainer = ForecastModelTrainer
__main__.FrostRiskEngine = FrostRiskEngine
__main__.FrostClassificationWrapper = FrostClassificationWrapper
# ==================================================

# Load Configurations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Models globally...
try:
    factory = joblib.load(config["models"]["feature_factory"])
    trainer = joblib.load(config["models"]["xgboost_trainer"])
    risk_engine = joblib.load(config["models"]["risk_engine"])
    wrapper = joblib.load(config["models"]["classification_wrapper"])
except Exception as e:
    raise RuntimeError(f"Failed to load models from disk. Error: {e}")

# Reconstruct feature column order expected by the model
base_features = [
    "tmin", "tmax", "td_m", "um", "ffm", "lapse_rate", "dtr", "dtr_lag_1",
    "tmin_lag_1", "tmin_lag_2", "tmin_lag_3", "ffm_lag_1", "td_m_roll_3",
    "tmin_volatility_3d", "sin_doy", "cos_doy", "tmin_ewma_3", "td_m_ewma_3",
    "um_ffm_interaction", "tmin_lag_5", "tmin_lag_7", "ffm_lag_5", "ffm_lag_7"
]
gmm_cols = [f"gmm_prob_{i}" for i in range(factory.n_components)]
MODEL_FEATURES = base_features + gmm_cols

# Initialize API
app = FastAPI(title="Frost Prediction API - ML Engine", version="1.0.0")

# Pydantic Schemas for Strict Data Validation
class DailyWeatherRecord(BaseModel):
    datetime: str = Field(..., description="Date format: YYYY-MM-DD")
    tmin: float
    tmax: float
    td_m: float
    um: float
    ffm: float

class PredictionRequest(BaseModel):
    history: List[DailyWeatherRecord] = Field(
        ..., 
        min_length=config["inference"]["history_window_days"],
        max_length=config["inference"]["history_window_days"],
        description=f"Requires exactly {config['inference']['history_window_days']} days of historical data for lag/ewma computation."
    )

@app.post("/predict", tags=["Inference"])
def predict_frost(request: PredictionRequest):
    try:
        # 1. Parse Input to DataFrame
        df = pd.DataFrame([record.model_dump() for record in request.history])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        
        # 2. Extract Features using Factory
        df_feat = factory.transform(df)
        
        # 3. Slice the latest day (which now contains all computed lags from history)
        X_latest = df_feat[MODEL_FEATURES].iloc[[-1]]
        
        # 4. Multi-Output Prediction: [tmin_next, tmax_next, td_m_next, ffm_next]
        preds = trainer.model.predict(X_latest)[0]
        
        # 5. Probabilistic Risk Scoring
        risk_prob = risk_engine.calculate_risk_probability(np.array([preds[0]]))[0]
        
        # 6. Business Logic Classification
        final_class, class_msg = wrapper.classify(preds, risk_prob)
        
        return {
            "date_target": "t+1 (Tomorrow)",
            "predicted_thermodynamics": {
                "tmin": float(round(preds[0], 2)),
                "tmax": float(round(preds[1], 2)),
                "tdew": float(round(preds[2], 2)),
                "wind": float(round(preds[3], 2)),
            },
            "frost_risk_probability_pct": float(round(risk_prob * 100, 1)),
            "final_frost_class": int(final_class),
            "alert_message": class_msg
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature computation: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal inference error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "api:app", 
        host=config["server"]["host"], 
        port=config["server"]["port"], 
        reload=config["server"].get("reload", False)
    )
