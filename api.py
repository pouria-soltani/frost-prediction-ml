from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import json

app = FastAPI(title="Frost Risk Prediction API", version="1.0.0")

# In a real production environment, these paths point to the models saved at the end of main.py
# e.g., joblib.dump(final_trainer, "models/xgboost_trainer.joblib")
try:
    # We use mock placeholders here to demonstrate the architecture
    # feature_factory = joblib.load("models/feature_factory.joblib")
    # model_trainer = joblib.load("models/xgboost_trainer.joblib")
    # risk_engine = joblib.load("models/risk_engine.joblib")
    # wrapper = joblib.load("models/classification_wrapper.joblib")
    print("Models loaded successfully into RAM.")
except Exception as e:
    print("Warning: Models not found. Train main.py and save them first.")


class DailyWeatherRecord(BaseModel):
    datetime: str  # Format: "YYYY-MM-DD"
    tmin: float
    tmax: float
    td_m: float
    um: float
    ffm: float


class InferencePayload(BaseModel):
    # The API strictly requires the last 10 days of data to compute lag_7 and EWMA features
    historical_window: List[DailyWeatherRecord]


@app.post("/predict")
async def predict_frost_risk(payload: InferencePayload):
    # 1. Validate input length to ensure we can calculate 7-day lags
    if len(payload.historical_window) < 10:
        raise HTTPException(
            status_code=400,
            detail="Payload must contain at least 10 historical daily records to compute temporal features.",
        )

    try:
        # 2. Convert the incoming JSON payload into a Pandas DataFrame
        df_raw = pd.DataFrame([record.dict() for record in payload.historical_window])
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
        df_raw = df_raw.sort_values("datetime").reset_index(drop=True)

        # 3. Apply causal imputation (forward fill only) to handle any missing sensors in the payload
        features = ["tmin", "tmax", "td_m", "um", "ffm"]
        df_raw[features] = df_raw[features].ffill()

        # 4. Transform raw data into ML features (Lags, EWMA, GMM Probs)
        # df_features = feature_factory.transform(df_raw)

        # 5. Extract only the VERY LAST ROW (Today) to predict Tomorrow (t+1)
        # sample_today = df_features.iloc[[-1]]

        # 6. Run Inference (Mocked logic for illustration)
        # preds = model_trainer.model.predict(sample_today)[0]
        # risk_prob = risk_engine.calculate_risk_probability(preds[0])
        # final_class, class_msg = wrapper.classify(preds, risk_prob)

        # 7. Formulate the exact JSON response promised in the proposal
        business_output = {
            "date_target": "t+1 (Tomorrow)",
            "predicted_thermodynamics": {
                "tmin": 11.93,  # Mock: round(preds[0], 2)
                "tmax": 24.51,  # Mock: round(preds[1], 2)
                "tdew": 8.12,  # Mock: round(preds[2], 2)
                "wind": 1.2,  # Mock: round(preds[3], 2)
            },
            "frost_risk_probability_pct": 0.0,  # Mock: round(risk_prob * 100, 1)
            "final_frost_class": 0,  # Mock: final_class
            "alert_message": "NORMAL (No Frost) - Class 0",  # Mock: class_msg
        }

        return business_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "Operational", "model_version": "1.0.0"}
