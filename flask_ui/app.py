"""
Flask UI Layer for the Frost Prediction System.

This app does NOT reimplement any ML logic. It is a thin presentation
layer that:
  1. Collects 10 days of weather history from the user (manual table or CSV upload)
  2. Forwards it as JSON to the existing FastAPI ML backend (api.py, run via uvicorn)
  3. Renders the classification + probability response back to the user

Run alongside the FastAPI backend:
    Terminal 1:  uvicorn api:app --host 0.0.0.0 --port 8000
    Terminal 2:  python flask_ui/app.py
"""

import io
from datetime import datetime, timedelta

import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Must match config.yaml -> server.port of the FastAPI backend (api.py)
FASTAPI_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{FASTAPI_BASE_URL}/predict"

HISTORY_WINDOW_DAYS = 10  # must match config.yaml -> inference.history_window_days
REQUIRED_FIELDS = ["datetime", "tmin", "tmax", "td_m", "um", "ffm"]

# Human-readable labels for the 4 classes returned by FrostClassificationWrapper
CLASS_META = {
    0: {"label": "بدون ریسک یخبندان", "level": "safe"},
    1: {"label": "یخبندان تشعشعی (آسمان صاف)", "level": "warning"},
    2: {"label": "یخبندان جابجایی (بادی)", "level": "warning"},
    3: {"label": "یخبندان خشک شدید", "level": "danger"},
}


@app.route("/")
def index():
    """Renders the input form (manual table + CSV upload tab)."""
    # Prefill 10 sequential dates ending today, purely as a UX convenience.
    today = datetime.now().date()
    default_dates = [
        (today - timedelta(days=HISTORY_WINDOW_DAYS - 1 - i)).isoformat()
        for i in range(HISTORY_WINDOW_DAYS)
    ]
    return render_template(
        "index.html",
        window=HISTORY_WINDOW_DAYS,
        default_dates=default_dates,
    )


def _rows_from_manual_form(form):
    """Builds the 10-day history list from the manual table's POST fields."""
    rows = []
    for i in range(HISTORY_WINDOW_DAYS):
        row = {
            "datetime": form.get(f"datetime_{i}", "").strip(),
            "tmin": form.get(f"tmin_{i}", "").strip(),
            "tmax": form.get(f"tmax_{i}", "").strip(),
            "td_m": form.get(f"td_m_{i}", "").strip(),
            "um": form.get(f"um_{i}", "").strip(),
            "ffm": form.get(f"ffm_{i}", "").strip(),
        }
        rows.append(row)
    return rows


def _rows_from_csv(file_storage):
    """Parses an uploaded CSV and returns the last HISTORY_WINDOW_DAYS rows."""
    raw = file_storage.read().decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(raw))

    missing_cols = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ستون‌های زیر در فایل CSV یافت نشد: {', '.join(missing_cols)}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    if len(df) < HISTORY_WINDOW_DAYS:
        raise ValueError(
            f"فایل باید حداقل {HISTORY_WINDOW_DAYS} ردیف داشته باشد "
            f"(الان {len(df)} ردیف دارد)."
        )

    df = df.tail(HISTORY_WINDOW_DAYS).copy()
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d")

    return df[REQUIRED_FIELDS].to_dict(orient="records")


def _validate_and_cast(rows):
    """Ensures every row has valid, non-empty values before hitting the API."""
    cleaned = []
    for idx, row in enumerate(rows, start=1):
        if not row.get("datetime"):
            raise ValueError(f"ردیف {idx}: تاریخ خالی است.")
        try:
            numeric = {
                "tmin": float(row["tmin"]),
                "tmax": float(row["tmax"]),
                "td_m": float(row["td_m"]),
                "um": float(row["um"]),
                "ffm": float(row["ffm"]),
            }
        except (ValueError, TypeError):
            raise ValueError(f"ردیف {idx}: یک یا چند مقدار عددی نامعتبر است.")
        cleaned.append({"datetime": row["datetime"], **numeric})
    return cleaned


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts either the manual table or a CSV upload (selected via the
    'mode' field), validates it, forwards it to the FastAPI /predict
    endpoint, and returns the JSON result for the frontend to render.
    """
    mode = request.form.get("mode", "manual")

    try:
        if mode == "csv":
            if "csv_file" not in request.files or request.files["csv_file"].filename == "":
                return jsonify({"error": "لطفاً یک فایل CSV انتخاب کنید."}), 400
            rows = _rows_from_csv(request.files["csv_file"])
        else:
            rows = _rows_from_manual_form(request.form)

        history = _validate_and_cast(rows)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json={"history": history},
            timeout=15,
        )
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": (
                "اتصال به سرور مدل برقرار نشد. مطمئن شوید api.py با دستور "
                "'uvicorn api:app --port 8000' در حال اجراست."
            )
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "درخواست به سرور مدل زمان‌بر شد (timeout)."}), 504

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return jsonify({"error": f"خطا از سرور مدل: {detail}"}), response.status_code

    result = response.json()
    result["class_meta"] = CLASS_META.get(
        result.get("final_frost_class"), {"label": "نامشخص", "level": "warning"}
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)