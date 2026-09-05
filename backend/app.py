from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__)

# Project folders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "model", "aqi_random_forest.pkl")

# Load ML model
model = joblib.load(MODEL_PATH)


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


# Show frontend
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# Show CSS and other frontend files
@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# AQI prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    CO = float(data["CO"])
    NOx = float(data["NOx"])
    NO2 = float(data["NO2"])
    C6H6 = float(data["C6H6"])
    temperature = float(data["temperature"])
    humidity = float(data["humidity"])
    AH = float(data["AH"])

    features = np.array([[
        CO,
        NOx,
        NO2,
        C6H6,
        temperature,
        humidity,
        AH
    ]])

    prediction = model.predict(features)[0]

    prediction = max(0, min(500, prediction))

    category = get_aqi_category(prediction)

    return jsonify({
        "AQI": round(float(prediction), 2),
        "category": category
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        debug=False
    )