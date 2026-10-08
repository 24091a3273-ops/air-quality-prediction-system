from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__)

# ============================================
# PROJECT PATHS
# ============================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "aqi_random_forest.pkl"
)


# ============================================
# LOAD MACHINE LEARNING MODEL
# ============================================

model = joblib.load(MODEL_PATH)


# ============================================
# AQI CATEGORY
# ============================================

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


# ============================================
# HOME PAGE
# ============================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================
# FRONTEND FILES
# ============================================

@app.route("/<path:filename>")
def frontend_files(filename):

    return send_from_directory(
        FRONTEND_DIR,
        filename
    )


# ============================================
# AQI PREDICTION API
# ============================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON data from frontend

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No sensor data received."
            }), 400


        # ========================================
        # READ SENSOR VALUES
        # ========================================

        CO = float(data["CO"])

        NOx = float(data["NOx"])

        NO2 = float(data["NO2"])

        C6H6 = float(data["C6H6"])

        temperature = float(
            data["temperature"]
        )

        humidity = float(
            data["humidity"]
        )

        AH = float(
            data["AH"]
        )


        # ========================================
        # VALIDATE VALUES
        # ========================================

        values = [
            CO,
            NOx,
            NO2,
            C6H6,
            temperature,
            humidity,
            AH
        ]


        for value in values:

            if not np.isfinite(value):

                return jsonify({
                    "error":
                    "Invalid sensor value received."
                }), 400


        # ========================================
        # PREPARE ML INPUT
        # ========================================

        features = np.array([[
            CO,
            NOx,
            NO2,
            C6H6,
            temperature,
            humidity,
            AH
        ]])


        # ========================================
        # MACHINE LEARNING PREDICTION
        # ========================================

        prediction = model.predict(features)[0]


        # Keep AQI between 0 and 500

        prediction = max(
            0,
            min(500, prediction)
        )


        prediction = round(
            float(prediction),
            2
        )


        # ========================================
        # GET AQI CATEGORY
        # ========================================

        category = get_aqi_category(
            prediction
        )


        # ========================================
        # SEND RESULT TO FRONTEND
        # ========================================

        return jsonify({

            "AQI": prediction,

            "category": category,

            "status": "success"

        })


    except KeyError as error:

        return jsonify({

            "error":
            f"Missing sensor value: {error.args[0]}"

        }), 400


    except ValueError:

        return jsonify({

            "error":
            "Sensor values must be numeric."

        }), 400


    except Exception as error:

        print("Prediction error:", error)

        return jsonify({

            "error":
            "Unable to process AQI prediction."

        }), 500


# ============================================
# RUN FLASK SERVER
# ============================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )