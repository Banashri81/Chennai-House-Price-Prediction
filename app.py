
from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

from database import (
    save_prediction,
    get_prediction_history,
    get_dashboard_stats
)

app = Flask(__name__)

# Get project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained machine learning model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tuned_random_forest_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------
# HOME PAGE
# --------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------
# HOUSE PRICE PREDICTION
# --------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Collect property details from form

        house_data = pd.DataFrame([{
            "AREA": request.form["AREA"],
            "INT_SQFT": float(request.form["INT_SQFT"]),
            "DIST_MAINROAD": float(request.form["DIST_MAINROAD"]),
            "N_BEDROOM": int(request.form["N_BEDROOM"]),
            "N_BATHROOM": int(request.form["N_BATHROOM"]),
            "PARK_FACIL": request.form["PARK_FACIL"],
            "BUILDTYPE": request.form["BUILDTYPE"],
            "UTILITY_AVAIL": request.form["UTILITY_AVAIL"],
            "STREET": request.form["STREET"],
            "BUILDING_AGE": int(request.form["BUILDING_AGE"])
        }])

        # Generate prediction

        predicted_price = float(
            model.predict(house_data)[0]
        )

        # Save prediction in MongoDB

        save_prediction(
            house_data.iloc[0].to_dict(),
            predicted_price
        )

        # Display prediction result

        return render_template(
            "result.html",
            predicted_price=round(predicted_price, 2),
            house=house_data.iloc[0].to_dict()
        )

    except Exception as error:

        return (
            f"Prediction error: {str(error)}",
            400
        )


# --------------------------------
# PREDICTION HISTORY AND DASHBOARD
# --------------------------------


@app.route("/history")
def history():
    try:
        records = get_prediction_history(limit=50)

        stats = get_dashboard_stats()

        model_metrics = {
            "r2": 0.9431,
            "mae": 703431.68,
            "rmse": 864727.02,
            "mape": 7.71
        }

        return render_template(
            "history.html",
            records=records,
            stats=stats,
            model_metrics=model_metrics
        )

    except Exception as error:
        return f"Dashboard error: {str(error)}", 500

    try:
        # Get prediction records from MongoDB

        records = get_prediction_history(limit=50)

        # Get dashboard statistics

        stats = get_dashboard_stats()

        # Machine learning model evaluation results

        model_metrics = {
            "r2": 0.9431,
            "mae": 703431.68,
            "rmse": 864727.02,
            "mape": 7.71
        }

        # Send data to history.html

        return render_template(
            "history.html",
            records=records,
            stats=stats,
            model_metrics=model_metrics
        )

    except Exception as error:

        return (
            f"Dashboard error: {str(error)}",
            500
        )


# --------------------------------
# RUN FLASK APPLICATION
# --------------------------------

if __name__ == "__main__":
    app.run(debug=True)
