
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file")


# ==========================================
# 2. CONNECT TO MONGODB ATLAS
# ==========================================

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=10000
)

db = client["Chennai_House_Price"]

predictions_collection = db["prediction_history"]


# ==========================================
# 3. TEST DATABASE CONNECTION
# ==========================================

def test_connection():

    try:
        client.admin.command("ping")

        print("MongoDB connected successfully!")
        print("Database: Chennai_House_Price")
        print("Collection: prediction_history")

        return True

    except Exception as error:

        print("MongoDB connection failed:", error)

        return False


# ==========================================
# 4. SAVE HOUSE PRICE PREDICTION
# ==========================================

def save_prediction(house_data, predicted_price):

    record = house_data.copy()

    record["predicted_price"] = float(predicted_price)

    record["created_at"] = datetime.now(timezone.utc)

    result = predictions_collection.insert_one(record)

    return str(result.inserted_id)


# ==========================================
# 5. RETRIEVE PREDICTION HISTORY
# ==========================================

def get_prediction_history(limit=50):

    records = predictions_collection.find(
        {},
        {"_id": 0}
    ).sort(
        "created_at", -1
    ).limit(limit)

    return list(records)


# ==========================================
# 6. GET DASHBOARD STATISTICS
# ==========================================

def get_dashboard_stats():

    # --------------------------------------
    # A. Total Predictions
    # --------------------------------------

    total_predictions = predictions_collection.count_documents({})


    # --------------------------------------
    # B. Average, Highest and Lowest Price
    # --------------------------------------

    summary_pipeline = [

        {
            "$group": {
                "_id": None,

                "average_price": {
                    "$avg": "$predicted_price"
                },

                "highest_price": {
                    "$max": "$predicted_price"
                },

                "lowest_price": {
                    "$min": "$predicted_price"
                }
            }
        }

    ]

    summary_result = list(
        predictions_collection.aggregate(summary_pipeline)
    )

    if summary_result:

        average_price = float(
            summary_result[0]["average_price"] or 0
        )

        highest_price = float(
            summary_result[0]["highest_price"] or 0
        )

        lowest_price = float(
            summary_result[0]["lowest_price"] or 0
        )

    else:

        average_price = 0
        highest_price = 0
        lowest_price = 0


    # --------------------------------------
    # C. Locality-wise Prediction Count
    # --------------------------------------

    locality_pipeline = [

        {
            "$group": {
                "_id": "$AREA",

                "count": {
                    "$sum": 1
                }
            }
        },

        {
            "$sort": {
                "count": -1
            }
        }

    ]

    locality_summary = list(
        predictions_collection.aggregate(locality_pipeline)
    )


    # --------------------------------------
    # D. Property Type Distribution
    # --------------------------------------

    property_type_pipeline = [

        {
            "$group": {
                "_id": "$BUILDTYPE",

                "count": {
                    "$sum": 1
                }
            }
        },

        {
            "$sort": {
                "count": -1
            }
        }

    ]

    property_type_summary = list(
        predictions_collection.aggregate(property_type_pipeline)
    )


    # --------------------------------------
    # E. Average Predicted Price by Locality
    # --------------------------------------

    locality_price_pipeline = [

        {
            "$group": {
                "_id": "$AREA",

                "average_price": {
                    "$avg": "$predicted_price"
                }
            }
        },

        {
            "$sort": {
                "average_price": 1
            }
        }

    ]

    locality_price_summary = list(
        predictions_collection.aggregate(locality_price_pipeline)
    )


    # --------------------------------------
    # F. RETURN ALL DASHBOARD DATA
    # --------------------------------------

    return {

        "total_predictions": total_predictions,

        "average_price": average_price,

        "highest_price": highest_price,

        "lowest_price": lowest_price,

        "locality_summary": locality_summary,

        "property_type_summary": property_type_summary,

        "locality_price_summary": locality_price_summary

    }


# ==========================================
# 7. TEST DATABASE CONNECTION
# ==========================================

if __name__ == "__main__":

    test_connection()

    print("\nDashboard Statistics:")

    dashboard_data = get_dashboard_stats()

    for key, value in dashboard_data.items():

        print(f"\n{key}:")

        print(value)
