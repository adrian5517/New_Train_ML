# app.py
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

app = Flask(__name__)

# -------------------------------
# Load trained models
# -------------------------------
knn_model_path = os.path.join("models", "apartment_knn_model.pkl")
kmeans_model_path = os.path.join("models", "apartment_kmeans_model.pkl")
thresholds_path = os.path.join("models", "category_thresholds.pkl")

knn = joblib.load(knn_model_path)
kmeans = joblib.load(kmeans_model_path)

# Load category thresholds
try:
    thresholds = joblib.load(thresholds_path)
    percentile_33 = thresholds['percentile_33']
    percentile_67 = thresholds['percentile_67']
    print(f"✅ Category thresholds loaded:")
    print(f"   Low Budget: ≤ ₱{percentile_33:,.0f}")
    print(f"   Mid Range: ₱{percentile_33:,.0f} - ₱{percentile_67:,.0f}")
    print(f"   High End: > ₱{percentile_67:,.0f}")
except FileNotFoundError:
    # Fallback to default values if thresholds file doesn't exist
    print("⚠️ Thresholds file not found, using default values")
    percentile_33 = 7000
    percentile_67 = 14000
    thresholds = {
        'percentile_33': percentile_33,
        'percentile_67': percentile_67,
        'min_price': 2000,
        'max_price': 20000,
        'median_price': 10000
    }

# Optional: Map KMeans clusters to labels
cluster_to_category = {0: "Low Budget", 1: "Mid Range", 2: "High End"}

# -------------------------------
# Endpoint: / (Health Check)
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "Apartment ML API is running",
        "endpoints": {
            "/predict_knn": "POST - Predict category using KNN",
            "/predict_kmeans": "POST - Predict cluster using KMeans"
        },
        "categories": {
            "Low Budget": f"≤ ₱{percentile_33:,.0f}",
            "Mid Range": f"₱{percentile_33:,.0f} - ₱{percentile_67:,.0f}",
            "High End": f"> ₱{percentile_67:,.0f}"
        },
        "price_range": {
            "min": f"₱{thresholds['min_price']:,.0f}",
            "max": f"₱{thresholds['max_price']:,.0f}",
            "median": f"₱{thresholds['median_price']:,.0f}"
        },
        "version": "2.0.0"
    })

# -------------------------------
# Endpoint: /predict_knn
# -------------------------------
@app.route("/predict_knn", methods=["POST"])
def predict_knn():
    data = request.get_json()
    try:
        price = float(data.get("price"))
        latitude = float(data.get("latitude"))
        longitude = float(data.get("longitude"))
    except Exception as e:
        return jsonify({"error": "Invalid input data", "details": str(e)}), 400

    df_input = pd.DataFrame([[price, latitude, longitude]], columns=['Price','Latitude','Longitude'])
    prediction = knn.predict(df_input)[0]

    return jsonify({"category": prediction})

# -------------------------------
# Endpoint: /predict_kmeans
# -------------------------------
@app.route("/predict_kmeans", methods=["POST"])
def predict_kmeans():
    data = request.get_json()
    try:
        price = float(data.get("price"))
        latitude = float(data.get("latitude"))
        longitude = float(data.get("longitude"))
    except Exception as e:
        return jsonify({"error": "Invalid input data", "details": str(e)}), 400

    df_input = pd.DataFrame([[price, latitude, longitude]], columns=['Price','Latitude','Longitude'])
    cluster_id = kmeans.predict(df_input)[0]
    cluster_label = cluster_to_category.get(cluster_id, str(cluster_id))

    return jsonify({"cluster_id": int(cluster_id), "cluster_label": cluster_label})

# -------------------------------
# Run Flask
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
