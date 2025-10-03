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

knn = joblib.load(knn_model_path)
kmeans = joblib.load(kmeans_model_path)

# Optional: Map KMeans clusters to labels
cluster_to_category = {0: "Low Budget", 1: "Mid Range", 2: "High End"}

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
    app.run(debug=True, port=5000)
