# train.py
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from pymongo import MongoClient

# -------------------------------
# 1️⃣ Load synthetic data
# -------------------------------
synthetic_path = os.path.join("data", "synthetic_apartments_naga.csv")
try:
    df_synthetic = pd.read_csv(synthetic_path)
    print(f"✅ CSV Loaded: {synthetic_path}")
except FileNotFoundError:
    print(f"❌ File not found: {synthetic_path}")
    exit()

# Parse location
df_synthetic['location'] = df_synthetic['location'].apply(ast.literal_eval)
df_synthetic['Latitude'] = df_synthetic['location'].apply(lambda x: x['latitude'])
df_synthetic['Longitude'] = df_synthetic['location'].apply(lambda x: x['longitude'])

# Keep only necessary columns
df_synthetic = df_synthetic[['price', 'Latitude', 'Longitude']].rename(columns={'price': 'Price'})
print("✅ Synthetic data prepared")

# -------------------------------
# 2️⃣ Load real data from MongoDB
# -------------------------------
try:
    # Get MongoDB URI from environment variable
    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb+srv://ajboncodin:VM6TyKYYfZVVf4RL@rentifydb.gaifxpk.mongodb.net/?retryWrites=true&w=majority&appName=RentifyDB")
    mongodb_database = os.environ.get("MONGODB_DATABASE", "test")
    mongodb_collection = os.environ.get("MONGODB_COLLECTION", "properties")
    
    client = MongoClient(mongodb_uri)
    db = client[mongodb_database]
    collection = db[mongodb_collection]

    docs = list(collection.find({}))
    df_real = pd.DataFrame(docs)

    # Flatten location
    df_real['Latitude'] = df_real['location'].apply(lambda x: x.get('latitude') if isinstance(x, dict) else None)
    df_real['Longitude'] = df_real['location'].apply(lambda x: x.get('longitude') if isinstance(x, dict) else None)
    
    df_real = df_real.rename(columns={'price':'Price'})
    df_real = df_real[['Price', 'Latitude', 'Longitude']].dropna()

    print(f"✅ Real data loaded: {len(df_real)} records")

except Exception as e:
    print(f"⚠️ Could not load real data: {e}")
    df_real = pd.DataFrame(columns=['Price','Latitude','Longitude'])

# -------------------------------
# 3️⃣ Combine datasets
# -------------------------------
df_combined = pd.concat([df_synthetic, df_real], ignore_index=True)
print(f"✅ Combined dataset size: {len(df_combined)}")

# -------------------------------
# 4️⃣ Add category label
# -------------------------------
def categorize(price):
    if price <= 7000:
        return "Low Budget"
    elif price <= 14000:
        return "Mid Range"
    else:
        return "High End"

df_combined['category'] = df_combined['Price'].apply(categorize)

# -------------------------------
# 5️⃣ Train KNN
# -------------------------------
X = df_combined[['Price', 'Latitude', 'Longitude']]
y = df_combined['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("✅ KNN Model trained")
print("Sample predictions:", knn.predict(X_test.head()))

# -------------------------------
# 6️⃣ Save trained model
# -------------------------------
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "apartment_knn_model.pkl")
joblib.dump(knn, model_path)
print(f"✅ Model saved at {model_path}")

# -------------------------------
# 7️⃣ Train KMeans
# -------------------------------
from sklearn.cluster import KMeans

# Features para sa clustering
X_cluster = df_combined[['Price', 'Latitude', 'Longitude']]

# KMeans with 3 clusters (Low, Mid, High)
kmeans = KMeans(n_clusters=3, random_state=42)
df_combined['cluster'] = kmeans.fit_predict(X_cluster)

print("✅ KMeans clustering done")
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Sample cluster assignment:\n", df_combined[['Price','Latitude','Longitude','cluster']].head())

# Save KMeans model
kmeans_model_path = os.path.join("models", "apartment_kmeans_model.pkl")
joblib.dump(kmeans, kmeans_model_path)
print(f"✅ KMeans model saved at {kmeans_model_path}")