# Apartment ML API - Render Deployment Guide

## 🚀 Deployment Steps

### 1. Prerequisites
- A [Render account](https://render.com)
- Your code pushed to GitHub
- Pre-trained models (`models/apartment_knn_model.pkl` and `models/apartment_kmeans_model.pkl`)

### 2. Deploy to Render

#### Option A: Using render.yaml (Recommended)
1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Blueprint"
4. Connect your GitHub repository
5. Render will automatically detect `render.yaml` and configure the service

#### Option B: Manual Setup
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `apartment-ml-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (or paid for production)

### 3. Set Environment Variables
In Render dashboard, add these environment variables:

```
MONGODB_URI=mongodb+srv://your_username:your_password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=test
MONGODB_COLLECTION=properties
```

⚠️ **IMPORTANT**: Replace with your actual MongoDB credentials!

### 4. Deploy
- Render will automatically deploy when you push to your main branch
- First deployment takes 5-10 minutes

## 📡 API Endpoints

Once deployed, your API will be available at: `https://apartment-ml-api.onrender.com`

### Predict with KNN
```bash
POST /predict_knn
Content-Type: application/json

{
  "price": 10000,
  "latitude": 13.6218,
  "longitude": 123.1948
}
```

### Predict with KMeans
```bash
POST /predict_kmeans
Content-Type: application/json

{
  "price": 10000,
  "latitude": 13.6218,
  "longitude": 123.1948
}
```

## 🔧 Local Testing

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run locally:
   ```bash
   python app.py
   ```

3. Test the API:
   ```bash
   curl -X POST http://localhost:5000/predict_knn \
     -H "Content-Type: application/json" \
     -d '{"price": 10000, "latitude": 13.6218, "longitude": 123.1948}'
   ```

## 📝 Notes

- Models are included in the repository (already trained)
- Free tier on Render may sleep after inactivity (first request takes longer)
- For production, consider upgrading to paid tier for better performance

## 🔒 Security
- MongoDB credentials are stored as environment variables (not in code)
- Debug mode is disabled in production
- Using Gunicorn as production WSGI server

## 📦 Files Changed
- ✅ `app.py` - Fixed PORT and host configuration
- ✅ `requirements.txt` - Added gunicorn
- ✅ `render.yaml` - Created deployment configuration
- ✅ `start.sh` - Updated to use gunicorn
- ✅ `train.py` - Using environment variables for MongoDB

## 🐛 Troubleshooting

### Build fails
- Check that all files are committed and pushed to GitHub
- Verify `requirements.txt` is present

### App crashes on startup
- Verify model files exist in `models/` directory
- Check environment variables are set correctly

### Can't connect to MongoDB
- Verify `MONGODB_URI` is correct
- Check MongoDB Atlas allows connections from all IPs (0.0.0.0/0)
