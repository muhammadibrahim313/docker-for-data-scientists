# app.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris Classifier API")

# Load model on startup
artifact = joblib.load("model.pkl")
model = artifact["model"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"status": "ok", "message": "Iris Classifier API is running v2 test"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()
    
    return {
        "prediction": int(prediction),
        "species": target_names[prediction],
        "confidence": round(float(probability), 4)
    }