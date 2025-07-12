from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from .model_loader import load_model_resources

try:
    model, scaler, feature_names = load_model_resources()
except Exception as e:
    raise RuntimeError("Failed to load model resources") from e

app = FastAPI(
    title="Spotify Track Popularity Predictor",
    description="API для предсказания популярности треков на Spotify",
    version="1.0.0"
)


# Модель входных данных по признакам
class TrackFeatures(BaseModel):
    danceability: float
    energy: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float


@app.get("/")
def home():
    return {
        "message": "Добро пожаловать в Spotify Popularity Predictor!",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(features: TrackFeatures):
    try:
        # Преобразуем входные данные в список в правильном порядке
        input_list = [
            features.danceability,
            features.energy,
            features.loudness,
            features.speechiness,
            features.acousticness,
            features.instrumentalness,
            features.liveness,
            features.valence,
            features.tempo
        ]

        # Преобразуем в numpy массив
        input_array = np.array(input_list).reshape(1, -1)

        # Масштабируем
        scaled_data = scaler.transform(input_array)

        # Делаем предсказание
        prediction = model.predict(scaled_data)[0]

        return {"predicted_popularity": float(prediction)}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )