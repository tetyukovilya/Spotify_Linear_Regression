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
    duration_ms: int
    key: int
    mode: int
    time_signature: int
    explicit: int

    # Пример значений по умолчанию
    class Config:
        schema_extra = {
            "example": {
                "danceability": 0.8,
                "energy": 0.7,
                "loudness": -5.0,
                "speechiness": 0.05,
                "acousticness": 0.01,
                "instrumentalness": 0.0,
                "liveness": 0.1,
                "valence": 0.9,
                "tempo": 120,
                "duration_ms": 210000,
                "key": 5,
                "mode": 1,
                "time_signature": 4,
                "explicit": 0
            }
        }


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
        # Основные признаки
        input_data = {
            'danceability': features.danceability,
            'energy': features.energy,
            'loudness': features.loudness,
            'speechiness': features.speechiness,
            'acousticness': features.acousticness,
            'instrumentalness': features.instrumentalness,
            'liveness': features.liveness,
            'valence': features.valence,
            'tempo': features.tempo,
            'duration_ms': features.duration_ms,
            'key': features.key,
            'mode': features.mode,
            'time_signature': features.time_signature,
            'explicit': features.explicit
        }

        # Вычисляем производные признаки
        input_data['energy_dance'] = features.energy * features.danceability
        input_data['acoustic_energy'] = features.acousticness * (1 - features.energy)
        input_data['energy_combo'] = features.energy * (1 - features.acousticness) * (1 - features.instrumentalness)

        # Создаем массив в правильном порядке
        feature_names = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms', 'key', 'mode', 'time_signature', 'explicit',
            'energy_combo'  # Последний признак
        ]

        input_array = np.array([input_data[col] for col in feature_names]).reshape(1, -1)

        # Масштабируем и предсказываем
        scaled_data = scaler.transform(input_array)
        prediction = model.predict(scaled_data)[0]

        return {"predicted_popularity": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")