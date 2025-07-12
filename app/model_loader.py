import joblib
import json
import numpy as np
from pathlib import Path
import os


def load_model_resources():
    try:
        base_dir = Path.cwd()

        # Пути к файлам
        model_path = base_dir / "random_forest_model.pkl"
        scaler_path = base_dir / "scaler.pkl"
        features_path = base_dir / "model_features.json"

        # Проверка существования файлов, потому что у меня они не находились
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Features config not found: {features_path}")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        with open(features_path, 'r') as f:
            features_config = json.load(f)
            feature_names = features_config['features']

        print("Модель загружена")
        return model, scaler, feature_names

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        raise