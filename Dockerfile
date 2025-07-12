# Используем образ Python
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Тут станавливаем рабочую директорию
WORKDIR /app

COPY requirements.txt .

# Тут устанавливаем Py зависимости
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app ./app
COPY random_forest_model.pkl .
COPY scaler.pkl .
COPY model_features.json .

EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]