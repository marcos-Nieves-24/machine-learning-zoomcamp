# EVA-Rendimientos — imagen de la API de predicción
# Requiere en api/: history.csv, cultivar_map.csv, lookup.csv (src/build_history.py)
# y model.joblib + feature_config.json (copias de models/).
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py history.csv cultivar_map.csv lookup.csv model.txt feature_config.json ./

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
