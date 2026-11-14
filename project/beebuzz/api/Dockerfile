# Build from the project root:
#   docker build -f api/Dockerfile -t beebuzz .
FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY models /app/models
COPY api/app.py /app/app.py

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
