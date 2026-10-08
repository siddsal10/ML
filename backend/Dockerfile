FROM python:3.11-slim

# System deps needed by OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Classical ML only - no torch. Trained models ship in ml_models/.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/uploads /app/data/processed /app/data/demo_samples

ENV PYTHONUNBUFFERED=1 \
    UPLOAD_DIR=/app/data/uploads \
    PROCESSED_DIR=/app/data/processed \
    DATABASE_URL=sqlite:////app/data/tutor.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
