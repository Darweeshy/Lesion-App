# back/Dockerfile
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache

# OpenCV/Pillow system libs (trixie/bookworm-safe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ⬅️ context is ./back, so files are at the root of context:
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your model + app code (no 'back/' prefix)
COPY unet_full_model.keras /app/unet_full_model.keras
COPY . /app/

# (Optional) Pre-cache HF model
RUN python - << 'PY'
from transformers import AutoImageProcessor, AutoModelForImageClassification
name = "ALM-AHME/convnextv2-large-1k-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20-V2"
AutoImageProcessor.from_pretrained(name)
AutoModelForImageClassification.from_pretrained(name)
print("HF model cached.")
PY

EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-b", "0.0.0.0:5000", "app:app"]
