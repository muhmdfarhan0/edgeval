FROM python:3.12-slim

WORKDIR /app

# libgl1/libglib2.0-0: required by opencv-python-headless for image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch/torchvision wheels are far smaller than the default CUDA
# build and are all this service needs for inference.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV DATA_DIR=/app/data
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
