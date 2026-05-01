# 1. Use Python 3.9 as requested
FROM python:3.9-slim-buster

# 2. Install system dependencies for OpenCV, MLflow, and AWS/Gdown
RUN apt-get update -y && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup working directory
WORKDIR /app

# 4. Layer Caching: Install heavy requirements first
# This prevents re-installing TensorFlow every time you change a line of code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application
COPY . /app

# 6. Handle the "-e ." (installing your local package)
# If you have a setup.py, this ensures your internal modules are findable
RUN pip install -e .

# 7. Flask default port is usually 5000, but Hugging Face prefers 7860
# We will map this in the CMD below
EXPOSE 7860

# 8. Run the Flask app
# We use 0.0.0.0 so it's accessible outside the container
CMD ["python3", "app.py"]