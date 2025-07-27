# Use pre-built TensorFlow image (saves ~3GB download)
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /app

# Install only the additional system dependencies we need for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install only the packages not already in the base image
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose port
EXPOSE 7860

# Run your app
CMD ["python", "app.py"]