# Use a stable Python base image
FROM python:3.10-slim

# Install system packages required by dlib, face_recognition, and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependencies first for better caching
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire app code
COPY . .

# Expose the port that the app will run on
EXPOSE 8000

# Set the command to run your Flask app using Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8000"]
