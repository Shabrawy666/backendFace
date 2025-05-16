import os

class Config:
    # Update the database URL to use pg8000 dialect
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:NraeRTIAGwBMQoAJXbzJhmqKtSwVxYCQ@centerbeam.proxy.rlwy.net:52150/railway')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL.replace('postgres://', 'postgresql+pg8000://') if DATABASE_URL.startswith('postgres://') else DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ML Configuration
    # Face Recognition Settings
    FACE_DETECTION_CONFIDENCE = float(os.getenv('FACE_DETECTION_CONFIDENCE', '0.9'))
    FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_RECOGNITION_THRESHOLD', '0.6'))
    IMAGE_SIZE = (int(os.getenv('IMAGE_SIZE_WIDTH', '224')), int(os.getenv('IMAGE_SIZE_HEIGHT', '224')))
    
    # Image Storage Paths
    TEMP_IMAGE_DIR = os.getenv('TEMP_IMAGE_DIR', "temp_images/")
    STORED_IMAGES_DIR = os.getenv('STORED_IMAGES_DIR', "stored_images/")
    FACES_DIR = os.getenv('FACES_DIR', "faces/")  # You were referencing this in ml_service.py
    
    # Logging Settings
    LOG_FILE = os.getenv('LOG_FILE', "facial_recognition.log")
    LOG_LEVEL = os.getenv('LOG_LEVEL', "INFO")