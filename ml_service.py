# ml_service.py
import os
import time
import logging
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from models import db, Student

# Import components from core/ml
from core.models.image_processor import ImagePreprocessor
from core.models.liveness_detection import LivenessDetector
from core.models.face_recognition import FaceRecognitionSystem
from core.utils.config import Config
from core.utils.encoding_cache import EncodingCache
from core.utils.exceptions import SystemInitializationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=Config.LOG_FILE
)

# Create directories if they don't exist
os.makedirs(Config.TEMP_IMAGE_DIR, exist_ok=True)
os.makedirs(Config.STORED_IMAGES_DIR, exist_ok=True)

# Import DeepFace conditionally
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logging.info("DeepFace library loaded successfully")
except ImportError:
    logging.error("DeepFace library not available. Facial recognition will be limited.")
    DEEPFACE_AVAILABLE = False

class MLService:
    """Main service class that interfaces with the Flask application"""
    
    def __init__(self):
        self.recognizer = FaceRecognitionSystem()
        self.preprocessor = self.recognizer.image_preprocessor
        self.liveness = self.recognizer.liveness_detector
        self.threshold = Config.FACE_RECOGNITION_THRESHOLD
        self.deepface_available = DEEPFACE_AVAILABLE
        logging.info("ML Service initialized successfully")
    
    def get_face_encoding(self, image: np.ndarray, skip_preprocessing: bool = False) -> dict:
        """Get face encoding with option to skip preprocessing"""
        # For already preprocessed images, skip internal preprocessing
        return self.recognizer.get_face_encoding_for_storage(image, skip_preprocessing=skip_preprocessing)
    
    def verify_student_identity(self, student_id: str, image: np.ndarray) -> dict:
        """Verify student identity using FaceRecognitionSystem"""
        return self.recognizer.verify_student(student_id, image)
    
    def check_liveness(self, image: np.ndarray) -> Dict:
        """Perform liveness detection using LivenessDetector"""
        return self.liveness.analyze(image)
    
    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image using ImagePreprocessor"""
        return self.preprocessor.preprocess_image(image)
    
    def verify_student_images(self, image1: np.ndarray, image2: np.ndarray) -> dict:
        """Compare two face images using FaceRecognitionSystem"""
        return self.recognizer.verify_student_images(image1, image2)
    
    def get_student_encoding(self, student_id: str) -> Optional[list]:
        """Get stored encoding for a student"""
        return self.recognizer.get_student_encoding(student_id)
    
    def get_performance_metrics(self) -> Dict:
        """Get system performance metrics"""
        return self.recognizer.get_performance_metrics()

    # Additional utility methods
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image using ImagePreprocessor"""
        return self.preprocessor.adjust_brightness_contrast(image)
    
    def detect_and_align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and align face using ImagePreprocessor"""
        return self.preprocessor.detect_and_align_face(image)
    
    def check_face_quality(self, image: np.ndarray) -> bool:
        """Check face image quality using ImagePreprocessor"""
        return self.preprocessor.check_face_quality(image)

# Create a singleton instance
ml_service = MLService()