# ml_service.py
import os
import time
import logging
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from models import db, Student

# Import components from core/ml
from core.ml.preprocessing import ImagePreprocessor
from core.ml.liveness import LivenessDetector
from core.ml.recognition import FaceRecognitionSystem
from core.utils.config import Config  # Import Config from your utils
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
os.makedirs(Config.FACES_DIR, exist_ok=True)  # Add this directory for consistency

# Import DeepFace conditionally (to handle potential import errors)
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
        self.liveness = self.recognizer.liveness_detector
        self.preprocessor = self.recognizer.image_preprocessor
        self.threshold = Config.FACE_RECOGNITION_THRESHOLD
        self.deepface_available = DEEPFACE_AVAILABLE
        logging.info("ML Service initialized successfully")
        
    def verify_face(self, student_id: str, image: np.ndarray) -> dict:
        """Complete face verification pipeline"""
        return self.recognizer.verify_face(student_id, image)
        
    def get_face_encoding(self, image: np.ndarray) -> dict:
        """Get face encoding for storage"""
        return self.recognizer.get_face_encoding(image)
    
    # For backward compatibility with your existing code
    def get_face_encoding_for_storage(self, image: np.ndarray) -> dict:
        """Alias for get_face_encoding for backward compatibility"""
        return self.recognizer.get_face_encoding(image)
        
    def verify_student_identity(self, student_id: str, image: np.ndarray) -> dict:
        """Verify student identity with database stored encoding"""
        return self.recognizer.verify_student_identity(student_id, image)
        
    def verify_liveness(self, image: np.ndarray) -> dict:
        """Standalone liveness detection"""
        return self.recognizer.verify_liveness(image)
    
    def compare_faces(self, image1: np.ndarray, image2: np.ndarray) -> dict:
        """Compare two face images directly"""
        return self.recognizer.compare_faces(image1, image2)
    
    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess an image for face recognition"""
        return self.preprocessor.preprocess_image(image)
    
    def get_performance_metrics(self) -> Dict:
        """Get system performance metrics"""
        return self.recognizer.get_performance_metrics()


# Create a singleton instance
ml_service = MLService()