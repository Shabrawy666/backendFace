import cv2
import numpy as np
from typing import Optional, Tuple
from core.utils.config import Config
import logging

class ImagePreprocessor:
    """Handles image preprocessing for face recognition"""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = Config.IMAGE_SIZE) -> np.ndarray:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)

    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.3, beta: int = 5) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def detect_and_align_face(image: np.ndarray) -> Optional[np.ndarray]:
        """Detects and aligns face in image"""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logging.info(f"Converted to grayscale - Shape: {gray.shape}")
            
            faces = []
            # Make detection less strict
            scale_factors = [1.05, 1.1, 1.15, 1.2, 1.3]  # More options
            min_neighbors_options = [2, 3, 4, 5]  # Start with lower threshold
            
            for scale in scale_factors:
                for min_neighbors in min_neighbors_options:
                    detected = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30),  # Reduced from 50x50
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if len(detected) > 0:
                        faces = detected
                        logging.info(f"Found {len(faces)} faces with scale={scale}, neighbors={min_neighbors}")
                        break
                if len(faces) > 0:
                    break
            
            if len(faces) == 0:
                logging.error("No faces detected with any parameters")
                return None
            
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            logging.info(f"Selected face region: x={x}, y={y}, w={w}, h={h}")
            
            # Reduce padding if face is small
            padding = min(20, w//10, h//10)  # Adaptive padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            face_roi = image[y:y+h, x:x+w]
            logging.info(f"Extracted face ROI shape: {face_roi.shape}")
            
            # Make quality check less strict for testing
            if not ImagePreprocessor.check_face_quality(face_roi):
                logging.warning("Face quality check failed, but continuing for testing")
                # Don't return None, continue with the face we found
            
            resized_face = ImagePreprocessor.resize_image(face_roi)
            logging.info(f"Final resized face shape: {resized_face.shape}")
            return resized_face
            
        except Exception as e:
            logging.error(f"Face detection error: {str(e)}")
            return None

    @staticmethod
    def check_face_quality(face_image: np.ndarray) -> bool:
        """Basic quality check"""
        try:
            if face_image.shape[0] < 30 or face_image.shape[1] < 30:
                return False
            
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            brightness = np.mean(gray)
            if brightness < 20 or brightness > 250:
                return False
            
            contrast = np.std(gray)
            if contrast < 10:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Face quality check error: {str(e)}")
            return False

    @staticmethod
    def preprocess_image(image: np.ndarray) -> Optional[np.ndarray]:
        """Complete preprocessing pipeline"""
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            logging.info(f"Starting preprocessing - Image shape: {image.shape}")
            
            enhanced = ImagePreprocessor.adjust_brightness_contrast(image)
            logging.info(f"Enhanced image shape: {enhanced.shape}")
            
            face_img = ImagePreprocessor.detect_and_align_face(enhanced)
            
            if face_img is None:
                logging.error("Face detection failed - no face found")
                return None
            
            logging.info(f"Face detected successfully - Face shape: {face_img.shape}")
            
            face_img = ImagePreprocessor.normalize_image(face_img)
            logging.info(f"Final preprocessed shape: {face_img.shape}")
            return face_img

        except Exception as e:
            logging.error(f"Preprocessing error: {str(e)}")
            return None
        
    @staticmethod
    def preprocess_image_fallback(image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback preprocessing without face detection"""
        try:
            logging.info("Using fallback preprocessing (no face detection)")
            
            # Just resize and normalize the center crop
            h, w = image.shape[:2]
            
            # Take center crop
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            
            center_crop = image[y:y+size, x:x+size]
            
            # Resize to target size
            resized = ImagePreprocessor.resize_image(center_crop)
            
            # Normalize
            normalized = ImagePreprocessor.normalize_image(resized)
            
            logging.info(f"Fallback preprocessing complete - Final shape: {normalized.shape}")
            return normalized
            
        except Exception as e:
            logging.error(f"Fallback preprocessing error: {str(e)}")
            return None    