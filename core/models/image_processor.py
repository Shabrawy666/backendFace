import cv2
import numpy as np
from typing import Optional, Tuple
from core.utils.config import Config
import logging

# Configure logger
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for face recognition"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = Config.IMAGE_SIZE) -> np.ndarray:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)

    def adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image brightness and contrast"""
        try:
            # Ensure image is uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)

            # Merge channels
            limg = cv2.merge((cl,a,b))

            # Convert back to RGB
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            self.logger.info("Image enhancement completed")
            return enhanced

        except Exception as e:
            self.logger.error(f"Enhancement error: {str(e)}")
            return image

    def detect_and_align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detects and aligns face in image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = []
            scale_factors = [1.1, 1.2, 1.3]
            min_neighbors_options = [3, 4, 5]
            
            for scale in scale_factors:
                for min_neighbors in min_neighbors_options:
                    detected = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(50, 50),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    if len(detected) > 0:
                        faces = detected
                        break
                if len(faces) > 0:
                    break
            
            if len(faces) == 0:
                return None
            
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            
            padding = 30
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            face_roi = image[y:y+h, x:x+w]
            
            if not self.check_face_quality(face_roi):
                return None
            
            return self.resize_image(face_roi)
            
        except Exception as e:
            self.logger.error(f"Face detection error: {str(e)}")
            return None

    def check_face_quality(self, face_image: np.ndarray) -> bool:
        """Enhanced face quality check"""
        try:
            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                self.logger.error("Face image too small")
                return False

            # Convert to grayscale for quality checks
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = face_image

            # Check brightness
            brightness = np.mean(gray)
            if brightness < 40 or brightness > 250:
                self.logger.error(f"Poor brightness: {brightness}")
                return False

            # Check contrast
            contrast = np.std(gray)
            if contrast < 20:
                self.logger.error(f"Poor contrast: {contrast}")
                return False

            # Check blur
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                self.logger.error(f"Image too blurry: {laplacian_var}")
                return False

            self.logger.info(f"Face quality metrics - Brightness: {brightness}, Contrast: {contrast}, Sharpness: {laplacian_var}")
            return True

        except Exception as e:
            self.logger.error(f"Quality check error: {str(e)}")
            return False

    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Complete preprocessing pipeline"""
        try:
            # 1. Initial image validation
            if image is None or image.size == 0:
                self.logger.error("Invalid input image: Image is None or empty")
                return None

            self.logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")

            # 2. Ensure image is uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # 3. Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.logger.info("Converted image to RGB")
            else:
                self.logger.error(f"Invalid image format: shape {image.shape}")
                return None

            # 4. Resize if needed (optional)
            if image_rgb.shape[0] > 1000 or image_rgb.shape[1] > 1000:
                image_rgb = cv2.resize(image_rgb, (640, 480))
                self.logger.info(f"Resized image to: {image_rgb.shape}")

            # 5. Enhance image
            enhanced = self.adjust_brightness_contrast(image_rgb)
            self.logger.info("Enhanced image brightness and contrast")

            # 6. Detect face
            gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                self.logger.error("No face detected in image")
                return None

            self.logger.info(f"Detected {len(faces)} faces")

            # 7. Get largest face
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Add padding
            padding = 40
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(enhanced.shape[1] - x, w + 2*padding)
            h = min(enhanced.shape[0] - y, h + 2*padding)

            # Extract face ROI
            face_roi = enhanced[y:y+h, x:x+w]
            self.logger.info(f"Extracted face ROI: {face_roi.shape}")

            # 8. Check face quality
            if not self.check_face_quality(face_roi):
                self.logger.error("Face quality check failed")
                return None

            # 9. Resize to final size
            final_size = (224, 224)
            face_resized = cv2.resize(face_roi, final_size)
            
            # 10. Normalize
            face_normalized = face_resized.astype('float32') / 255.0

            self.logger.info("Preprocessing completed successfully")
            return face_normalized

        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            return None