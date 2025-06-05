import os
import cv2
import numpy as np
import time
import logging
from typing import Dict, Optional, List
from deepface import DeepFace
import io
import base64
from PIL import Image

def base64_to_image(base64_str):
    """Convert base64 string to numpy array."""
    header, encoded = base64_str.split(',', 1) if ',' in base64_str else ('', base64_str)
    image_data = base64.b64decode(encoded)
    return np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))

from core.utils.config import Config
from core.utils.exceptions import SystemInitializationError, FaceRecognitionError
from core.utils.encoding_cache import EncodingCache
from core.ml.preprocessing import ImagePreprocessor
from core.ml.liveness import LivenessDetector
from data.structures import RecognitionResult

class FaceRecognitionSystem:
    """Handles student face registration & verification with lightweight liveness."""

    def __init__(self):
        try:
            self.encoding_cache = EncodingCache()
            self.image_preprocessor = ImagePreprocessor()
            self.liveness_detector = LivenessDetector()
            self.stored_images = self._load_stored_images()
            self._cache_stored_images()

            # Performance monitoring
            self.metrics = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "avg_time": 0.0
            }
        except Exception as e:
            logging.error(f"Init failed: {e}")
            raise SystemInitializationError(str(e))

    def _load_stored_images(self) -> List[str]:
        """Load stored face images"""
        os.makedirs(Config.STORED_IMAGES_DIR, exist_ok=True)
        return [
            os.path.join(Config.STORED_IMAGES_DIR, f)
            for f in os.listdir(Config.STORED_IMAGES_DIR)
            if f.lower().endswith(".jpg")
        ]

    def _cache_stored_images(self):
        """Pre-cache encodings for stored images"""
        for path in self.stored_images:
            self.encoding_cache.get_encoding(path)

    def get_face_encoding_for_storage(self, img: np.ndarray) -> Dict:
        """Generate face encoding for registration"""
        try:
            print(f"Input image shape: {img.shape}")
            print(f"Input image dtype: {img.dtype}")
            preprocessed = self.image_preprocessor.preprocess_image(img)
            if preprocessed is None:
                print("Preprocessing returned None")
                return {
                    "success": False,
                    "message": "Preprocessing failed",
                    "encoding": None
                }
            print(f"Preprocessed image shape: {preprocessed.shape}")
            print(f"Preprocessed image dtype: {preprocessed.dtype}")
            temp_path = f"temp_preprocessed_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))
            print(f"Saved temporary image to: {temp_path}")
            try:
                encoding = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=True
                )
                if encoding:
                    emb = encoding[0]["embedding"]
                    if isinstance(emb, np.ndarray):
                        emb = emb.tolist()
                    return {
                        "success": True,
                        "encoding": emb,
                        "message": "OK"
                    }
            except Exception as deep_face_error:
                print(f"DeepFace error: {str(deep_face_error)}")
                return {
                    "success": False,
                    "message": f"DeepFace processing failed: {str(deep_face_error)}",
                    "encoding": None
                }
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            return {
                "success": False,
                "message": "Encoding failed",
                "encoding": None
            }
        except Exception as e:
            print(f"General error in face encoding: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "encoding": None
            }

    def verify_student(self, student_id: str, captured_image_or_base64) -> RecognitionResult:
        """
        Verify a student’s identity by comparing the captured face to the stored encoding.
        Liveness detection is DISABLED / BYPASSED in this version.
        Accepts either np.ndarray or base64 string for the image.
        """
        start_time = time.time()
        self.metrics["attempts"] += 1

        # Accept image as base64 string or np.ndarray
        if isinstance(captured_image_or_base64, str):
            captured_image = base64_to_image(captured_image_or_base64)
        else:
            captured_image = captured_image_or_base64

        try:
            print(f"[Verify] Student ID  : {student_id}")
            print(f"[Verify] Image shape: {captured_image.shape}")

            live_result = {"live": True, "score": 1.0, "message": "Liveness check bypassed"}
            print(f"[Verify] Liveness bypass: {live_result}")

            stored_repr = self.get_student_encoding(student_id)
            if stored_repr is None:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message="No stored profile found",
                    verification_type="storage"
                )

            preprocessed = self.image_preprocessor.preprocess_image(captured_image)
            if preprocessed is None:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message="Failed to preprocess image",
                    verification_type="preprocessing"
                )

            temp_path = f"temp_verify_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))
            try:
                live_repr = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    enforce_detection=True
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            if not live_repr:
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message="Failed to generate encoding",
                    verification_type="encoding"
                )

            a = np.array(live_repr[0]["embedding"])
            b = np.array(stored_repr[0]["embedding"])
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            distance = 1.0 - similarity
            verified = distance <= Config.FACE_RECOGNITION_THRESHOLD

            elapsed = time.time() - start_time
            if verified:
                self.metrics["successes"] += 1
                prev = self.metrics["successes"] - 1
                self.metrics["avg_time"] = ((self.metrics["avg_time"] * prev) + elapsed) / self.metrics["successes"]
            else:
                self.metrics["failures"] += 1

            return RecognitionResult(
                success=verified,
                confidence_score=similarity,
                verification_time=elapsed,
                verification_type="face",
                data={"distance": distance}
            )

        except Exception as e:
            logging.error(f"Verification error: {e}")
            self.metrics["failures"] += 1
            return RecognitionResult(
                success=False,
                error_message=str(e),
                verification_type="error"
            )

    def get_student_encoding(self, student_id: str) -> Optional[list]:
        """Get stored face encoding for a student from DB field if available, else from image file."""
        try:
            from models import Student
            student = Student.query.filter_by(student_id=student_id).first()
            if student and student.face_encoding:
                emb = np.array([float(x) for x in student.face_encoding.split(',')])
                return [{"embedding": emb}]
        except Exception as e:
            print(f"DB encoding lookup failed: {e}")

        path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
        if os.path.exists(path):
            return self.encoding_cache.get_encoding(path)
        return None

    def get_performance_metrics(self) -> Dict:
        return {**self.metrics, "cached": len(self.stored_images)}

    def verify_student_images(self, stored_image: np.ndarray, captured_image: np.ndarray) -> Dict:
        try:
            stored_processed = self.image_preprocessor.preprocess_image(stored_image)
            captured_processed = self.image_preprocessor.preprocess_image(captured_image)
            if stored_processed is None or captured_processed is None:
                return {
                    "success": False,
                    "confidence_score": 0.0,
                    "message": "Failed to preprocess one or both images"
                }
            temp_stored = f"temp_stored_{int(time.time())}.jpg"
            temp_captured = f"temp_captured_{int(time.time())}.jpg"
            cv2.imwrite(temp_stored, (stored_processed * 255).astype(np.uint8))
            cv2.imwrite(temp_captured, (captured_processed * 255).astype(np.uint8))
            try:
                stored_repr = DeepFace.represent(
                    img_path=temp_stored,
                    model_name="Facenet",
                    enforce_detection=True
                )
                captured_repr = DeepFace.represent(
                    img_path=temp_captured,
                    model_name="Facenet",
                    enforce_detection=True
                )
                if stored_repr and captured_repr:
                    a = np.array(stored_repr[0]["embedding"])
                    b = np.array(captured_repr[0]["embedding"])
                    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                    distance = 1.0 - similarity
                    return {
                        "success": distance <= Config.FACE_RECOGNITION_THRESHOLD,
                        "confidence_score": similarity,
                        "distance": distance
                    }
                else:
                    return {
                        "success": False,
                        "confidence_score": 0.0,
                        "message": "Failed to generate representations"
                    }
            finally:
                for temp_file in [temp_stored, temp_captured]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        except Exception as e:
            logging.error(f"Error comparing images: {str(e)}")
            return {
                "success": False,
                "confidence_score": 0.0,
                "message": str(e)
            }