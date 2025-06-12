import os
import cv2
import numpy as np
import time
import logging
import json
from typing import Dict, Optional, List
from deepface import DeepFace
from core.utils.config import Config
from core.utils.exceptions import SystemInitializationError, FaceRecognitionError
from core.utils.encoding_cache import EncodingCache
from core.models.image_processor import ImagePreprocessor
from core.models.liveness_detection import LivenessDetector
from data.structures import RecognitionResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    """Enhanced face recognition system with multiple encodings and improved accuracy."""

    def __init__(self):
        try:
            self.encoding_cache = EncodingCache()
            self.image_preprocessor = ImagePreprocessor()
            self.liveness_detector = LivenessDetector()
            self.stored_images = self._load_stored_images()
            self._cache_stored_images()
            self.multiple_encodings = {}
            self.encoding_weights = {}
            self.student_thresholds = {}
            self._load_multiple_encodings()
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

    def _load_multiple_encodings(self):
        """Load multiple encodings and thresholds per student"""
        multiple_encodings_file = os.path.join(Config.STORED_IMAGES_DIR, "multiple_encodings.json")
        if os.path.exists(multiple_encodings_file):
            try:
                with open(multiple_encodings_file, 'r') as f:
                    data = json.load(f)
                    self.multiple_encodings = data.get('encodings', {})
                    self.encoding_weights = data.get('weights', {})
                    self.student_thresholds = data.get('thresholds', {})
            except Exception as e:
                logging.error(f"Error loading multiple encodings: {e}")

    def _save_multiple_encodings(self):
        """Save multiple encodings and thresholds to file"""
        try:
            data = {
                'encodings': self.multiple_encodings,
                'weights': self.encoding_weights,
                'thresholds': self.student_thresholds
            }
            multiple_encodings_file = os.path.join(Config.STORED_IMAGES_DIR, "multiple_encodings.json")
            with open(multiple_encodings_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f"Error saving multiple encodings: {e}")

    def _assess_image_quality(self, img: np.ndarray) -> float:
        """Assess image quality for face recognition"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate multiple quality metrics
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            brightness = np.mean(gray)
            contrast = gray.std()
            
            # Normalize metrics
            sharpness_score = min(sharpness / 500.0, 1.0)  # Adjusted threshold
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            contrast_score = min(contrast / 40.0, 1.0)  # Adjusted threshold
            
            # Weighted combination
            quality = (sharpness_score * 0.5 + 
                    brightness_score * 0.3 + 
                    contrast_score * 0.2)
            
            # Ensure minimum quality threshold
            return max(quality, 0.4)
        except Exception as e:
            logger.error(f"Error assessing image quality: {e}")
            return 0.5

    def _calculate_encoding_confidence(self, embedding: List[float]) -> float:
        """Calculate confidence score for an encoding"""
        try:
            # Convert to numpy array if not already
            emb_array = np.array(embedding)
            
            # Calculate statistical measures
            variance = np.var(emb_array)
            mean_abs = np.mean(np.abs(emb_array))
            
            # Combine multiple metrics for better confidence
            confidence = (variance * 100 + mean_abs) / 2
            
            # Normalize to 0-1 range with a higher minimum
            normalized_confidence = min(max(confidence, 0.3), 1.0)
            
            logger.debug(f"Calculated confidence: {normalized_confidence}")
            return normalized_confidence
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)

    def _get_dynamic_threshold(self, student_id: str) -> float:
        """Get dynamic threshold for student based on historical performance"""
        base_threshold = Config.FACE_RECOGNITION_THRESHOLD
        if student_id in self.student_thresholds:
            return self.student_thresholds[student_id]
        return base_threshold

    def _update_student_threshold(self, student_id: str, similarity: float, verified: bool):
        """Update dynamic threshold based on verification results"""
        if student_id not in self.student_thresholds:
            self.student_thresholds[student_id] = Config.FACE_RECOGNITION_THRESHOLD
        
        current_threshold = self.student_thresholds[student_id]
        if verified and similarity > current_threshold + 0.1:
            self.student_thresholds[student_id] = min(current_threshold + 0.02, 0.9)
        elif not verified and similarity < current_threshold:
            self.student_thresholds[student_id] = max(current_threshold - 0.01, 0.3)

    def _add_multiple_encoding(self, student_id: str, encoding: List[float], quality: float):
        """Add encoding to multiple encodings storage"""
        if student_id not in self.multiple_encodings:
            self.multiple_encodings[student_id] = []
            self.encoding_weights[student_id] = []
        
        if len(self.multiple_encodings[student_id]) >= 5:
            min_idx = np.argmin(self.encoding_weights[student_id])
            self.multiple_encodings[student_id].pop(min_idx)
            self.encoding_weights[student_id].pop(min_idx)
        
        self.multiple_encodings[student_id].append(encoding)
        self.encoding_weights[student_id].append(quality)
        self._save_multiple_encodings()

    def get_face_encoding_for_storage(self, img: np.ndarray, student_id: str = None) -> Dict:
        try:
            logger.info("Starting face encoding generation")
            
            # Preprocess image
            preprocessed = self.image_preprocessor.preprocess_image(img)
            if preprocessed is None:
                return {
                    "success": False,
                    "message": "Preprocessing failed",
                    "encoding": None
                }
                
            # Calculate quality score
            quality_score = self._assess_image_quality(preprocessed)
            logger.info(f"Image quality score: {quality_score}")
            
            # Save temporary image
            temp_path = f"temp_encoding_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))
            
            try:
                # Try multiple models
                best_encoding = None
                best_confidence = 0
                
                for model in ["VGG-Face", "Facenet"]:
                    try:
                        encoding = DeepFace.represent(
                            img_path=temp_path,
                            model_name=model,
                            enforce_detection=True
                        )
                        
                        if encoding:
                            confidence = self._calculate_encoding_confidence(encoding[0]["embedding"])
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_encoding = encoding[0]["embedding"]
                                
                    except Exception as model_error:
                        logger.warning(f"Model {model} failed: {str(model_error)}")
                        continue
                
                if best_encoding:
                    return {
                        "success": True,
                        "encoding": best_encoding,
                        "confidence": best_confidence,
                        "quality_score": quality_score,
                        "message": "Face registered successfully"
                    }
                
                return {
                    "success": False,
                    "message": "Failed to generate encoding",
                    "encoding": None
                }
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            logger.error(f"Error in face encoding: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "encoding": None
            }

    def verify_student(self, student_id: str, captured_image: np.ndarray) -> RecognitionResult:
        """Enhanced verification with multiple encodings and liveness"""
        start_time = time.time()
        self.metrics["attempts"] += 1

        try:
            logger.info(f"Starting verification for student ID: {student_id}")
            
            # First check if student exists
            from models import Student
            student = Student.query.get(student_id)
            if not student:
                logger.error(f"No student found with ID: {student_id}")
                return RecognitionResult(
                    success=False,
                    error_message="Student not found in database",
                    verification_type="database"
                )
                
            # Check if student has face encoding
            if not student.face_encoding:
                logger.error(f"No face encoding found for student: {student_id}")
                return RecognitionResult(
                    success=False,
                    error_message="No face encoding registered",
                    verification_type="encoding"
                )

            logger.info(f"Found student and encoding. Proceeding with verification.")

            # Liveness check
            live_result = self.liveness_detector.analyze(captured_image)
            logger.info(f"Liveness check result: {live_result}")
            
            if not live_result.get("live", True):
                self.metrics["failures"] += 1
                return RecognitionResult(
                    success=False,
                    error_message=f"Liveness check failed: {live_result.get('message', 'Unknown')}",
                    verification_type="liveness",
                    data={"liveness_score": live_result.get("score", 0.0)}
                )

            # Get stored encoding
            stored_encoding = student.face_encoding
            if not stored_encoding:
                logger.error("No stored encoding found after database check")
                return RecognitionResult(
                    success=False,
                    error_message="No stored encoding found",
                    verification_type="storage"
                )

            # Process captured image
            preprocessed = self.image_preprocessor.preprocess_image(captured_image)
            if preprocessed is None:
                logger.error("Failed to preprocess captured image")
                return RecognitionResult(
                    success=False,
                    error_message="Failed to process captured image",
                    verification_type="preprocessing"
                )

            # Generate encoding for captured image
            temp_path = f"temp_verify_{int(time.time())}.jpg"
            try:
                cv2.imwrite(temp_path, (preprocessed * 255).astype(np.uint8))
                logger.info("Generating encoding for captured image")
                
                live_repr = DeepFace.represent(
                    img_path=temp_path,
                    model_name="VGG-Face",  # Try VGG-Face instead of Facenet
                    enforce_detection=False
                )
                
                if not live_repr:
                    logger.error("Failed to generate encoding for captured image")
                    return RecognitionResult(
                        success=False,
                        error_message="Failed to process face",
                        verification_type="encoding"
                    )

                # Calculate similarity
                captured_embedding = np.array(live_repr[0]["embedding"])
                stored_embedding = np.array(stored_encoding)
                
                similarity = self._calculate_similarity(captured_embedding, stored_embedding)
                logger.info(f"Calculated similarity: {similarity}")

                # Get threshold
                threshold = self._get_dynamic_threshold(student_id)
                logger.info(f"Using threshold: {threshold}")

                # Verify
                verified = similarity >= threshold
                logger.info(f"Verification result: {verified}")

                if verified:
                    self.metrics["successes"] += 1
                else:
                    self.metrics["failures"] += 1

                return RecognitionResult(
                    success=verified,
                    confidence_score=similarity,
                    verification_time=time.time() - start_time,
                    verification_type="face",
                    data={
                        "similarity": similarity,
                        "threshold": threshold,
                        "liveness_score": live_result.get("score", 1.0)
                    }
                )

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            self.metrics["failures"] += 1
            return RecognitionResult(
                success=False,
                error_message=str(e),
                verification_type="error"
            )

    def get_student_encoding(self, student_id: str) -> Optional[List]:
        """Get stored face encoding for a student"""
        path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
        return self.encoding_cache.get_encoding(path) if os.path.exists(path) else None

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            **self.metrics, 
            "cached": len(self.stored_images),
            "multiple_encodings": len(self.multiple_encodings),
            "dynamic_thresholds": len(self.student_thresholds)
        }

    def verify_student_images(self, stored_image: np.ndarray, captured_image: np.ndarray) -> Dict:
        """Compare two face images directly with enhanced similarity calculation"""
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
                similarities = []
                models = ["Facenet", "VGG-Face"]
                
                for model in models:
                    try:
                        stored_repr = DeepFace.represent(
                            img_path=temp_stored,
                            model_name=model,
                            enforce_detection=True
                        )
                        
                        captured_repr = DeepFace.represent(
                            img_path=temp_captured,
                            model_name=model,
                            enforce_detection=False
                        )
                        
                        if stored_repr and captured_repr:
                            a = np.array(stored_repr[0]["embedding"])
                            b = np.array(captured_repr[0]["embedding"])
                            similarity = self._calculate_similarity(a, b)
                            similarities.append(similarity)
                    except:
                        continue
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    distance = 1.0 - avg_similarity
                    
                    return {
                        "success": distance <= Config.FACE_RECOGNITION_THRESHOLD,
                        "confidence_score": avg_similarity,
                        "distance": distance,
                        "models_used": len(similarities)
                    }
                else:
                    return {
                        "success": False,
                        "confidence_score": 0.0,
                        "message": "Failed to generate representations with any model"
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

    def register_multiple_face_angles(self, student_id: str, face_images: List[np.ndarray]) -> Dict:
        """Register multiple face angles for a student"""
        try:
            if not face_images:
                return {"success": False, "message": "No images provided"}
            
            successful_encodings = 0
            quality_scores = []
            
            for i, img in enumerate(face_images[:5]):
                result = self.get_face_encoding_for_storage(img, student_id)
                if result["success"]:
                    successful_encodings += 1
                    quality_scores.append(result.get("quality_score", 0.5))
            
            if successful_encodings > 0:
                avg_quality = np.mean(quality_scores)
                return {
                    "success": True,
                    "message": f"Successfully registered {successful_encodings} encodings",
                    "encodings_count": successful_encodings,
                    "average_quality": avg_quality
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to register any face encodings"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Registration error: {str(e)}"
            }

    def get_student_verification_stats(self, student_id: str) -> Dict:
        """Get verification statistics for a specific student"""
        return {
            "has_multiple_encodings": student_id in self.multiple_encodings,
            "encoding_count": len(self.multiple_encodings.get(student_id, [])),
            "average_encoding_quality": np.mean(self.encoding_weights.get(student_id, [0.5])),
            "dynamic_threshold": self._get_dynamic_threshold(student_id),
            "default_threshold": Config.FACE_RECOGNITION_THRESHOLD
        }

    def optimize_student_threshold(self, student_id: str, verification_history: List[Dict]):
        """Optimize threshold based on verification history"""
        if len(verification_history) < 5:
            return
        
        successful_similarities = [v["similarity"] for v in verification_history if v["verified"]]
        failed_similarities = [v["similarity"] for v in verification_history if not v["verified"]]
        
        if successful_similarities and failed_similarities:
            min_success = min(successful_similarities)
            max_fail = max(failed_similarities)
            optimal_threshold = 1.0 - ((min_success + max_fail) / 2)
            optimal_threshold = max(0.3, min(0.9, optimal_threshold))
            self.student_thresholds[student_id] = optimal_threshold
            self._save_multiple_encodings()

    def cleanup_old_encodings(self, days_old: int = 30):
        """Clean up old temporary files and optimize storage"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 3600)
            
            temp_files = [f for f in os.listdir(".") if f.startswith("temp_")]
            for temp_file in temp_files:
                try:
                    if os.path.getctime(temp_file) < cutoff_time:
                        os.remove(temp_file)
                except:
                    continue
            
            for student_id in list(self.multiple_encodings.keys()):
                if len(self.multiple_encodings[student_id]) > 5:
                    weights = self.encoding_weights.get(student_id, [])
                    if weights:
                        sorted_indices = np.argsort(weights)[::-1][:5]
                        self.multiple_encodings[student_id] = [
                            self.multiple_encodings[student_id][i] for i in sorted_indices
                        ]
                        self.encoding_weights[student_id] = [
                            self.encoding_weights[student_id][i] for i in sorted_indices
                        ]
            
            self._save_multiple_encodings()
            logging.info("Cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")    