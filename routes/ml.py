from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import numpy as np
import base64
import io
import cv2
import os
import time
from PIL import Image
from ml_service import ml_service
from models import db, Student
import logging
from core.utils.config import Config
from core.models.face_recognition import FaceRecognitionSystem
from core.models.liveness_detection import LivenessDetector
from datetime import datetime

ml_bp = Blueprint('ml', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy image"""
    try:
        if not base64_str:
            raise ValueError("Empty image data")
            
        header, encoded = base64_str.split(',', 1) if ',' in base64_str else ('', base64_str)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return np.array(image)
    except Exception as e:
        logger.error(f"Image conversion error: {str(e)}")
        raise ValueError("Invalid image format")

@ml_bp.route('/register_face', methods=['POST'])
@jwt_required()
def register_face():
    """Register face for current student"""
    try:
        student_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or not data.get('image_base64'):
            return jsonify({"error": "Image data is required"}), 400
            
        image = base64_to_image(data['image_base64'])
        encoding_result = ml_service.get_face_encoding(image)
        
        if not encoding_result['success']:
            return jsonify({
                "success": False,
                "message": encoding_result['message']
            }), 400
            
        # Save to database
        student = Student.query.get(student_id)
        if not student:
            return jsonify({"error": "Student not found"}), 404
            
        student.face_encoding = encoding_result['encoding']
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Face registered successfully",
            "quality_metrics": encoding_result.get('quality_metrics', {})
        }), 200
        
    except ValueError as ve:
        return jsonify({
            "success": False,
            "message": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Face registration error: {str(e)}")
        db.session.rollback()
        return jsonify({
            "success": False,
            "message": "Face registration failed. Please try again."
        }), 500

@ml_bp.route('/verify', methods=['POST'])
@jwt_required()
def verify_face():
    """Verify student face with liveness check"""
    try:
        student_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or not data.get('image_base64'):
            return jsonify({"error": "Image data is required"}), 400
            
        image = base64_to_image(data['image_base64'])
        result = ml_service.verify_student_identity(student_id, image)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except ValueError as ve:
        return jsonify({
            "success": False,
            "message": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Face verification error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Face verification failed. Please try again."
        }), 500

@ml_bp.route('/liveness', methods=['POST'])
def check_liveness():
    """Standalone liveness check endpoint"""
    try:
        data = request.get_json()
        
        if not data or not data.get('image_base64'):
            return jsonify({"error": "Image data is required"}), 400
            
        image = base64_to_image(data['image_base64'])
        result = ml_service.verify_liveness(image)
        
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({
            "success": False,
            "message": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Liveness check error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Liveness check failed. Please try again."
        }), 500

@ml_bp.route('/compare_faces', methods=['POST'])
def compare_faces():
    """Compare two faces directly"""
    try:
        data = request.get_json()
        
        if not data or not data.get('image1_base64') or not data.get('image2_base64'):
            return jsonify({"error": "Two images are required"}), 400
            
        image1 = base64_to_image(data['image1_base64'])
        image2 = base64_to_image(data['image2_base64'])
        
        result = ml_service.compare_faces(image1, image2)
        
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({
            "success": False,
            "message": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Face comparison error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Face comparison failed. Please try again."
        }), 500

@ml_bp.route('/register_multiple_faces', methods=['POST'])
@jwt_required()
def register_multiple_faces():
    """Register multiple face angles for better recognition"""
    try:
        student_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or not data.get('images'):
            return jsonify({"error": "Multiple face images are required"}), 400
            
        face_images = []
        for image_base64 in data['images']:
            try:
                image = base64_to_image(image_base64)
                face_images.append(image)
            except ValueError as ve:
                logger.warning(f"Skipping invalid image: {str(ve)}")
                continue
                
        if not face_images:
            return jsonify({"error": "No valid images provided"}), 400
            
        result = ml_service.register_multiple_face_angles(student_id, face_images)
        
        if result['success']:
            student = Student.query.get(student_id)
            if student:
                student.face_registered_at = datetime.utcnow()
                db.session.commit()
                
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Multiple face registration error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Face registration failed",
            "error": str(e)
        }), 500

@ml_bp.route('/verify_quality', methods=['POST'])
@jwt_required()
def verify_image_quality():
    """Check image quality before processing"""
    try:
        data = request.get_json()
        
        if not data or not data.get('image_base64'):
            return jsonify({"error": "Image data is required"}), 400
            
        image = base64_to_image(data['image_base64'])
        
        # Perform quality checks
        preprocessed = ml_service.preprocess_image(image)
        if preprocessed is None:
            return jsonify({
                "success": False,
                "message": "Image quality too low",
                "details": {
                    "face_detected": False,
                    "suggestions": [
                        "Ensure face is clearly visible",
                        "Improve lighting conditions",
                        "Center face in frame"
                    ]
                }
            }), 400
            
        quality_result = ml_service.check_face_quality(preprocessed)
        
        return jsonify({
            "success": True,
            "quality_score": quality_result,
            "details": {
                "face_detected": True,
                "image_size": image.shape,
                "preprocessing_success": True
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Quality check error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Quality check failed",
            "error": str(e)
        }), 500

@ml_bp.route('/verify_with_liveness', methods=['POST'])
@jwt_required()
def verify_with_liveness():
    """Combined face verification and liveness check"""
    try:
        student_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or not data.get('image_base64'):
            return jsonify({"error": "Image data is required"}), 400
            
        image = base64_to_image(data['image_base64'])
        
        # First check liveness
        liveness_result = ml_service.check_liveness(image)
        if not liveness_result.get('live', False):
            return jsonify({
                "success": False,
                "message": "Liveness check failed",
                "liveness_details": liveness_result
            }), 400
            
        # Then verify face
        verification_result = ml_service.verify_student_identity(student_id, image)
        
        return jsonify({
            "success": verification_result['success'],
            "verification_details": verification_result,
            "liveness_details": liveness_result,
            "processing_time": {
                "total": verification_result.get('verification_time', 0),
                "liveness": liveness_result.get('processing_time', 0)
            }
        }), 200 if verification_result['success'] else 400
        
    except Exception as e:
        logger.error(f"Combined verification error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Verification failed",
            "error": str(e)
        }), 500

@ml_bp.route('/optimize_recognition', methods=['POST'])
@jwt_required()
def optimize_recognition():
    """Optimize face recognition parameters for a student"""
    try:
        student_id = get_jwt_identity()
        data = request.get_json()
        
        verification_history = data.get('verification_history', [])
        
        # Optimize threshold based on history
        ml_service.optimize_student_threshold(student_id, verification_history)
        
        # Get current stats
        stats = ml_service.get_student_verification_stats(student_id)
        
        return jsonify({
            "success": True,
            "message": "Recognition parameters optimized",
            "current_stats": stats
        }), 200
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Optimization failed",
            "error": str(e)
        }), 500

# Enhance existing health check
@ml_bp.route('/health', methods=['GET'])
def ml_health_check():
    """Enhanced ML service health check"""
    try:
        metrics = ml_service.get_performance_metrics()
        
        # Check if all components are operational
        components_status = {
            "face_recognition": ml_service.deepface_available,
            "liveness_detection": True,  # Add actual check
            "image_processor": True      # Add actual check
        }
        
        # Get system resources
        import psutil
        system_resources = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return jsonify({
            "status": "ok",
            "message": "ML service is operational",
            "components": components_status,
            "performance_metrics": metrics,
            "system_resources": system_resources,
            "config": {
                "image_size": Config.IMAGE_SIZE,
                "recognition_threshold": Config.FACE_RECOGNITION_THRESHOLD
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"ML service error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@ml_bp.route('/preprocess', methods=['POST'])
def preprocess_image():
    """Preprocess an image and return the result"""
    try:
        data = request.get_json()
        
        if not data or not data.get('image_base64'):
            return jsonify({"error": "Image data is required"}), 400
            
        image = base64_to_image(data['image_base64'])
        preprocessed = ml_service.preprocess_image(image)
        
        if preprocessed is None:
            return jsonify({
                "success": False,
                "message": "Failed to preprocess image - no face detected"
            }), 400
            
        # Convert preprocessed image back to base64
        pil_img = Image.fromarray((preprocessed * 255).astype(np.uint8))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
            
        return jsonify({
            "success": True,
            "preprocessed_image": f"data:image/jpeg;base64,{img_str}"
        }), 200
        
    except ValueError as ve:
        return jsonify({
            "success": False,
            "message": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Image preprocessing failed. Please try again."
        }), 500
    
@ml_bp.route('/test-verify-student', methods=['POST'])
@jwt_required()
def test_verify_student():
    try:
        student_id = get_jwt_identity()
        
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        file = request.files['image']
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        logger.info(f"Testing verification for student {student_id}")
        
        # Get stored student data
        student = Student.query.get(student_id)
        if not student:
            return jsonify({"error": "Student not found"}), 404
            
        if not student.face_encoding:
            return jsonify({"error": "No face encoding stored"}), 400

        # Preprocess image
        preprocessed = ml_service.preprocess_image(image)
        if preprocessed is None:
            return jsonify({
                "error": "Image preprocessing failed",
                "details": "Could not process the image properly"
            }), 400

        # Test verification with detailed logging
        try:
            result = ml_service.verify_student_identity(student_id, preprocessed)
            
            # Convert RecognitionResult to dictionary
            result_dict = result.to_dict()
            
            # Add additional debug info
            result_dict["debug_info"] = {
                "image_shape": image.shape if image is not None else None,
                "preprocessed_shape": preprocessed.shape if preprocessed is not None else None,
                "student_id": student_id,
                "has_encoding": True,
                "encoding_length": len(student.face_encoding) if student.face_encoding else 0
            }

            return jsonify(result_dict), 200
            
        except Exception as verify_error:
            logger.error(f"Verification process error: {str(verify_error)}")
            return jsonify({
                "error": "Verification process failed",
                "details": str(verify_error)
            }), 500
        
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        return jsonify({
            "error": "Test endpoint failed",
            "details": str(e)
        }), 500

@ml_bp.route('/check-storage', methods=['GET'])
@jwt_required()
def check_storage():
    try:
        student_id = get_jwt_identity()
        
        # Check if student exists
        student = Student.query.get(student_id)
        if not student:
            return jsonify({"error": "Student not found"}), 404

        # Check database encoding
        db_encoding = student.face_encoding is not None
        
        # Check file storage
        file_path = os.path.join(Config.STORED_IMAGES_DIR, f"{student_id}.jpg")
        file_exists = os.path.exists(file_path)
        
        # Check cached encoding
        cached_encoding = ml_service.recognizer.encoding_cache.get_encoding(file_path) is not None if file_exists else False
        
        # Check multiple encodings
        multiple_encodings = student_id in ml_service.recognizer.multiple_encodings
        
        return jsonify({
            "student_id": student_id,
            "storage_status": {
                "database_encoding": db_encoding,
                "file_exists": file_exists,
                "file_path": file_path,
                "cached_encoding": cached_encoding,
                "has_multiple_encodings": multiple_encodings
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500