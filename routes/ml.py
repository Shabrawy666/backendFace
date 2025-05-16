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

@ml_bp.route('/health', methods=['GET'])
def ml_health_check():
    """Check if ML service is functioning properly"""
    try:
        # Perform a simple check to see if the ML service is operational
        metrics = ml_service.get_performance_metrics()
        
        return jsonify({
            "status": "ok",
            "message": "ML service is operational",
            "metrics": metrics,
            "deepface_available": ml_service.deepface_available
        }), 200
    except Exception as e:
        logger.error(f"ML health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"ML service error: {str(e)}"
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