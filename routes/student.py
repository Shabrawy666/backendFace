from flask import Blueprint, request, jsonify
import bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from models import db, Student, Attendancelog, Course
import re
import numpy as np
from datetime import timedelta
from ml_service import ml_service
import base64
import io
from PIL import Image
import logging
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='student_routes.log'
)
logger = logging.getLogger(__name__)

student_bp = Blueprint('student', __name__, url_prefix='/api/student')

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def base64_to_image(base64_str):
    """Convert base64 string to numpy array"""
    try:
        header, encoded = base64_str.split(',', 1) if ',' in base64_str else ('', base64_str)
        image_data = base64.b64decode(encoded)
        image = np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))
        
        # Add validation and logging
        if image is None or image.size == 0:
            raise ValueError("Decoded image is empty")
            
        logger.info(f"Image decoded successfully. Shape: {image.shape}, dtype: {image.dtype}")
        return image

    except Exception as e:
        logger.error(f"Image conversion error: {str(e)}")
        raise ValueError(f"Invalid image format: {str(e)}")

@student_bp.route('/login', methods=['POST'])
def login_student():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Basic validation
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        # Verify email and password
        student = Student.query.filter_by(email=email).first()
        if not student or not bcrypt.checkpw(password.encode('utf-8'), student._password.encode('utf-8')):
            return jsonify({"error": "Invalid email or password"}), 401

        # Create token
        access_token = create_access_token(
            identity=student.student_id,
            expires_delta=timedelta(hours=1)
        )

        # Check if face encoding exists
        if student.face_encoding:
            # Face already registered, proceed with normal login
            return jsonify({
                "message": "Login successful",
                "access_token": access_token,
                "student_data": {
                    "student_id": student.student_id,
                    "name": student.name,
                    "email": student.email
                },
                "face_registered": True
            }), 200
        else:
            # Send 202 to request face capture
            return jsonify({
                "message": "Face registration required",
                "access_token": access_token,
                "face_registered": False,
                "action_required": {
                    "type": "capture_face",
                    "instructions": "Please center your face in the frame",
                    "requirements": {
                        "lighting": "Even lighting, no shadows",
                        "angle": "Face directly facing camera",
                        "distance": "About arm's length away",
                        "movement": "Maintain natural movement for liveness check",
                        "eyes": "Keep eyes open and blink naturally"
                    }
                }
            }), 202

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@student_bp.route('/register-face', methods=['POST'])
@jwt_required()
def register_face():
    try:
        student_id = get_jwt_identity()
        data = request.get_json()
        face_image = data.get('face_image')
        
        if not face_image:
            return jsonify({"error": "Face image is required"}), 400

        student = Student.query.get(student_id)
        if not student:
            return jsonify({"error": "Student not found"}), 404

        print("DEBUG: Starting face registration...")
        
        # Step 1: Convert image
        try:
            image = base64_to_image(face_image)
            print(f"DEBUG: Image converted: {image.shape}")
        except Exception as e:
            print(f"DEBUG: Image conversion failed: {str(e)}")
            return jsonify({"error": "Image conversion failed", "details": str(e)}), 400
        
        # Step 2: Convert to BGR
        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print("DEBUG: BGR conversion done")
        except Exception as e:
            print(f"DEBUG: BGR conversion failed: {str(e)}")
            return jsonify({"error": "BGR conversion failed", "details": str(e)}), 400
        
        # Step 3: Preprocess
        try:
            preprocessed = ml_service.preprocess_image(image_bgr)
            if preprocessed is None:
                print("DEBUG: Preprocessing failed")
                return jsonify({"error": "Preprocessing failed", "details": "Could not detect face"}), 400
            print(f"DEBUG: Preprocessing successful: {preprocessed.shape}")
        except Exception as e:
            print(f"DEBUG: Preprocessing exception: {str(e)}")
            return jsonify({"error": "Preprocessing exception", "details": str(e)}), 400
        
        # Step 4: BYPASS LIVENESS COMPLETELY FOR FRONTEND
        print("DEBUG: Bypassing liveness for frontend compatibility...")
        liveness_bypassed = True
        
        # Step 5: Face encoding with detailed debugging
        # Step 5: Face encoding (skip internal preprocessing since we already did it)
        try:
            print("DEBUG: Getting face encoding...")
            print(f"DEBUG: Input to encoding - Shape: {preprocessed.shape}, dtype: {preprocessed.dtype}")
            
            # Skip preprocessing since we already preprocessed the image
            encoding_result = ml_service.get_face_encoding(preprocessed, skip_preprocessing=True)
            
            print(f"DEBUG: Encoding result: {encoding_result.get('success', False)}")
            
            if not encoding_result.get('success', False):
                return jsonify({
                    "error": "Face encoding failed",
                    "details": encoding_result.get('message', 'Unknown encoding error')
                }), 400
        
        except Exception as e:
            print(f"DEBUG: Encoding exception: {str(e)}")
            print(f"DEBUG: Exception type: {type(e).__name__}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            return jsonify({
                "error": "Face encoding exception", 
                "details": str(e),
                "exception_type": type(e).__name__
            }), 400
        
        # Step 6: Save to database
        try:
            print("DEBUG: Saving to database...")
            student.face_encoding = encoding_result.get('encoding')
            db.session.commit()
            print("DEBUG: Database save successful")
            
            return jsonify({
                "message": "Face registration successful",
                "student_id": student_id,
                "encoding_length": len(encoding_result.get('encoding', [])),
                "liveness_bypassed": liveness_bypassed,
                "debug_info": {
                    "preprocessing_successful": True,
                    "encoding_successful": True,
                    "frontend_mode": True
                }
            }), 200
            
        except Exception as e:
            print(f"DEBUG: Database error: {str(e)}")
            db.session.rollback()
            return jsonify({"error": "Database save failed", "details": str(e)}), 400
    
    except Exception as e:
        print(f"DEBUG: Outer exception: {str(e)}")
        db.session.rollback()
        return jsonify({"error": "Internal error", "details": str(e)}), 500

@student_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_student_profile():
    try:
        student_id = get_jwt_identity()
        student = Student.query.filter_by(student_id=student_id).first()
        
        if not student:
            return jsonify({"error": "Student not found"}), 404

        records = Attendancelog.query.filter_by(student_id=student_id).all()
        attendance_list = []
        for record in records:
            course = Course.query.get(record.course_id)
            attendance_list.append({
                "course_name": course.course_name if course else "Unknown",
                "date": record.date.strftime("%Y-%m-%d"),
                "time": record.time.strftime("%H:%M:%S"),
                "status": record.status,
                "verified_with": record.connection_strength
            })

        return jsonify({
            "message": "Profile retrieved successfully",
            "profile": {
                "student_id": student.student_id,
                "name": student.name,
                "email": student.email,
                "face_registered": student.face_encoding is not None,
                "attendance_records": attendance_list
            }
        }), 200

    except Exception as e:
        logger.error(f"Profile retrieval error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@student_bp.route('/attendance', methods=['GET'])
@jwt_required()
def get_student_attendance():
    try:
        student_id = get_jwt_identity()
        records = Attendancelog.query.filter_by(student_id=student_id).all()

        attendance_list = []
        for record in records:
            course = Course.query.get(record.course_id)
            attendance_list.append({
                "course_name": course.course_name if course else "Unknown",
                "date": record.date.strftime("%Y-%m-%d"),
                "time": record.time.strftime("%H:%M:%S"),
                "status": record.status
            })

        return jsonify({
            "student_id": student_id,
            "attendance_records": attendance_list
        }), 200

    except Exception as e:
        logger.error(f"Attendance retrieval error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@student_bp.route('/register_course', methods=['POST'])
@jwt_required()
def register_course():
    try:
        data = request.get_json()
        course_id = data.get("course_id")
        student_id = get_jwt_identity()

        student = Student.query.get(student_id)
        course = Course.query.get(course_id)

        if not student or not course:
            return jsonify({"error": "Student or Course not found"}), 404

        if course in student.courses:
            return jsonify({"message": "Already registered in this course"}), 200

        student.courses.append(course)
        db.session.commit()

        return jsonify({
            "message": f"Registered for {course.course_name} successfully",
            "requires_face_verification": not student.face_encoding
        }), 200

    except Exception as e:
        logger.error(f"Course registration error: {str(e)}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# New endpoint for checking face image quality
@student_bp.route('/check_face_quality', methods=['POST'])
@jwt_required()
def check_face_quality():
    try:
        data = request.get_json()
        face_image = data.get('face_image')
        
        if not face_image:
            return jsonify({"error": "Face image is required"}), 400

        image = base64_to_image(face_image)
        
        # Check image quality
        preprocessed = ml_service.preprocess_image(image)
        if preprocessed is None:
            return jsonify({
                "success": False,
                "message": "Image quality check failed",
                "suggestions": [
                    "Ensure good lighting",
                    "Face the camera directly",
                    "Keep face centered in frame"
                ]
            }), 400

        # Check face quality
        quality_check = ml_service.check_face_quality(preprocessed)
        if not quality_check:
            return jsonify({
                "success": False,
                "message": "Face quality check failed",
                "suggestions": [
                    "Ensure proper lighting",
                    "Keep face clearly visible",
                    "Avoid blurry images"
                ]
            }), 400

        # Check liveness
        liveness_result = ml_service.check_liveness(preprocessed)

        # Get system metrics
        metrics = ml_service.get_performance_metrics()

        return jsonify({
            "success": True,
            "quality_check": {
                "image_quality": "good",
                "liveness": liveness_result['live'],
                "liveness_score": liveness_result['score'],
                "liveness_details": liveness_result['explanation']
            },
            "system_metrics": metrics
        }), 200

    except Exception as e:
        logger.error(f"Face quality check error: {str(e)}")
        return jsonify({"error": str(e)}), 500
        