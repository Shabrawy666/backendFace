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
        return np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))
    except Exception as e:
        logger.error(f"Image conversion error: {str(e)}")
        raise ValueError("Invalid image format")

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
        # Get student_id from JWT token
        student_id = get_jwt_identity()

        data = request.get_json()
        face_image = data.get('face_image')
        
        if not face_image:
            return jsonify({"error": "Face image is required"}), 400

        # Find the student using token's student_id
        student = Student.query.get(student_id)
        if not student:
            logger.error(f"Student not found: {student_id}")
            return jsonify({"error": "Student not found"}), 404

        # Convert and validate image
        image = base64_to_image(face_image)
        
        # Check image quality
        preprocessed = ml_service.preprocess_image(image)
        if preprocessed is None:
            return jsonify({
                "error": "Image quality check failed",
                "details": "Face image does not meet quality requirements",
                "requirements": {
                    "lighting": "Ensure even lighting",
                    "clarity": "Image must be clear and sharp",
                    "position": "Face must be centered"
                },
                "retry_available": True
            }), 400

        # Check liveness
        liveness_result = ml_service.verify_liveness(image)
        if not liveness_result['live']:
            return jsonify({
                "error": "Liveness check failed",
                "details": liveness_result['explanation'],
                "requirements": {
                    "movement": "Show natural movement",
                    "eyes": "Blink naturally",
                    "lighting": "Maintain good lighting"
                },
                "retry_available": True
            }), 400

        # Get face encoding
        encoding_result = ml_service.get_face_encoding(preprocessed)
        if not encoding_result['success']:
            return jsonify({
                "error": "Face registration failed",
                "details": encoding_result.get('message', 'Face encoding failed'),
                "retry_available": True
            }), 400

        # Store in database
        student.face_encoding = encoding_result['encoding']
        db.session.commit()

        # Return success response with student_id from token
        return jsonify({
            "message": "Face registration successful",
            "student_id": student_id,
            "registration_metrics": {
                "image_quality": "good",
                "liveness_score": liveness_result['score'],
                "liveness_details": liveness_result['explanation']
            }
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Face registration error: {str(e)}")
        return jsonify({
            "error": "Face processing error",
            "details": str(e),
            "retry_available": True
        }), 500

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
        
        # Updated preprocessing call
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

        # Updated liveness check
        liveness_result = ml_service.verify_liveness(image)

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