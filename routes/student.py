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

        # Get registered courses
        registered_courses = []
        for course in student.courses:
            registered_courses.append({
                "course_id": course.course_id,
                "course_name": course.course_name,
                "course_code": getattr(course, 'course_code', '')
            })

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
                "student_id": student.student_id,
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
    """Register face for current student"""
    try:
        student_id = get_jwt_identity()
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "details": "Please upload an image file",
                "retry_available": True
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "error": "No selected file",
                "details": "Please select a file to upload",
                "retry_available": True
            }), 400

        # Read and convert image file to numpy array
        try:
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    "error": "Invalid image file",
                    "details": "Could not process the uploaded image",
                    "retry_available": True
                }), 400

            logger.info(f"Image loaded successfully. Shape: {image.shape}")

            # Process image using ml_service
            preprocessed = ml_service.preprocessor.preprocess_image(image)  # Fixed this line
            
            if preprocessed is None:
                return jsonify({
                    "error": "Face registration failed",
                    "details": "Could not detect a clear face in the image",
                    "requirements": {
                        "face": "Ensure face is clearly visible",
                        "lighting": "Good lighting conditions required",
                        "position": "Face should be centered and not tilted"
                    },
                    "retry_available": True
                }), 400

            # Check liveness
            liveness_result = ml_service.liveness.analyze(preprocessed)
            
            if not liveness_result.get('live', False):
                return jsonify({
                    "error": "Liveness check failed",
                    "details": liveness_result.get('explanation', 'Liveness check failed'),
                    "requirements": {
                        "movement": "Show natural movement",
                        "lighting": "Ensure good lighting",
                        "position": "Face the camera directly"
                    },
                    "retry_available": True
                }), 400

            # Get face encoding
            encoding_result = ml_service.recognizer.get_face_encoding_for_storage(preprocessed)
            
            if not encoding_result.get('success', False):
                return jsonify({
                    "error": "Face encoding failed",
                    "details": encoding_result.get('message', 'Failed to generate face encoding'),
                    "retry_available": True
                }), 400

            # Save to database
            student = Student.query.get(student_id)
            if not student:
                return jsonify({"error": "Student not found"}), 404

            student.face_encoding = encoding_result.get('encoding')
            db.session.commit()

            return jsonify({
                "success": True,
                "message": "Face registered successfully",
                "details": {
                    "liveness_score": liveness_result.get('score', 0),
                    "quality_score": encoding_result.get('quality_score', 0)
                }
            }), 200

        except Exception as img_error:
            logger.error(f"Image processing error: {str(img_error)}")
            return jsonify({
                "error": "Image processing failed",
                "details": str(img_error),
                "retry_available": True
            }), 400

    except Exception as e:
        logger.error(f"Face registration error: {str(e)}")
        db.session.rollback()
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
    
@student_bp.route('/check-registration', methods=['GET'])
@jwt_required()
def check_registration():
    try:
        student_id = get_jwt_identity()
        student = Student.query.get(student_id)
        
        if not student:
            return jsonify({"error": "Student not found"}), 404
            
        return jsonify({
            "has_face_encoding": student.face_encoding is not None,
            "encoding_length": len(student.face_encoding) if student.face_encoding else 0
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@student_bp.route('/courses', methods=['GET'])
@jwt_required()
def get_registered_courses():
    """Get all courses registered by the current student"""
    try:
        student_id = get_jwt_identity()
        student = Student.query.get(student_id)
        
        if not student:
            return jsonify({"error": "Student not found"}), 404

        registered_courses = []
        for course in student.courses:
            # Count attendance for this course
            course_attendance = Attendancelog.query.filter_by(
                student_id=student_id, 
                course_id=course.course_id
            ).count()
            
            # Get latest attendance for this course
            latest_attendance = Attendancelog.query.filter_by(
                student_id=student_id, 
                course_id=course.course_id
            ).order_by(Attendancelog.date.desc(), Attendancelog.time.desc()).first()
            
            registered_courses.append({
                "course_id": course.course_id,
                "course_name": course.course_name,
                "course_code": getattr(course, 'course_code', ''),
                "description": getattr(course, 'description', ''),
                "instructor": getattr(course, 'instructor', ''),
                "attendance_count": course_attendance,
                "last_attendance": {
                    "date": latest_attendance.date.strftime("%Y-%m-%d") if latest_attendance else None,
                    "time": latest_attendance.time.strftime("%H:%M:%S") if latest_attendance else None,
                    "status": latest_attendance.status if latest_attendance else None
                } if latest_attendance else None
            })

        return jsonify({
            "message": "Registered courses retrieved successfully",
            "student_id": student_id,
            "registered_courses": registered_courses,
            "total_courses": len(registered_courses)
        }), 200

    except Exception as e:
        logger.error(f"Courses retrieval error: {str(e)}")
        return jsonify({"error": str(e)}), 500