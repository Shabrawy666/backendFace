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

student_bp = Blueprint('student', __name__, url_prefix='/api/student')

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def base64_to_image(base64_str):
    """Convert base64 string to numpy array"""
    header, encoded = base64_str.split(',', 1) if ',' in base64_str else ('', base64_str)
    image_data = base64.b64decode(encoded)
    return np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))

@student_bp.route('/login', methods=['POST'])
def login_student():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        face_image = data.get('face_image')  # Optional field

        # Validation
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        student = Student.query.filter_by(email=email).first()
        if not student or not bcrypt.checkpw(password.encode('utf-8'), student._password.encode('utf-8')):
            return jsonify({"error": "Invalid email or password"}), 401

        # Create token (valid for 1 hour)
        access_token = create_access_token(
            identity=student.student_id,
            expires_delta=timedelta(hours=1)
        )  # Fixed: Added missing closing parenthesis here
            
        # Case 1: Already has face registered
        if student.face_encoding:
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

        # Case 2: Needs face registration but no image provided
        if not face_image:
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
                        "distance": "About arm's length away"
                    },
                    "field_name": "face_image",
                    "retry_limit": 3
                }
            }), 202  # Accepted status code

        # Case 3: Processing face registration
        try:
            image = base64_to_image(face_image)
            encoding_result = ml_service.recognizer.get_face_encoding_for_storage(image)
            
            if not encoding_result['success']:
                return jsonify({
                    "error": "Face registration failed",
                    "details": encoding_result.get('message', 'Quality check failed'),
                    "retry_available": True
                }), 400

            # Save to database
            student.face_encoding = encoding_result['encoding']
            db.session.commit()

            return jsonify({
                "message": "Login and face registration completed",
                "access_token": access_token,
                "face_registered": True,
                "quality_metrics": encoding_result.get('quality_metrics', {})
            }), 200

        except Exception as e:
            db.session.rollback()
            return jsonify({
                "error": "Face processing error",
                "details": str(e),
                "retry_available": True
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        db.session.rollback()
        return jsonify({"error": str(e)}), 500