from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student, Course
from datetime import datetime
import pytz
from ml_service import ml_service
import base64
import io
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='attendance_routes.log'
)
logger = logging.getLogger(__name__)

attendance_bp = Blueprint('attendance', __name__, url_prefix='/api/attendance')

def base64_to_image(base64_str):
    """Convert base64 to numpy image"""
    try:
        header, encoded = base64_str.split(',', 1) if ',' in base64_str else ('', base64_str)
        image_data = base64.b64decode(encoded)
        return np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))
    except Exception as e:
        logger.error(f"Image conversion error: {str(e)}")
        raise ValueError("Invalid image format")

@attendance_bp.route('/mark', methods=['POST'])
def mark_attendance():
    try:
        data = request.get_json()
        course_id = data.get('course_id')
        image_base64 = data.get('image_base64')

        # Validate input
        if not course_id or not image_base64:
            return jsonify({
                "error": "course_id and image_base64 are both required."
            }), 400

        course = Course.query.get(course_id)
        if not course:
            return jsonify({"error": "Course not found."}), 404

        # Convert and preprocess image
        image = base64_to_image(image_base64)
        
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
                }
            }), 400

        # Perform liveness detection
        liveness_result = ml_service.verify_liveness(image)
        if not liveness_result['live']:
            return jsonify({
                "error": "Liveness check failed",
                "details": liveness_result['explanation'],
                "requirements": {
                    "movement": "Show natural movement",
                    "eyes": "Blink naturally",
                    "lighting": "Maintain good lighting"
                }
            }), 400

        # Loop through all students and find a face match
        matched_student = None
        highest_confidence = 0
        verification_time = 0

        for student in Student.query.all():
            if not student.face_encoding:
                continue
            
            # Verify student face
            result = ml_service.verify_student_identity(student.student_id, preprocessed)
            
            if result.success and result.confidence_score > highest_confidence:
                matched_student = student
                highest_confidence = result.confidence_score
                verification_time = result.verification_time

        if not matched_student:
            return jsonify({
                "error": "Face verification failed",
                "details": "Face does not match any registered student",
                "suggestions": [
                    "Ensure proper lighting",
                    "Face the camera directly",
                    "Make sure you're registered in the system"
                ]
            }), 401

        if not matched_student.courses.filter_by(course_id=course_id).first():
            return jsonify({
                "error": "Student is not registered in this course",
                "details": "Please register for the course first"
            }), 403

        # Check for active session
        session = AttendanceSession.query.filter_by(
            course_id=course_id
        ).order_by(AttendanceSession.session_number.desc()).first()
        
        if not session:
            return jsonify({
                "error": "No active session for this course",
                "details": "Please wait for teacher to start the session"
            }), 404

        # Check for existing attendance
        existing_log = Attendancelog.query.filter_by(
            student_id=matched_student.student_id,
            session_id=session.id
        ).first()
        
        if existing_log:
            return jsonify({
                "message": "Attendance already marked",
                "student_id": matched_student.student_id,
                "session_id": session.id,
                "marked_time": existing_log.time.strftime("%H:%M:%S")
            }), 200

        # Mark attendance
        now = datetime.now(pytz.timezone('Africa/Cairo'))
        attendance_time = now.time().replace(tzinfo=None)
        student_ip = request.remote_addr
        connection_strength = 'strong' if session.ip_address == student_ip else 'weak'

        new_log = Attendancelog(
            student_id=matched_student.student_id,
            session_id=session.id,
            teacher_id=session.teacher_id,
            course_id=course_id,
            date=now.date(),
            time=attendance_time,
            status='present',
            connection_strength=connection_strength
        )
        
        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Attendance marked successfully",
            "student_id": matched_student.student_id,
            "course_id": course_id,
            "session_id": session.id,
            "verification_metrics": {
                "confidence_score": highest_confidence,
                "verification_time": verification_time,
                "liveness_score": liveness_result['score'],
                "connection_type": connection_strength
            }
        }), 200

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({
            "error": "Invalid input",
            "details": str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Attendance marking error: {str(e)}")
        db.session.rollback()
        return jsonify({
            "error": "Failed to process attendance",
            "details": str(e)
        }), 500

# Add new endpoint for checking attendance status
@attendance_bp.route('/check_status', methods=['POST'])
def check_attendance_status():
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        course_id = data.get('course_id')
        
        if not student_id or not course_id:
            return jsonify({
                "error": "student_id and course_id are required"
            }), 400
            
        # Get latest session
        session = AttendanceSession.query.filter_by(
            course_id=course_id
        ).order_by(AttendanceSession.session_number.desc()).first()
        
        if not session:
            return jsonify({
                "message": "No active session found",
                "status": "no_session"
            }), 200
            
        # Check attendance
        log = Attendancelog.query.filter_by(
            student_id=student_id,
            session_id=session.id
        ).first()
        
        if log:
            return jsonify({
                "message": "Attendance already marked",
                "status": "marked",
                "time": log.time.strftime("%H:%M:%S"),
                "verification_type": log.connection_strength
            }), 200
        else:
            return jsonify({
                "message": "Attendance not marked",
                "status": "not_marked",
                "session_number": session.session_number
            }), 200
            
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({
            "error": "Failed to check attendance status",
            "details": str(e)
        }), 500