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

        try:
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

            # Check face quality
            quality_check = ml_service.check_face_quality(preprocessed)
            if not quality_check:
                return jsonify({
                    "error": "Face quality check failed",
                    "details": "Face image does not meet quality standards",
                    "requirements": {
                        "clarity": "Ensure image is clear",
                        "brightness": "Check lighting conditions",
                        "position": "Face must be clearly visible"
                    }
                }), 400

            # Perform liveness detection
            liveness_result = ml_service.check_liveness(preprocessed)
            if not liveness_result.get('live', False):
                return jsonify({
                    "error": "Liveness check failed",
                    "details": liveness_result.get('explanation', 'Liveness check failed'),
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
                
                # Verify student face using the correct method
                result = ml_service.verify_student_identity(student.student_id, preprocessed)
                
                if result.get('success', False) and result.get('confidence_score', 0) > highest_confidence:
                    matched_student = student
                    highest_confidence = result.get('confidence_score', 0)
                    verification_time = result.get('verification_time', 0)

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

            # Rest of your code for course registration check and attendance marking...
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

            # Get final metrics
            metrics = ml_service.get_performance_metrics()

            return jsonify({
                "success": True,
                "message": "Attendance marked successfully",
                "student_id": matched_student.student_id,
                "course_id": course_id,
                "session_id": session.id,
                "verification_metrics": {
                    "confidence_score": highest_confidence,
                    "verification_time": verification_time,
                    "liveness_score": liveness_result.get('score', 1.0),
                    "liveness_details": liveness_result.get('explanation', ''),
                    "connection_type": connection_strength,
                    "system_metrics": metrics
                }
            }), 200

        except Exception as e:
            logger.error(f"Face processing error: {str(e)}")
            return jsonify({
                "error": "Face processing error",
                "details": str(e)
            }), 400

    except Exception as e:
        logger.error(f"Attendance marking error: {str(e)}")
        db.session.rollback()
        return jsonify({
            "error": "Failed to process attendance",
            "details": str(e)
        }), 500