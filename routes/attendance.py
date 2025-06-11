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
import cv2

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
        # Log the start of attendance marking
        logger.info("Starting attendance marking process")

        # Validate course_id
        course_id = request.form.get('course_id')
        if not course_id:
            return jsonify({
                "error": "No course_id provided",
                "details": "Course ID is required"
            }), 400

        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "details": "Please provide a face image"
            }), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "error": "No selected file",
                "details": "Please select an image file"
            }), 400

        # Read and convert image
        logger.info("Processing uploaded image")
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Invalid image file",
                "details": "Could not process the image file"
            }), 400

        logger.info(f"Image processed successfully. Shape: {image.shape}")

        # Process image
        preprocessed = ml_service.preprocessor.preprocess_image(image)
        if preprocessed is None:
            return jsonify({
                "error": "Image preprocessing failed",
                "details": "Could not detect a clear face in the image",
                "requirements": {
                    "face": "Ensure face is clearly visible",
                    "lighting": "Good lighting conditions required",
                    "position": "Face must be centered"
                }
            }), 400

        # Get course and check if it exists
        course = Course.query.get(course_id)
        if not course:
            return jsonify({"error": "Course not found"}), 404

        # Get enrolled students directly from the course
        enrolled_students = list(course.students)
        logger.info(f"Checking {len(enrolled_students)} enrolled students")

        # Find matching student
        matched_student = None
        highest_confidence = 0
        verification_time = 0

        for student in enrolled_students:
            if not student.face_encoding:
                logger.warning(f"Student {student.student_id} has no face encoding")
                continue
            
            logger.info(f"Checking student {student.student_id}")
            result = ml_service.verify_student_identity(student.student_id, preprocessed)
            
            if isinstance(result, dict):
                success = result.get('success', False)
                confidence = result.get('confidence_score', 0)
                ver_time = result.get('verification_time', 0)
            else:
                # If result is RecognitionResult object
                success = result.success
                confidence = result.confidence_score if hasattr(result, 'confidence_score') else 0
                ver_time = result.verification_time if hasattr(result, 'verification_time') else 0
            
            if success and confidence > highest_confidence:
                matched_student = student
                highest_confidence = confidence
                verification_time = ver_time
                logger.info(f"New best match: Student {student.student_id} with confidence {highest_confidence}")

        if not matched_student:
            return jsonify({
                "error": "Face verification failed",
                "details": "No matching student found",
                "suggestions": [
                    "Ensure you're registered in the system",
                    "Try with better lighting",
                    "Face the camera directly"
                ]
            }), 401

        # Check for active session
        session = AttendanceSession.query.filter_by(
            course_id=course_id,
            end_time=None
        ).order_by(AttendanceSession.start_time.desc()).first()

        if not session:
            return jsonify({
                "error": "No active session",
                "details": "No active attendance session found for this course"
            }), 404

        # Check for existing attendance
        existing_log = Attendancelog.query.filter_by(
            student_id=matched_student.student_id,
            session_id=session.id
        ).first()

        if existing_log:
            return jsonify({
                "message": "Attendance already marked",
                "details": {
                    "student_id": matched_student.student_id,
                    "student_name": matched_student.name,
                    "marked_time": existing_log.time.strftime("%H:%M:%S"),
                    "verification_score": highest_confidence
                }
            }), 200

        # Mark attendance
        try:
            now = datetime.now()
            student_ip = request.remote_addr
            
            # Determine connection strength
            connection_strength = 'strong' if session.ip_address == student_ip else 'weak'

            new_log = Attendancelog(
                student_id=matched_student.student_id,
                session_id=session.id,
                teacher_id=session.teacher_id,
                course_id=course_id,
                date=now.date(),
                time=now.time(),
                status='present',
                connection_strength=connection_strength
            )
            
            db.session.add(new_log)
            db.session.commit()

            return jsonify({
                "success": True,
                "message": "Attendance marked successfully",
                "details": {
                    "student_id": matched_student.student_id,
                    "student_name": matched_student.name,
                    "course_id": course_id,
                    "session_id": session.id,
                    "marked_time": now.strftime("%H:%M:%S"),
                    "verification_metrics": {
                        "confidence_score": highest_confidence,
                        "verification_time": verification_time,
                        "connection_type": connection_strength
                    }
                }
            }), 200

        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            db.session.rollback()
            return jsonify({
                "error": "Database error",
                "details": "Failed to save attendance record"
            }), 500

    except Exception as e:
        logger.error(f"Attendance marking error: {str(e)}")
        return jsonify({
            "error": "Failed to process attendance",
            "details": str(e)
        }), 500