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

def calculate_connection_strength(session, student_ip, face_verification_score=None):
    """
    Calculate connection strength based on multiple factors
    Returns: 'strong', 'moderate', or 'weak'
    """
    factors = {
        'ip_match': False,
        'face_verified': False,
        'time_valid': False
    }
    
    # 1. Check IP address match
    factors['ip_match'] = (session.ip_address == student_ip)
    
    # 2. Check face verification if available
    if face_verification_score is not None:
        factors['face_verified'] = (face_verification_score > 0.8)  # 80% confidence threshold
    
    # 3. Check if attendance is marked within valid timeframe
    session_start = session.start_time
    current_time = datetime.utcnow()
    time_difference = (current_time - session_start).total_seconds() / 60  # in minutes
    factors['time_valid'] = (time_difference <= 30)  # within 30 minutes of session start
    
    # Calculate strength based on factors
    if all(factors.values()):
        return 'strong'
    elif sum(factors.values()) >= 2:
        return 'moderate'
    else:
        return 'weak'

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
        logger.info("Starting attendance marking process")

        # Validate course_id and image
        course_id = request.form.get('course_id')
        if not course_id:
            return jsonify({
                "error": "No course_id provided",
                "details": "Course ID is required"
            }), 400

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

        # Process image
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({
                "error": "Invalid image file",
                "details": "Could not process the image file"
            }), 400

        logger.info(f"Image processed successfully. Shape: {image.shape}")

        # Preprocess image
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

        # Verify course and get students
        course = Course.query.get(course_id)
        if not course:
            return jsonify({"error": "Course not found"}), 404

        enrolled_students = list(course.students)
        logger.info(f"Checking {len(enrolled_students)} enrolled students")

        # Find matching student with highest confidence
        matched_student = None
        highest_confidence = 0
        verification_time = 0

        for student in enrolled_students:
            if not student.face_encoding:
                logger.warning(f"Student {student.student_id} has no face encoding")
                continue
            
            result = ml_service.verify_student_identity(student.student_id, preprocessed)
            
            if isinstance(result, dict):
                success = result.get('success', False)
                confidence = result.get('confidence_score', 0)
                ver_time = result.get('verification_time', 0)
            else:
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

        # Check active session
        session = AttendanceSession.query.filter_by(
            course_id=course_id,
            end_time=None
        ).order_by(AttendanceSession.start_time.desc()).first()

        if not session:
            return jsonify({
                "error": "No active session",
                "details": "No active attendance session found for this course"
            }), 404

        # Check existing attendance
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

        # Mark attendance with enhanced connection strength
        try:
            now = datetime.utcnow()
            student_ip = request.remote_addr
            
            # Calculate connection strength
            connection_strength = calculate_connection_strength(
                session=session,
                student_ip=student_ip,
                face_verification_score=highest_confidence
            )

            # Create new attendance log with only the fields that exist in your model
            new_log = Attendancelog(
                student_id=matched_student.student_id,
                session_id=session.id,
                course_id=course_id,
                teacher_id=session.teacher_id,
                date=now.date(),
                time=now.time(),
                status='present',
                connection_strength=connection_strength
            )
            
            db.session.add(new_log)
            db.session.commit()

            logger.info(f"Attendance marked successfully for student {matched_student.student_id}")

            # Prepare verification factors for response only
            verification_factors = {
                'ip_match': session.ip_address == student_ip,
                'face_verified': highest_confidence > 0.8,
                'time_valid': True
            }

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
                        "connection_strength": connection_strength,
                        "factors": verification_factors
                    }
                }
            }), 200

        except Exception as db_error:
            logger.error(f"Database error while marking attendance: {str(db_error)}")
            # Don't rollback if the data was actually saved
            if 'new_log' not in locals() or not new_log.id:
                db.session.rollback()
            
            # Check if attendance was actually saved despite the error
            existing_log = Attendancelog.query.filter_by(
                student_id=matched_student.student_id,
                session_id=session.id
            ).first()
            
            if existing_log:
                # Attendance was saved, return success
                return jsonify({
                    "success": True,
                    "message": "Attendance marked successfully (with warning)",
                    "warning": "There was a minor issue, but attendance was recorded",
                    "details": {
                        "student_id": matched_student.student_id,
                        "student_name": matched_student.name,
                        "marked_time": existing_log.time.strftime("%H:%M:%S")
                    }
                }), 200
            else:
                # Attendance wasn't saved, return error
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