from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student, Course
from datetime import datetime
import pytz
from ml_service import ml_service  # Add this import
import base64
import io
import numpy as np
from PIL import Image

attendance_bp = Blueprint('attendance', __name__, url_prefix='/api/attendance')

def base64_to_image(base64_str):
    """Convert base64 to numpy image"""
    header, encoded = base64_str.split(',', 1) if ',' in base64_str else ('', base64_str)
    image_data = base64.b64decode(encoded)
    return np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))

@attendance_bp.route('/mark', methods=['POST'])
def mark_attendance():
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
        # Decode base64 image to numpy array
        image = base64_to_image(image_base64)

        # Loop through all students and find a face match
        matched_student = None
        for student in Student.query.all():
            if not student.face_encoding:
                continue
            # Use the correct face verification method
            result = ml_service.recognizer.verify_student(student.student_id, image)
            # result: RecognitionResult object
            if hasattr(result, "success") and result.success:
                matched_student = student
                break

        if not matched_student:
            return jsonify({"error": "Face does not match any registered student."}), 401
        if not matched_student.courses.filter_by(course_id=course_id).first():
            return jsonify({"error": "Student is not registered in this course."}), 403

        # Usual attendance and session logic
        session = AttendanceSession.query.filter_by(
            course_id=course_id
        ).order_by(AttendanceSession.session_number.desc()).first()
        if not session:
            return jsonify({"error": "No active session for this course"}), 404

        existing_log = Attendancelog.query.filter_by(
            student_id=matched_student.student_id,
            session_id=session.id
        ).first()
        if existing_log:
            return jsonify({
                "message": "Attendance already marked.",
                "student_id": matched_student.student_id,
                "session_id": session.id
            }), 200

        # Mark the student as present
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
            "session_id": session.id
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to process attendance", "details": str(e)}), 500