from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student
from datetime import datetime
import numpy as np
import cv2
from sqlalchemy import desc
import pytz
import base64
import face_recognition

attendance_bp = Blueprint('attendance', __name__, url_prefix='/api/attendance')

def get_local_time():
    utc_time = datetime.utcnow()
    egypt_timezone = pytz.timezone('Africa/Cairo')
    return utc_time.replace(tzinfo=pytz.utc).astimezone(egypt_timezone)

def base64_to_image(base64_string):
    try:
        # Remove header if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

@attendance_bp.route('/mark', methods=['POST'])
def mark_attendance():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('course_id'):
            return jsonify({"error": "Course ID is required"}), 400
        if not data.get('image_base64'):
            return jsonify({"error": "Face image is required"}), 400

        course_id = data['course_id']
        base64_image = data['image_base64']
        student_ip = request.remote_addr
        local_time = get_local_time()

        # Convert base64 to OpenCV image
        frame = base64_to_image(base64_image)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return jsonify({"error": "No face detected in the image"}), 400

        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not encodings:
            return jsonify({"error": "Could not extract face features"}), 400

        captured_encoding = encodings[0]  # Take first face found

        # Get all students with face encodings
        students = Student.query.filter(Student.face_encoding.isnot(None)).all()

        # Find matching student
        matched_student = None
        for student in students:
            known_encoding = np.array(student.face_encoding)
            match = face_recognition.compare_faces([known_encoding], captured_encoding, tolerance=0.5)
            if match[0]:
                matched_student = student
                break

        if not matched_student:
            return jsonify({"error": "No matching student found"}), 404

        # Get current session
        session = AttendanceSession.query.filter_by(
            course_id=course_id
        ).order_by(desc(AttendanceSession.session_number)).first()

        if not session:
            return jsonify({"error": "No active session found for this course"}), 404

        if session.end_time:
            return jsonify({"error": "This session has already ended"}), 400

        # Check if already marked attendance
        existing_log = Attendancelog.query.filter_by(
            student_id=matched_student.student_id,
            session_id=session.id
        ).first()

        if existing_log:
            return jsonify({
                "message": "Attendance already marked",
                "student_id": matched_student.student_id,
                "course_id": course_id,
                "session_id": session.id
            }), 200

        # Determine connection strength
        connection_strength = 'strong' if session.ip_address == student_ip else 'weak'

        # Create new attendance log
        new_log = Attendancelog(
            student_id=matched_student.student_id,
            session_id=session.id,
            teacher_id=session.teacher_id,
            course_id=course_id,
            date=local_time.date(),
            time=local_time.time(),
            status='present',
            connection_strength=connection_strength
        )

        db.session.add(new_log)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Attendance marked successfully",
            "student_id": matched_student.student_id,
            "student_name": matched_student.name,
            "course_id": course_id,
            "session_id": session.id,
            "connection_strength": connection_strength,
            "timestamp": local_time.isoformat()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "error": "Failed to process attendance",
            "details": str(e)
        }), 500