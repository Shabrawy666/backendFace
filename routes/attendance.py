from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student
from datetime import datetime
import numpy as np
import face_recognition
import cv2
from sqlalchemy import desc
import logging
import pytz

attendance_bp = Blueprint('attendance', __name__, url_prefix='/api/attendance')

# Function to get Egypt local time (UTC+2)
def get_local_time():
    utc_time = datetime.utcnow()
    egypt_timezone = pytz.timezone('Africa/Cairo')
    local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(egypt_timezone)
    return local_time


# Route 1: Mark Attendance (Student)
@attendance_bp.route('/mark', methods=['POST'])
def mark_attendance():
    try:
        course_id = request.json.get('course_id')
        if not course_id:
            return jsonify({"error": "Course ID is required"}), 400

        student_ip = request.remote_addr
        local_time = get_local_time()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"error": "Could not access the webcam"}), 500

        print("Press SPACE to capture image for attendance")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Attendance - Press SPACE", frame)
            key = cv2.waitKey(1)
            if key % 256 == 32:
                break

        cap.release()
        cv2.destroyAllWindows()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame)

        if not encodings:
            return jsonify({"error": "No face detected"}), 400

        captured_encoding = encodings[0]
        students = Student.query.filter(Student.face_encoding != None).all()

        matched_student = None
        for student in students:
            known_encoding = np.array(student.face_encoding)
            results = face_recognition.compare_faces([known_encoding], captured_encoding)
            if results[0]:
                matched_student = student
                break

        if not matched_student:
            return jsonify({"error": "Face not recognized"}), 404

        student_id = matched_student.student_id
        session = AttendanceSession.query.filter_by(course_id=course_id).order_by(desc(AttendanceSession.session_number)).first()

        if not session:
            return jsonify({"error": "No active session found"}), 404

        if session.end_time:
            return jsonify({"error": "The session has already ended. Please contact the teacher to mark your attendance manually."}), 400

        connection_strength = 'strong' if session.ip_address == student_ip else 'weak'

        existing_log = Attendancelog.query.filter_by(student_id=student_id, session_id=session.id).first()
        if existing_log:
            return jsonify({"message": "Attendance already marked for this student in this session"}), 200

        attendancelog = Attendancelog(
            student_id=student_id,
            session_id=session.id,
            teacher_id=session.teacher_id,
            course_id=course_id,
            date=local_time.date(),
            time=local_time,
            status='present',
            connection_strength=connection_strength
        )

        db.session.add(attendancelog)
        db.session.commit()

        return jsonify({
            "message": f"Attendance marked for student {student_id} in session {session.session_number}",
            "connection_strength": connection_strength
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
