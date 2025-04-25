from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student, Teacher
from flask_jwt_extended import create_access_token
from datetime import datetime
import logging
import bcrypt
from sqlalchemy import desc  # Needed for ordering by session ID

teacher_bp = Blueprint('teacher', __name__, url_prefix='/api/teacher')

# Set up logging to a file
logging.basicConfig(filename='app_errors.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Route 1: Login Teacher
@teacher_bp.route('/login', methods=['POST'])
def login_teacher():
    try:
        data = request.get_json()
        teacher_id = data.get('teacher_id')
        password = data.get('password')
        logging.info(f"Received teacher_id: {teacher_id}, password: {password}")
        if not teacher_id or not password:
            logging.error("Missing teacher_id or password")
            return jsonify({"error": "Teacher ID and password are required"}), 400

        teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
        logging.info(f"Teacher query result: {teacher}")
        if not teacher:
            logging.error(f"Teacher with ID {teacher_id} not found")
            return jsonify({"error": "Invalid teacher ID or password"}), 401

        if not bcrypt.checkpw(password.encode('utf-8'), teacher._password.encode('utf-8')):
            logging.error("Invalid password for teacher ID %s", teacher_id)
            return jsonify({"error": "Invalid teacher ID or password"}), 401

        access_token = create_access_token(identity={"teacher_id": teacher.teacher_id, "role": "teacher"})
        return jsonify({"access_token": access_token}), 200

    except Exception as e:
        logging.error(f"Error during teacher login: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# Route 2: Start Attendance Session
@teacher_bp.route('/start_session', methods=['POST'])
def start_attendance_session():
    try:
        data = request.get_json()
        teacher_id = data.get('teacher_id')
        course_id = data.get('course_id')

        if not teacher_id or not course_id:
            return jsonify({"error": "Teacher ID and Course ID are required"}), 400

        # Get the teacher's IP address
        teacher_ip = request.remote_addr
        local_time = datetime.utcnow()

        last_session = AttendanceSession.query.filter_by(course_id=course_id).order_by(desc(AttendanceSession.id)).first()
        new_session_number = 1
        if last_session:
            new_session_number = last_session.session_number + 1

        session = AttendanceSession(
            course_id=course_id,
            teacher_id=teacher_id,
            ip_address=teacher_ip,
            start_time=local_time
        )

        session.session_number = new_session_number

        db.session.add(session)
        db.session.commit()

        return jsonify({
            "message": "Attendance session started",
            "session_id": session.id,
            "session_number": new_session_number,
            "ip_address": teacher_ip
        }), 201

    except Exception as e:
        logging.error(f"Error starting attendance session: {str(e)}")
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500


# Route 3: End Attendance Session
@teacher_bp.route('/end_session', methods=['POST'])
def end_attendance_session():
    try:
        data = request.get_json()
        teacher_id = data.get('teacher_id')
        course_id = data.get('course_id')

        if not teacher_id or not course_id:
            return jsonify({"error": "Teacher ID and Course ID are required"}), 400

        session = AttendanceSession.query.filter_by(course_id=course_id).order_by(desc(AttendanceSession.id)).first()

        if not session:
            return jsonify({"error": "No active session found"}), 404

        session.end_time = datetime.utcnow()

        db.session.commit()

        return jsonify({
            "message": "Attendance session ended",
            "session_id": session.id
        }), 200

    except Exception as e:
        logging.error(f"Error ending attendance session: {str(e)}")
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500


# Route 4: Manually Edit Attendance (Toggle between "present" and "absent")
@teacher_bp.route('/edit_attendance', methods=['POST'])
def edit_attendance():
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        session_number = data.get('session_number')  # Changed from session_id to session_number
        course_id = data.get('course_id')

        if not student_id or not session_number or not course_id:
            return jsonify({"error": "Student ID, Session Number, and Course ID are required"}), 400

        # Retrieve the session based on course_id and session_number
        session = AttendanceSession.query.filter_by(course_id=course_id, session_number=session_number).first()

        if not session:
            return jsonify({"error": "Session not found for the given course and session number"}), 404

        # Retrieve the attendance log based on student_id, session_number, and course_id
        log = Attendancelog.query.filter_by(student_id=student_id, session_id=session.id, course_id=course_id).first()

        if not log:
            return jsonify({"error": "Attendance log not found"}), 404

        # Toggle the attendance status between "present" and "absent"
        if log.status == 'present':
            log.status = 'absent'
        elif log.status == 'absent':
            log.status = 'present'

        db.session.commit()

        return jsonify({"message": f"Attendance for student {student_id} updated to {log.status} in session {session_number} for course {course_id}"}), 200

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error editing attendance: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
