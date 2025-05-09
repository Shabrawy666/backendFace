from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student, Teacher, Course
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

# Helper function to check course ownership
def teacher_owns_course(teacher_id, course_id):
    course = db.session.get(Course, course_id)
    return course and course.teacher_id == teacher_id


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

        if not teacher_owns_course(teacher_id, course_id):
            return jsonify({"error": "Unauthorized access to course"}), 403

        teacher_ip = request.remote_addr
        local_time = datetime.utcnow()

        last_session = AttendanceSession.query.filter_by(course_id=course_id).order_by(desc(AttendanceSession.id)).first()
        new_session_number = (last_session.session_number + 1) if last_session else 1

        session = AttendanceSession(
            course_id=course_id,
            teacher_id=teacher_id,
            ip_address=teacher_ip,
            start_time=local_time,
            session_number=new_session_number
        )

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

        if not teacher_owns_course(teacher_id, course_id):
            return jsonify({"error": "Unauthorized access to course"}), 403

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


# Route 4: Manually Edit Attendance
@teacher_bp.route('/edit_attendance', methods=['POST'])
def edit_attendance():
    try:
        data = request.get_json()
        teacher_id = data.get('teacher_id')
        student_id = data.get('student_id')
        session_number = data.get('session_number')
        course_id = data.get('course_id')

        if not all([teacher_id, student_id, session_number, course_id]):
            return jsonify({"error": "Teacher ID, Student ID, Session Number, and Course ID are required"}), 400

        if not teacher_owns_course(teacher_id, course_id):
            return jsonify({"error": "Unauthorized access to course"}), 403

        session = AttendanceSession.query.filter_by(course_id=course_id, session_number=session_number).first()

        if not session:
            return jsonify({"error": "Session not found for the given course and session number"}), 404

        log = Attendancelog.query.filter_by(student_id=student_id, session_id=session.id, course_id=course_id).first()

        if not log:
            return jsonify({"error": "Attendance log not found"}), 404

        log.status = 'absent' if log.status == 'present' else 'present'

        db.session.commit()

        return jsonify({
            "message": f"Attendance for student {student_id} updated to {log.status} in session {session_number} for course {course_id}"
        }), 200

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error editing attendance: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
