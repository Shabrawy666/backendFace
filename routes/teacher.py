from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student, Teacher, Course
from flask_jwt_extended import create_access_token
from flask_jwt_extended import jwt_required
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


@teacher_bp.route('/login', methods=['POST'])
def login_teacher():
    """Teacher login endpoint using email and password"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        teacher = Teacher.query.filter_by(email=email).first()
        
        if not teacher or not bcrypt.checkpw(password.encode('utf-8'), teacher._password.encode('utf-8')):
            return jsonify({"error": "Invalid email or password"}), 401

        access_token = create_access_token(
            identity=teacher.teacher_id,
            expires_delta=timedelta(hours=1),
            additional_claims={"role": "teacher"}
        )

        teacher_data = {
            "teacher_id": teacher.teacher_id,
            "name": teacher.name,
            "email": teacher.email,
                 }

        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "teacher_data": teacher_data
        }), 200

    except Exception as e:
        logging.error(f"Teacher login error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@teacher_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_teacher_profile():
    """Get complete teacher profile with courses and sessions"""
    try:
        teacher_id = get_jwt_identity()
        teacher = Teacher.query.get(teacher_id)
        
        if not teacher:
            return jsonify({"error": "Teacher not found"}), 404

        # Get courses with enrollment counts
        courses = Course.query.filter_by(teacher_id=teacher_id).all()
        course_list = []
        for course in courses:
            course_list.append({
                "course_id": course.course_id,
                "course_name": course.course_name,
                "student_count": course.students.count(),
                "session_count": AttendanceSession.query.filter_by(course_id=course.course_id).count()
            })

        # Get active sessions with attendance data
        active_sessions = AttendanceSession.query.filter_by(
            teacher_id=teacher_id,
            end_time=None
        ).all()

        session_list = []
        for session in active_sessions:
            present_count = Attendancelog.query.filter_by(
                session_id=session.id,
                status='present'
            ).count()
            
            session_list.append({
                "session_id": session.id,
                "course_id": session.course_id,
                "course_name": session.course.course_name,
                "session_number": session.session_number,
                "start_time": session.start_time.isoformat(),
                "students_present": present_count
            })

        return jsonify({
            "teacher_id": teacher.teacher_id,
            "name": teacher.name,
            "email": teacher.email,
            "total_courses": len(courses),
            "active_sessions": len(active_sessions),
            "courses": course_list,
            "active_sessions": session_list
        }), 200

    except Exception as e:
        logging.error(f"Profile error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500



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
