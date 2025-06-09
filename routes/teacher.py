from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student, Teacher, Course
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta, datetime
import logging
import bcrypt
from sqlalchemy import desc
from ml_service import ml_service
import base64
import io
import numpy as np
from PIL import Image

teacher_bp = Blueprint('teacher', __name__, url_prefix='/api/teacher')

# Enhanced logging setup
logging.basicConfig(
    filename='teacher_routes.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def base64_to_image(base64_str):
    """Convert base64 string to numpy array"""
    try:
        header, encoded = base64_str.split(',', 1) if ',' in base64_str else ('', base64_str)
        image_data = base64.b64decode(encoded)
        return np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))
    except Exception as e:
        logger.error(f"Image conversion error: {str(e)}")
        raise ValueError("Invalid image format")

# Helper function to check course ownership
def teacher_owns_course(teacher_id, course_id):
    course = Course.query.get(course_id)
    return course is not None and str(course.teacher_id) == str(teacher_id)

@teacher_bp.route('/login', methods=['POST'])
def login_teacher():
    """Teacher login endpoint using teacher id and password"""
    try:
        data = request.get_json()
        teacher_id = str(data.get('teacher_id'))
        password = data.get('password')

        if not teacher_id or not password:
            return jsonify({"error": "Teacher ID and password are required"}), 400

        teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()

        if not teacher or not bcrypt.checkpw(password.encode('utf-8'), teacher._password.encode('utf-8')):
            return jsonify({"error": "Invalid teacher ID or password"}), 401

        access_token = create_access_token(
            identity=teacher.teacher_id,
            expires_delta=timedelta(hours=1),
            additional_claims={"role": "teacher"}
        )

        courses = Course.query.filter_by(teacher_id=teacher.teacher_id).all()
        current_courses = []
        for course in courses:
            # Using len() for simplicity
            students = course.students.all()
            verified_students = len([s for s in students if s.face_encoding is not None])
            
            current_courses.append({
                "course_id": course.course_id,
                "course_name": course.course_name,
                "total_students": len(students),
                "verified_students": verified_students
            })

        teacher_data = {
            "teacher_id": teacher.teacher_id,
            "name": teacher.name,
            "courses": current_courses
        }

        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "teacher_data": teacher_data
        }), 200

    except Exception as e:
        logger.error(f"Teacher login error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@teacher_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_teacher_profile():
    """Get complete teacher profile with courses and sessions"""
    try:
        teacher_id = get_jwt_identity()
        teacher = Teacher.query.get(teacher_id)
        
        if not teacher:
            return jsonify({"error": "Teacher not found"}), 404

        courses = Course.query.filter_by(teacher_id=teacher_id).all()
        course_list = []
        for course in courses:
            total_students = course.students.count()
            verified_students = course.students.filter(Student.face_encoding.isnot(None)).count()
            course_list.append({
                "course_id": course.course_id,
                "course_name": course.course_name,
                "student_count": total_students,
                "verified_students": verified_students,
                "session_count": AttendanceSession.query.filter_by(course_id=course.course_id).count()
            })

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
            
            face_verified = Attendancelog.query.filter_by(
                session_id=session.id,
                connection_strength='strong'
            ).count()
            
            session_list.append({
                "session_id": session.id,
                "course_id": session.course_id,
                "course_name": session.course.course_name,
                "session_number": session.session_number,
                "start_time": session.start_time.isoformat(),
                "students_present": present_count,
                "face_verified_count": face_verified
            })

        # Get ML system status
        ml_metrics = ml_service.get_performance_metrics()
        ml_status = {
            "system_operational": True,
            "face_recognition_active": ml_service.deepface_available,
            "performance_metrics": ml_metrics
        }

        return jsonify({
            "teacher_id": teacher.teacher_id,
            "name": teacher.name,
            "email": teacher.email,
            "total_courses": len(courses),
            "active_sessions": len(active_sessions),
            "courses": course_list,
            "active_sessions": session_list,
            "ml_system_status": ml_status
        }), 200

    except Exception as e:
        logger.error(f"Profile error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@teacher_bp.route('/start_session', methods=['POST'])
@jwt_required()
def start_attendance_session():
    """Start a new attendance session for a course"""
    try:
        teacher_id = get_jwt_identity()
        data = request.get_json()
        course_id = data.get('course_id')
        
        if not course_id:
            return jsonify({"error": "Course ID is required"}), 400

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

        # Get ML system status for session start
        ml_status = {
            "system_ready": True,
            "face_recognition_active": ml_service.deepface_available
        }

        return jsonify({
            "message": "Attendance session started",
            "session_id": session.id,
            "session_number": new_session_number,
            "ip_address": teacher_ip,
            "ml_status": ml_status
        }), 201

    except Exception as e:
        logger.error(f"Start session error: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@teacher_bp.route('/end_session', methods=['POST'])
@jwt_required()
def end_attendance_session():
    """End an active attendance session"""
    try:
        teacher_id = get_jwt_identity()
        data = request.get_json()
        course_id = data.get('course_id')

        if not course_id:
            return jsonify({"error": "Course ID is required"}), 400

        if not teacher_owns_course(teacher_id, course_id):
            return jsonify({"error": "Unauthorized access to course"}), 403

        session = AttendanceSession.query.filter_by(
            course_id=course_id,
            end_time=None
        ).order_by(desc(AttendanceSession.id)).first()

        if not session:
            return jsonify({"error": "No active session found"}), 404

        session.end_time = datetime.utcnow()
        db.session.commit()

        # Get session statistics
        total_attendance = Attendancelog.query.filter_by(session_id=session.id).count()
        face_verified = Attendancelog.query.filter_by(
            session_id=session.id,
            connection_strength='strong'
        ).count()

        return jsonify({
            "message": "Attendance session ended",
            "session_id": session.id,
            "session_stats": {
                "total_attendance": total_attendance,
                "face_verified": face_verified,
                "verification_rate": (face_verified / total_attendance * 100) if total_attendance > 0 else 0
            }
        }), 200

    except Exception as e:
        logger.error(f"End session error: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

@teacher_bp.route('/verify_student', methods=['POST'])
@jwt_required()
def verify_student():
    """Verify a student's face manually"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        face_image = data.get('face_image')
        
        if not student_id or not face_image:
            return jsonify({"error": "Student ID and face image are required"}), 400
            
        # Convert and preprocess image
        image = base64_to_image(face_image)
        preprocessed = ml_service.preprocess_image(image)
        
        if preprocessed is None:
            return jsonify({
                "error": "Image preprocessing failed",
                "details": "Could not process face image"
            }), 400
            
        # Check liveness
        liveness_result = ml_service.verify_liveness(preprocessed)
        if not liveness_result['live']:
            return jsonify({
                "error": "Liveness check failed",
                "details": liveness_result['explanation']
            }), 400
            
        # Verify face
        verification_result = ml_service.verify_face(student_id, preprocessed)
        
        return jsonify({
            "success": verification_result['success'],
            "confidence_score": verification_result.get('confidence_score', 0),
            "liveness_score": liveness_result['score'],
            "verification_time": verification_result.get('verification_time', 0)
        }), 200
        
    except Exception as e:
        logger.error(f"Student verification error: {str(e)}")
        return jsonify({"error": "Verification failed"}), 500

@teacher_bp.route('/ml/metrics', methods=['GET'])
@jwt_required()
def get_ml_metrics():
    """Get ML system performance metrics"""
    try:
        metrics = ml_service.get_performance_metrics()
        return jsonify({
            "metrics": metrics,
            "system_status": {
                "operational": True,
                "deepface_available": ml_service.deepface_available
            }
        }), 200
    except Exception as e:
        logger.error(f"ML metrics error: {str(e)}")
        return jsonify({"error": "Failed to retrieve ML metrics"}), 500

@teacher_bp.route('/verification_stats', methods=['GET'])
@jwt_required()
def get_verification_stats():
    """Get statistics about face verification success rates"""
    try:
        teacher_id = get_jwt_identity()
        courses = Course.query.filter_by(teacher_id=teacher_id).all()
        
        stats = {}
        for course in courses:
            course_logs = Attendancelog.query.filter_by(course_id=course.course_id).all()
            total_attempts = len(course_logs)
            face_verified = len([log for log in course_logs if log.connection_strength == 'strong'])
            
            stats[course.course_id] = {
                "course_name": course.course_name,
                "total_attendance_records": total_attempts,
                "face_verified_count": face_verified,
                "face_verification_rate": (face_verified / total_attempts * 100) if total_attempts > 0 else 0
            }
            
        return jsonify({
            "verification_stats": stats,
            "system_metrics": ml_service.get_performance_metrics()
        }), 200
        
    except Exception as e:
        logger.error(f"Verification stats error: {str(e)}")
        return jsonify({"error": "Failed to retrieve verification stats"}), 500