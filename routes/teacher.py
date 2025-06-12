from flask import Blueprint, request, jsonify
from models import (
    db, 
    Attendancelog, 
    AttendanceSession, 
    Student, 
    Course, 
    Teacher, 
    student_courses  # Add this import
)
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

        # Modified course query
        courses = Course.query.filter_by(teacher_id=teacher.teacher_id).all()
        current_courses = []
        for course in courses:
            # Get students directly from the relationship
            students = [student for student in course.students]
            verified_students = sum(1 for student in students if student.face_encoding is not None)
            
            current_courses.append({
                "course_id": course.course_id,
                "course_name": course.course_name,
                "total_students": len(students),
                "verified_students": verified_students
            })

        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "teacher_data": {
                "teacher_id": teacher.teacher_id,
                "name": teacher.name,
                "courses": current_courses
            }
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

        # Get the last session number for this course
        last_session = AttendanceSession.query.filter_by(course_id=course_id).order_by(desc(AttendanceSession.session_number)).first()
        new_session_number = (last_session.session_number + 1) if last_session else 1

        # Create new session without specifying the ID
        session = AttendanceSession(
            course_id=course_id,
            teacher_id=teacher_id,
            ip_address=teacher_ip,
            start_time=local_time,
            session_number=new_session_number
        )

        try:
            db.session.add(session)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database error while creating session: {str(e)}")
            return jsonify({
                "error": "Failed to create session",
                "details": "Database error occurred"
            }), 500

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
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


@teacher_bp.route('/end_session', methods=['POST'])
@jwt_required()
def end_attendance_session():
    try:
        teacher_id = get_jwt_identity()
        data = request.get_json()
        course_id = int(data.get('course_id'))

        # Find active session
        session = AttendanceSession.query.filter_by(
            course_id=course_id,
            teacher_id=teacher_id,
            end_time=None
        ).first()

        if not session:
            return jsonify({"error": "No active session found"}), 404

        # Get course
        course = Course.query.get(course_id)
        if not course:
            return jsonify({"error": "Course not found"}), 404

        # Get registered students using the existing relationship
        registered_students = course.students.all()

        # Get attended students
        attended_students = Attendancelog.query.filter_by(
            session_id=session.id,
            course_id=course_id
        ).all()

        # Mark absent students
        current_time = datetime.utcnow()
        attended_ids = {a.student_id for a in attended_students}
        absent_count = 0

        for student in registered_students:
            if student.student_id not in attended_ids:
                absent_log = Attendancelog(
                    student_id=student.student_id,
                    course_id=course_id,
                    session_id=session.id,
                    teacher_id=teacher_id,
                    status='absent',
                    connection_strength='none',
                    date=current_time.date(),
                    time=current_time.time()
                )
                db.session.add(absent_log)
                absent_count += 1

        # End session
        session.end_time = current_time
        session.is_active = False
        session.status = 'completed'

        try:
            db.session.commit()
        except Exception as commit_error:
            db.session.rollback()
            raise commit_error

        # Calculate statistics
        total_students = len(registered_students)
        present_count = len(attended_students)
        face_verified = sum(1 for record in attended_students if record.connection_strength == 'strong')

        return jsonify({
            "message": "Attendance session ended successfully",
            "session_id": str(session.id),
            "session_number": session.session_number,
            "course_name": course.course_name,
            "end_time": current_time.isoformat(),
            "session_stats": {
                "total_registered": total_students,
                "students_present": present_count,
                "students_absent": absent_count,
                "attendance_rate": float(present_count / total_students * 100) if total_students > 0 else 0.0,
                "face_verified": face_verified,
                "verification_rate": float(face_verified / present_count * 100) if present_count > 0 else 0.0
            }
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"End session error: {str(e)}")
        return jsonify({
            "error": "An internal error occurred",
            "details": str(e)
        }), 500

# Optional: Add endpoint to get session status during active session
@teacher_bp.route('/session/<int:session_id>/live-status', methods=['GET'])
@jwt_required()
def get_live_session_status(session_id):
    """Get real-time status of an active session"""
    try:
        teacher_id = get_jwt_identity()
        
        session = AttendanceSession.query.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        if not teacher_owns_course(teacher_id, session.course_id):
            return jsonify({"error": "Unauthorized access to session"}), 403
        
        # Get course and registered students
        course = Course.query.get(session.course_id)
        total_registered = course.students.count()
        
        # Get current attendance
        current_attendance = Attendancelog.query.filter_by(
            session_id=session_id
        ).all()
        
        present_count = len(current_attendance)
        face_verified = len([record for record in current_attendance if record.connection_strength == 'strong'])
        
        # Get list of students who have marked attendance
        attended_students = []
        for record in current_attendance:
            student = Student.query.get(record.student_id)
            attended_students.append({
                "student_id": record.student_id,
                "student_name": student.name if student else "Unknown",
                "marked_time": record.time.isoformat() if record.time else None,
                "connection_strength": record.connection_strength,
                "is_face_verified": record.connection_strength == 'strong'
            })
        
        return jsonify({
            "session_id": session_id,
            "session_number": session.session_number,
            "course_name": course.course_name,
            "is_active": session.end_time is None,
            "start_time": session.start_time.isoformat(),
            "live_stats": {
                "total_registered": total_registered,
                "currently_present": present_count,
                "still_absent": total_registered - present_count,
                "attendance_rate": (present_count / total_registered * 100) if total_registered > 0 else 0,
                "face_verified": face_verified,
                "verification_rate": (face_verified / present_count * 100) if present_count > 0 else 0
            },
            "attended_students": attended_students
        }), 200
        
    except Exception as e:
        logger.error(f"Get live session status error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

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
        course_id = request.args.get('course_id')  # Get course_id from query parameters

        # If course_id is provided, get stats for specific course
        if course_id:
            course = Course.query.filter_by(
                teacher_id=teacher_id,
                course_id=course_id
            ).first()

            if not course:
                return jsonify({
                    "error": "Course not found or unauthorized"
                }), 404

            course_logs = Attendancelog.query.filter_by(course_id=course_id).all()
            total_attempts = len(course_logs)
            face_verified = len([log for log in course_logs if log.connection_strength == 'strong'])
            
            stats = {
                course_id: {
                    "course_name": course.course_name,
                    "total_attendance_records": total_attempts,
                    "face_verified_count": face_verified,
                    "face_verification_rate": (face_verified / total_attempts * 100) if total_attempts > 0 else 0
                }
            }

        # If no course_id, get stats for all courses
        else:
            courses = Course.query.filter_by(teacher_id=teacher_id).all()
            if not courses:
                return jsonify({
                    "message": "No courses found for this teacher",
                    "verification_stats": {},
                    "system_metrics": ml_service.get_performance_metrics()
                }), 200
            
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

        # Add detailed stats for face verification
        system_metrics = ml_service.get_performance_metrics()
        
        return jsonify({
            "verification_stats": stats,
            "system_metrics": system_metrics,
            "summary": {
                "total_courses": len(stats),
                "total_records": sum(s["total_attendance_records"] for s in stats.values()),
                "total_verified": sum(s["face_verified_count"] for s in stats.values()),
                "average_verification_rate": sum(s["face_verification_rate"] for s in stats.values()) / len(stats) if stats else 0
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Verification stats error: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve verification stats",
            "details": str(e)
        }), 500
    
    # Add these endpoints to your existing teacher.py file

@teacher_bp.route('/course/<int:course_id>/sessions', methods=['GET'])
@jwt_required()
def get_course_sessions(course_id):
    """Get all sessions for a specific course"""
    try:
        teacher_id = get_jwt_identity()
        
        # Check if teacher owns this course
        if not teacher_owns_course(teacher_id, course_id):
            return jsonify({"error": "Unauthorized access to course"}), 403
        
        course = Course.query.get(course_id)
        if not course:
            return jsonify({"error": "Course not found"}), 404
        
        # Get all sessions for this course
        sessions = AttendanceSession.query.filter_by(course_id=course_id).order_by(desc(AttendanceSession.session_number)).all()
        
        session_list = []
        for session in sessions:
            # Get attendance stats for each session
            total_registered = course.students.count()
            present_count = Attendancelog.query.filter_by(
                session_id=session.id,
                status='present'
            ).count()
            absent_count = total_registered - present_count
            
            # Get face verification count
            face_verified = Attendancelog.query.filter_by(
                session_id=session.id,
                connection_strength='strong'
            ).count()
            
            session_list.append({
                "session_id": session.id,
                "session_number": session.session_number,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "is_active": session.end_time is None,
                "total_registered": total_registered,
                "students_present": present_count,
                "students_absent": absent_count,
                "face_verified_count": face_verified,
                "attendance_rate": (present_count / total_registered * 100) if total_registered > 0 else 0
            })
        
        return jsonify({
            "course_id": course_id,
            "course_name": course.course_name,
            "total_sessions": len(sessions),
            "sessions": session_list
        }), 200
        
    except Exception as e:
        logger.error(f"Get course sessions error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@teacher_bp.route('/course/<int:course_id>/students', methods=['GET'])
@jwt_required()
def get_course_students(course_id):
    """Get all students registered in a specific course"""
    try:
        teacher_id = get_jwt_identity()
        
        # Check if teacher owns this course
        if not teacher_owns_course(teacher_id, course_id):
            return jsonify({"error": "Unauthorized access to course"}), 403
        
        course = Course.query.get(course_id)
        if not course:
            return jsonify({"error": "Course not found"}), 404
        
        # Get all students in this course
        students = course.students.all()
        
        student_list = []
        for student in students:
            # Get student's overall attendance stats for this course
            total_sessions = AttendanceSession.query.filter_by(course_id=course_id).count()
            attended_sessions = Attendancelog.query.filter_by(
                student_id=student.student_id,
                course_id=course_id,
                status='present'
            ).count()
            
            # Get face verification stats
            face_verified_sessions = Attendancelog.query.filter_by(
                student_id=student.student_id,
                course_id=course_id,
                connection_strength='strong'
            ).count()
            
            student_list.append({
                "student_id": student.student_id,
                "name": student.name,
                "email": student.email,
                "has_face_encoding": student.face_encoding is not None,
                "total_sessions": total_sessions,
                "attended_sessions": attended_sessions,
                "missed_sessions": total_sessions - attended_sessions,
                "attendance_rate": (attended_sessions / total_sessions * 100) if total_sessions > 0 else 0,
                "face_verified_sessions": face_verified_sessions,
                "face_verification_rate": (face_verified_sessions / attended_sessions * 100) if attended_sessions > 0 else 0
            })
        
        return jsonify({
            "course_id": course_id,
            "course_name": course.course_name,
            "total_students": len(students),
            "students": student_list
        }), 200
        
    except Exception as e:
        logger.error(f"Get course students error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@teacher_bp.route('/session/<int:session_id>/attendance', methods=['GET'])
@jwt_required()
def get_session_attendance(session_id):
    """Get attendance details for a specific session"""
    try:
        teacher_id = get_jwt_identity()
        
        # Get session and verify teacher ownership
        session = AttendanceSession.query.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        if not teacher_owns_course(teacher_id, session.course_id):
            return jsonify({"error": "Unauthorized access to session"}), 403
        
        # Get all students registered in the course
        course = Course.query.get(session.course_id)
        registered_students = course.students.all()
        
        # Get attendance records for this session
        attendance_records = Attendancelog.query.filter_by(session_id=session_id).all()
        attendance_dict = {record.student_id: record for record in attendance_records}
        
        student_attendance = []
        for student in registered_students:
            attendance_record = attendance_dict.get(student.student_id)
            
            if attendance_record:
                student_attendance.append({
                    "student_id": student.student_id,
                    "name": student.name,
                    "email": student.email,
                    "status": attendance_record.status,
                    "connection_strength": attendance_record.connection_strength,
                    "date": attendance_record.date.isoformat() if attendance_record.date else None,
                    "time": attendance_record.time.isoformat() if attendance_record.time else None,
                    "is_face_verified": attendance_record.connection_strength == 'strong'
                })
            else:
                # Student was absent (no attendance record)
                student_attendance.append({
                    "student_id": student.student_id,
                    "name": student.name,
                    "email": student.email,
                    "status": "absent",
                    "connection_strength": None,
                    "date": None,
                    "time": None,
                    "is_face_verified": False
                })
        
        # Calculate summary stats
        present_count = len([s for s in student_attendance if s["status"] == "present"])
        absent_count = len(registered_students) - present_count
        face_verified_count = len([s for s in student_attendance if s["is_face_verified"]])
        
        return jsonify({
            "session_id": session_id,
            "session_number": session.session_number,
            "course_id": session.course_id,
            "course_name": course.course_name,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "is_active": session.end_time is None,
            "summary": {
                "total_registered": len(registered_students),
                "present": present_count,
                "absent": absent_count,
                "face_verified": face_verified_count,
                "attendance_rate": (present_count / len(registered_students) * 100) if registered_students else 0
            },
            "attendance": student_attendance
        }), 200
        
    except Exception as e:
        logger.error(f"Get session attendance error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@teacher_bp.route('/attendance/edit', methods=['POST'])
@jwt_required()
def edit_attendance():
    """Manually edit attendance record for a student in a specific session"""
    try:
        teacher_id = get_jwt_identity()
        data = request.get_json()
        
        session_id = data.get('session_id')
        student_id = data.get('student_id')
        new_status = data.get('status')  # 'present' or 'absent'
        
        if not all([session_id, student_id, new_status]):
            return jsonify({"error": "Session ID, student ID, and status are required"}), 400
        
        if new_status not in ['present', 'absent']:
            return jsonify({"error": "Status must be 'present' or 'absent'"}), 400
        
        # Verify session exists and teacher owns the course
        session = AttendanceSession.query.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        if not teacher_owns_course(teacher_id, session.course_id):
            return jsonify({"error": "Unauthorized access to session"}), 403
        
        # Verify student is registered in the course
        student = Student.query.get(student_id)
        if not student:
            return jsonify({"error": "Student not found"}), 404
        
        course = Course.query.get(session.course_id)
        if student not in course.students:
            return jsonify({"error": "Student is not registered in this course"}), 400
        
        # Check if attendance record exists
        existing_record = Attendancelog.query.filter_by(
            session_id=session_id,
            student_id=student_id,
            course_id=session.course_id
        ).first()
        
        current_time = datetime.now()
        
        if new_status == 'present':
            if existing_record:
                # Update existing record
                existing_record.status = 'present'
                existing_record.connection_strength = 'manual_edit'
                existing_record.date = current_time.date()
                existing_record.time = current_time.time()
            else:
                # Create new attendance record
                new_record = Attendancelog(
                    student_id=student_id,
                    course_id=session.course_id,
                    session_id=session_id,
                    status='present',
                    connection_strength='manual_edit',
                    date=current_time.date(),
                    time=current_time.time()
                )
                db.session.add(new_record)
        
        else:  # new_status == 'absent'
            if existing_record:
                # Delete the attendance record (absent means no record)
                db.session.delete(existing_record)
            # If no existing record, student is already marked as absent
        
        db.session.commit()
        
        return jsonify({
            "message": f"Attendance updated successfully",
            "student_id": student_id,
            "student_name": student.name,
            "session_id": session_id,
            "session_number": session.session_number,
            "new_status": new_status,
            "edited_by": "teacher_manual",
            "edit_time": current_time.isoformat()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Edit attendance error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@teacher_bp.route('/course/<int:course_id>/attendance/bulk-edit', methods=['POST'])
@jwt_required()
def bulk_edit_attendance():
    """Bulk edit attendance for multiple students in a session"""
    try:
        teacher_id = get_jwt_identity()
        data = request.get_json()
        
        session_id = data.get('session_id')
        attendance_updates = data.get('attendance_updates')  # List of {student_id, status}
        
        if not session_id or not attendance_updates:
            return jsonify({"error": "Session ID and attendance updates are required"}), 400
        
        # Verify session and teacher ownership
        session = AttendanceSession.query.get(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        if not teacher_owns_course(teacher_id, session.course_id):
            return jsonify({"error": "Unauthorized access to session"}), 403
        
        updated_records = []
        errors = []
        
        for update in attendance_updates:
            student_id = update.get('student_id')
            new_status = update.get('status')
            
            if not student_id or new_status not in ['present', 'absent']:
                errors.append(f"Invalid data for student {student_id}")
                continue
            
            try:
                # Verify student is in course
                student = Student.query.get(student_id)
                course = Course.query.get(session.course_id)
                
                if not student or student not in course.students:
                    errors.append(f"Student {student_id} not found in course")
                    continue
                
                # Update attendance
                existing_record = Attendancelog.query.filter_by(
                    session_id=session_id,
                    student_id=student_id,
                    course_id=session.course_id
                ).first()
                
                current_time = datetime.now()
                
                if new_status == 'present':
                    if existing_record:
                        existing_record.status = 'present'
                        existing_record.connection_strength = 'manual_edit'
                        existing_record.date = current_time.date()
                        existing_record.time = current_time.time()
                    else:
                        new_record = Attendancelog(
                            student_id=student_id,
                            course_id=session.course_id,
                            session_id=session_id,
                            status='present',
                            connection_strength='manual_edit',
                            date=current_time.date(),
                            time=current_time.time()
                        )
                        db.session.add(new_record)
                else:  # absent
                    if existing_record:
                        db.session.delete(existing_record)
                
                updated_records.append({
                    "student_id": student_id,
                    "student_name": student.name,
                    "new_status": new_status
                })
                
            except Exception as e:
                errors.append(f"Error updating student {student_id}: {str(e)}")
        
        db.session.commit()
        
        return jsonify({
            "message": f"Bulk attendance update completed",
            "session_id": session_id,
            "session_number": session.session_number,
            "updated_count": len(updated_records),
            "updated_records": updated_records,
            "errors": errors,
            "edit_time": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Bulk edit attendance error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500