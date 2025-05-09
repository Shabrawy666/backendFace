from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student, Course
from datetime import datetime
import numpy as np
import cv2
from sqlalchemy import desc
import pytz
from deepface import DeepFace
import base64
import io
from PIL import Image
from flask_jwt_extended import jwt_required, get_jwt_identity

attendance_bp = Blueprint('attendance', __name__, url_prefix='/api/attendance')

# Initialize Facenet model
facenet_model = DeepFace.build_model("Facenet")

def get_local_time():
    utc_time = datetime.utcnow()
    egypt_timezone = pytz.timezone('Africa/Cairo')
    return utc_time.replace(tzinfo=pytz.utc).astimezone(egypt_timezone)

def base64_to_embedding(base64_image):
    try:
        # Decode base64 to numpy image
        header, encoded = base64_image.split(',', 1) if ',' in base64_image else ('', base64_image)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(image)
        
        # Resize and preprocess for Facenet
        img_resized = cv2.resize(img_np, (160, 160))
        img_preprocessed = img_resized.astype('float32') / 255.0
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
        
        # Get face embedding
        embedding = facenet_model.predict(img_preprocessed)[0]
        return embedding
        
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@attendance_bp.route('/mark', methods=['POST'])
@jwt_required()
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

        # Convert base64 to face embedding
        try:
            captured_embedding = base64_to_embedding(base64_image)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        # Get all students with face encodings
        students = Student.query.filter(Student.face_encoding.isnot(None)).all()

        # Find matching student
        matched_student = None
        for student in students:
            known_embedding = np.array(student.face_encoding)
            
            # Calculate Euclidean distance between embeddings
            distance = np.linalg.norm(known_embedding - captured_embedding)
            
            # Threshold for face recognition (adjust as needed)
            if distance < 10:  # Facenet typically uses threshold around 10
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
                "student_name": matched_student.name,
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

        # Get course name for response
        course = Course.query.get(course_id)
        course_name = course.course_name if course else "Unknown Course"

        return jsonify({
            "success": True,
            "message": "Attendance marked successfully",
            "student_id": matched_student.student_id,
            "student_name": matched_student.name,
            "course_id": course_id,
            "course_name": course_name,
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

@attendance_bp.route('/session_records', methods=['GET'])
@jwt_required()
def get_session_records():
    try:
        student_id = get_jwt_identity()
        course_id = request.args.get('course_id')
        
        if not course_id:
            return jsonify({"error": "Course ID is required"}), 400

        # Get all attendance records for this student in the specified course
        records = db.session.query(Attendancelog, AttendanceSession).join(
            AttendanceSession,
            Attendancelog.session_id == AttendanceSession.id
        ).filter(
            Attendancelog.student_id == student_id,
            Attendancelog.course_id == course_id
        ).order_by(AttendanceSession.session_number).all()

        attendance_records = []
        for record, session in records:
            attendance_records.append({
                "session_number": session.session_number,
                "date": record.date.strftime("%Y-%m-%d"),
                "time": record.time.strftime("%H:%M:%S"),
                "status": record.status,
                "connection_strength": record.connection_strength
            })

        return jsonify({
            "student_id": student_id,
            "course_id": course_id,
            "attendance_records": attendance_records
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500