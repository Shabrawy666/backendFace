from flask import Blueprint, request, jsonify
from models import db, Attendancelog, AttendanceSession, Student, Course
from datetime import datetime
import pytz
from flask_jwt_extended import jwt_required, get_jwt_identity
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
@jwt_required()
def mark_attendance():
    try:
        data = request.get_json()
        student_id = get_jwt_identity()
        
        if not data.get('course_id'):
            return jsonify({"error": "Course ID is required"}), 400
        if not data.get('image_base64'):
            return jsonify({"error": "Face image is required"}), 400

        # Convert image and verify with ML service
        image = base64_to_image(data['image_base64'])
        verification = ml_service.verify_face(student_id, image)
        
        if not verification['success']:
            return jsonify({
                "error": "Attendance verification failed",
                "details": verification
            }), 400

        # Rest of your existing attendance marking logic
        course_id = data['course_id']
        student_ip = request.remote_addr
        local_time = datetime.now(pytz.timezone('Africa/Cairo'))

        session = AttendanceSession.query.filter_by(
            course_id=course_id
        ).order_by(AttendanceSession.session_number.desc()).first()

        if not session:
            return jsonify({"error": "No active session found for this course"}), 404

        existing_log = Attendancelog.query.filter_by(
            student_id=student_id,
            session_id=session.id
        ).first()

        if existing_log:
            return jsonify({
                "message": "Attendance already marked",
                "student_id": student_id,
                "session_id": session.id
            }), 200

        connection_strength = 'strong' if session.ip_address == student_ip else 'weak'

        new_log = Attendancelog(
            student_id=student_id,
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
            "verification": verification,
            "student_id": student_id,
            "course_id": course_id,
            "session_id": session.id,
            "timestamp": local_time.isoformat()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "error": "Failed to process attendance",
            "details": str(e)
        }), 500