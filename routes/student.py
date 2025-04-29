from flask import Blueprint, request, jsonify
import bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from models import db, Student, Attendancelog, Course
import re
import cv2
import face_recognition
import numpy as np
from datetime import timedelta
from deepface import DeepFace

student_bp = Blueprint('student', __name__, url_prefix='/api/student')


def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def capture_face_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not access the webcam")

    print("Press SPACE to capture the image")
    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capture Face - Press SPACE", frame)
        key = cv2.waitKey(1)
        if key % 256 == 32:
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


@student_bp.route('/login', methods=['POST'])
def login_student():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        student = Student.query.filter_by(email=email).first()

        if not student or not bcrypt.checkpw(password.encode('utf-8'), student._password.encode('utf-8')):
            return jsonify({"error": "Invalid email or password"}), 401

        access_token = create_access_token(identity=student.student_id, expires_delta=timedelta(hours=1))

        if student.face_encoding:
            return jsonify({
                "message": "Login successful. Welcome back.",
                "student_id": student.student_id,
                "access_token": access_token
            }), 200

        frame = capture_face_image()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame)
        if not encodings:
            return jsonify({"error": "No face detected"}), 400

        face_encoding = encodings[0].tolist()
        student.face_encoding = face_encoding
        db.session.commit()

        return jsonify({
            "message": "Login successful. Face captured and saved.",
            "student_id": student.student_id,
            "access_token": access_token
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@student_bp.route('/capture_face', methods=['POST'])
def capture_face_from_camera():
    try:
        student_id = request.json.get('student_id')
        if not student_id:
            return jsonify({"error": "Student ID is required"}), 400

        student = Student.query.filter_by(student_id=student_id).first()
        if not student:
            return jsonify({"error": "Student not found"}), 404

        frame = capture_face_image()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame)
        if not encodings:
            return jsonify({"error": "No face detected"}), 400

        student.face_encoding = encodings[0].tolist()
        db.session.commit()

        return jsonify({"message": "Face captured from webcam and saved"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@student_bp.route('/attendance', methods=['GET'])
@jwt_required()
def get_student_attendance():
    try:
        student_id = get_jwt_identity()
        records = Attendancelog.query.filter_by(student_id=student_id).all()

        attendance_list = []
        for record in records:
            course = Course.query.get(record.course_id)
            attendance_list.append({
                "course_name": course.course_name if course else "Unknown",
                "date": record.date.strftime("%Y-%m-%d"),
                "time": record.time.strftime("%H:%M:%S"),
                "status": record.status
            })

        return jsonify({
            "student_id": student_id,
            "attendance_records": attendance_list
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
