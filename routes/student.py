from flask import Blueprint, request, jsonify
import bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from models import db, Student, Attendancelog, Course, bcrypt
import re
import cv2
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

        if not student:
            return jsonify({"error": "Invalid email or password"}), 401

        # Use the check_password method from your Student model
        if not student.check_password(password):
            return jsonify({"error": "Invalid email or password"}), 401

        access_token = create_access_token(identity=student.student_id, expires_delta=timedelta(hours=1))

        # Prepare student data (excluding password)
        student_data = {
            "student_id": student.student_id,
            "name": student.name,
            "email": student.email,
            "face_encoding": student.face_encoding
        }

        response_data = {
            "message": "Login successful. Welcome back." if student.face_encoding 
                      else "Login successful. Please capture your face.",
            "access_token": access_token,
            "student_data": student_data,
            "needs_face_capture": not bool(student.face_encoding)
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
import base64
import io
from PIL import Image
from deepface.basemodels import Facenet

facenet_model = Facenet.loadModel()

@student_bp.route('/save_face', methods=['POST'])
@jwt_required()
def save_face():
    try:
        student_id = get_jwt_identity()
        student = Student.query.filter_by(student_id=student_id).first()
        
        if not student:
            return jsonify({"error": "Student not found"}), 404

        data = request.get_json()
        base64_image = data.get('face_image')

        if not base64_image:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 to numpy image
        header, encoded = base64_image.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_np = np.array(image)

        # Resize and preprocess as required by Facenet
        img_resized = cv2.resize(img_np, (160, 160))
        img_preprocessed = img_resized.astype('float32') / 255.0
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

        # Get face embedding
        embedding = facenet_model.predict(img_preprocessed)[0]

        # Save embedding to database
        student.face_encoding = embedding.tolist()
        db.session.commit()

        return jsonify({
            "message": "Face captured and saved successfully",
            "student_id": student.student_id
        }), 200

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

@student_bp.route('/me', methods=['GET'])
@jwt_required()
def get_student_profile():
    try:
        student_id = get_jwt_identity()
        student = Student.query.filter_by(student_id=student_id).first()
        
        if not student:
            return jsonify({"error": "Student not found"}), 404

        student_data = {
            column.name: getattr(student, column.name)
            for column in student.__table__.columns
            if column.name != '_password'
        }

        return jsonify(student_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500