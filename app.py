from flask import Flask, jsonify, request
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from models import db, Student, Teacher, Course, Attendancelog
from config import Config
from sqlalchemy.exc import OperationalError
import logging
import os
from datetime import timedelta
from routes.student import student_bp
from routes.teacher import teacher_bp
from routes.attendance import attendance_bp
from flask_jwt_extended import JWTManager
from flask_cors import CORS

# For image processing and face encoding
import face_recognition
import numpy as np
import base64
import io
from PIL import Image

# Critical optimization for face recognition in cloud
os.environ['DISABLE_DLIB_AVX_INSTRUCTIONS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Initialize the Flask application
app = Flask(__name__)

# ===== OPEN CORS TO ALL DOMAINS ===== #
CORS(app)

# Load configuration from Config object
app.config.from_object(Config)

# JWT Configuration
app.config["JWT_SECRET_KEY"] = os.environ.get('JWT_SECRET_KEY', 'attendancebackend123')
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)

# Database configuration
app.config.update({
    'SQLALCHEMY_ENGINE_OPTIONS': {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 5,
        'max_overflow': 10
    }
})

# Initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
migrate = Migrate(app, db)
jwt = JWTManager(app)

# Register blueprints
app.register_blueprint(student_bp, url_prefix='/api/student')
app.register_blueprint(teacher_bp, url_prefix='/api/teacher')
app.register_blueprint(attendance_bp, url_prefix='/api/attendance')

# Logging configuration
logging.basicConfig(
    level=logging.INFO if os.environ.get('FLASK_ENV') == 'production' else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

@app.route('/health')
def health_check():
    try:
        db.session.execute('SELECT 1')
        return 'OK', 200
    except Exception as e:
        app.logger.error(f'Health check failed: {str(e)}')
        return 'Service Unavailable', 503

@app.route('/')
def home():
    return "Attendance System is up and running!"

@app.errorhandler(404)
def not_found_error(error):
    app.logger.error('404 Error: %s', error)
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error('500 Error: %s', error)
    return jsonify({"error": "Internal server error"}), 500

# ===============================================
# 🧠 NEW ROUTE: Convert base64 image to embedding
# ===============================================
@app.route('/api/face-encode', methods=['POST'])
def face_encode():
    data = request.get_json()
    base64_image = data.get("image")

    if not base64_image:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode base64 and remove metadata prefix if present
        img_data = base64.b64decode(base64_image.split(",")[-1])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        image_np = np.array(image)

        # Detect faces and generate encoding
        face_locations = face_recognition.face_locations(image_np)
        if not face_locations:
            return jsonify({"error": "No face detected"}), 400

        encodings = face_recognition.face_encodings(image_np, face_locations)
        return jsonify({"embedding": encodings[0].tolist()}), 200

    except Exception as e:
        app.logger.error(f"Encoding error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ===============================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        threaded=True,
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )
