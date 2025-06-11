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
from routes.ml import ml_bp
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from routes import register_blueprints
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from ml_service import ml_service
import base64
import numpy as np
from PIL import Image
import io
import cv2

# Critical optimization for face recognition in cloud
os.environ['DISABLE_DLIB_AVX_INSTRUCTIONS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__)

# ===== OPEN CORS TO ALL DOMAINS ===== #
CORS(app)

# Load configuration from Config object
app.config.from_object(Config)

# JWT Configuration
app.config["JWT_SECRET_KEY"] = os.environ.get('JWT_SECRET_KEY', 'attendancebackend123')
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)  # Now this will work

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
register_blueprints(app)

# Register blueprints
app.register_blueprint(student_bp, name='student_api', url_prefix='/api/student')
app.register_blueprint(teacher_bp, name='teacher_api',url_prefix='/api/teacher')
app.register_blueprint(attendance_bp, name='attendance_api',url_prefix='/api/attendance')
app.register_blueprint(ml_bp, name='ml_api',url_prefix='/api/ml')

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.environ.get('FLASK_ENV') == 'production' else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def base64_to_image(base64_str):
    """Convert base64 string to numpy array"""
    try:
        header, encoded = base64_str.split(',', 1) if ',' in base64_str else ('', base64_str)
        image_data = base64.b64decode(encoded)
        image = np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))
        
        # Add validation and logging
        if image is None or image.size == 0:
            raise ValueError("Decoded image is empty")
            
        logger.info(f"Image decoded successfully. Shape: {image.shape}, dtype: {image.dtype}")
        return image

    except Exception as e:
        logger.error(f"Image conversion error: {str(e)}")
        raise ValueError(f"Invalid image format: {str(e)}")

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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        threaded=True,
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )

@app.route('/api/student/register-face-test', methods=['POST'])
@jwt_required()
def register_face_test():
    """Face registration without liveness check for testing"""
    try:
        student_id = get_jwt_identity()
        data = request.get_json()
        face_image = data.get('face_image')
        
        if not face_image:
            return jsonify({"error": "Face image is required"}), 400

        student = Student.query.get(student_id)
        if not student:
            return jsonify({"error": "Student not found"}), 404

        try:
            # Convert and validate image
            logging.info("Converting base64 to image...")
            image = base64_to_image(face_image)
            
            # Convert to BGR for OpenCV
            logging.info("Converting to BGR format...")
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                return jsonify({
                    "error": "Invalid image format",
                    "details": "Image must be in RGB format"
                }), 400
            
            # Check liveness first (with testing bypass)
            logging.info("Checking liveness...")
            liveness_result = ml_service.check_liveness(image_bgr)
            logging.info(f"Liveness result: {liveness_result}")
            
            if not liveness_result.get('live', False):
                return jsonify({
                    "error": "Liveness check failed",
                    "details": liveness_result.get('explanation', 'Could not verify live face')
                }), 400

            # Get face encoding with detailed logging
            logging.info("Getting face encoding...")
            encoding_result = ml_service.get_face_encoding(image_bgr)
            logging.info(f"Encoding result keys: {encoding_result.keys()}")
            logging.info(f"Encoding success: {encoding_result.get('success')}")
            logging.info(f"Encoding message: {encoding_result.get('message')}")
            
            if not encoding_result.get('success', False):
                return jsonify({
                    "error": "Face encoding failed",
                    "details": encoding_result.get('message', 'Could not generate face encoding'),
                    "debug_info": {
                        "encoding_result": encoding_result,
                        "image_shape": image.shape,
                        "image_dtype": str(image.dtype)
                    }
                }), 400

            # Store in database
            student.face_encoding = encoding_result.get('encoding')
            db.session.commit()

            return jsonify({
                "message": "Face registration successful",
                "student_id": student_id,
                "debug_info": {
                    "encoding_length": len(encoding_result.get('encoding', [])),
                    "image_shape": image.shape
                }
            }), 200

        except Exception as e:
            logging.error(f"Face processing error: {str(e)}")
            return jsonify({
                "error": "Face processing error",
                "details": str(e)
            }), 400

    except Exception as e:
        db.session.rollback()
        logging.error(f"Face registration error: {str(e)}")
        return jsonify({
            "error": "Face processing error",
            "details": str(e)
        }), 500