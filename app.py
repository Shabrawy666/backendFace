from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from models import db, Student, Teacher, Course, Attendancelog
from config import Config
from sqlalchemy.exc import OperationalError
import logging
import os
from routes.student import student_bp
from routes.teacher import teacher_bp
from routes.attendance import attendance_bp
from flask_jwt_extended import JWTManager

# Critical optimization for face recognition in cloud
os.environ['DISABLE_DLIB_AVX_INSTRUCTIONS'] = '1'  # Avoids CPU compatibility issues
os.environ['OMP_NUM_THREADS'] = '1'  # Prevents over-subscription in cloud environment

# Initialize the Flask application
app = Flask(__name__)

# Load configuration from Config object
app.config.from_object(Config)

# Enhanced JWT Configuration
app.config["JWT_SECRET_KEY"] = os.environ.get('JWT_SECRET_KEY', 'attendancebackend123')
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)  # Token expiration
jwt = JWTManager(app)

# Database initialization with connection pool settings for Railway
app.config.update({
    'SQLALCHEMY_ENGINE_OPTIONS': {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 5,
        'max_overflow': 10
    }
})
db.init_app(app)

# Initialize Bcrypt with the app
bcrypt = Bcrypt(app)

# Initialize Flask-Migrate with the app and database
migrate = Migrate(app, db)

# Register the Blueprints for routes
app.register_blueprint(student_bp, url_prefix='/api/student')
app.register_blueprint(teacher_bp, url_prefix='/api/teacher')
app.register_blueprint(attendance_bp, url_prefix='/api/attendance')

# Production-optimized logging
logging.basicConfig(
    level=logging.INFO if os.environ.get('FLASK_ENV') == 'production' else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Better for cloud logging
)

# Health check endpoint for Railway monitoring
@app.route('/health')
def health_check():
    try:
        db.session.execute('SELECT 1')
        return 'OK', 200
    except Exception as e:
        app.logger.error(f'Health check failed: {str(e)}')
        return 'Service Unavailable', 503

# Basic route for testing
@app.route('/')
def home():
    return "Attendance System is up and running!"

# Enhanced error handlers
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
        threaded=True,  # Important for face recognition endpoints
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )