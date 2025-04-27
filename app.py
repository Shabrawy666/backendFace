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

# Initialize the Flask application
app = Flask(__name__)

# Load configuration from Config object
app.config.from_object(Config)

# JWT Configuration - Use environment variable for production
app.config["JWT_SECRET_KEY"] = os.environ.get('JWT_SECRET_KEY', 'attendancebackend123')  # Use env var in production
jwt = JWTManager(app)

# Initialize the database with the Flask app
db.init_app(app)

# Initialize Bcrypt with the app
bcrypt = Bcrypt(app)

# Initialize Flask-Migrate with the app and database
migrate = Migrate(app, db)

# Register the Blueprints for routes
app.register_blueprint(student_bp)
app.register_blueprint(teacher_bp)
app.register_blueprint(attendance_bp)

# Set up logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# In production, configure proper logging to stdout
if os.environ.get('FLASK_ENV') == 'production':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

# Basic route for testing
@app.route('/')
def home():
    return "Attendance System is up and running!"

# Error handling for 404 - Page Not Found
@app.errorhandler(404)
def not_found_error(error):
    app.logger.error('404 Error: %s', error)
    return "Page not found! Please check the URL.", 404

# Error handling for 500 - Internal Server Error
@app.errorhandler(500)
def internal_error(error):
    app.logger.error('500 Error: %s', error)
    return "An internal error occurred. Please try again later.", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)