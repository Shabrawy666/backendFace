from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from models import db, Student, Teacher, Course, Attendancelog
from config import Config
from sqlalchemy.exc import OperationalError
import logging
from routes.student import student_bp
from routes.teacher import teacher_bp
from routes.attendance import attendance_bp  # ✅ Import attendance blueprint
from flask_jwt_extended import JWTManager

# Initialize the Flask application
app = Flask(__name__)

# Load configuration from Config object
app.config.from_object(Config)

# JWT Configuration
app.config["JWT_SECRET_KEY"] = 'attendancebackend123'  # Use a secure key for production
jwt = JWTManager(app)  # Initialize JWT Manager

# Initialize the database with the Flask app
db.init_app(app)

# Initialize Bcrypt with the app
bcrypt = Bcrypt(app)

# Initialize Flask-Migrate with the app and database
migrate = Migrate(app, db)

# Register the Blueprints for routes
app.register_blueprint(student_bp)
app.register_blueprint(teacher_bp)
app.register_blueprint(attendance_bp)  # ✅ Register attendance blueprint

# Set up logging to a file
logging.basicConfig(
    filename='app_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    app.run(debug=True)