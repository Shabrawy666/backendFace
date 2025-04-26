from flask import Flask
from .student import student_bp
from .attendance import attendance_bp  # Import the attendance blueprint
from .teacher import teacher_bp  # Import the teacher blueprint

def create_app():
    app = Flask(__name__)

    # Register the blueprints
    app.register_blueprint(student_bp)  # Register the student blueprint
    app.register_blueprint(attendance_bp)  # Register the attendance blueprint
    app.register_blueprint(teacher_bp)  # Register the teacher blueprint

    return app