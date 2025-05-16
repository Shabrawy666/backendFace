from flask import Blueprint
from .student import student_bp
from .attendance import attendance_bp
from .teacher import teacher_bp
from .ml import ml_bp

def register_blueprints(app):
    # Register the blueprints
    app.register_blueprint(student_bp, url_prefix='/api/student')
    app.register_blueprint(attendance_bp, url_prefix='/api/attendance')
    app.register_blueprint(teacher_bp, url_prefix='/api/teacher')
    app.register_blueprint(ml_bp, url_prefix='/api/ml')