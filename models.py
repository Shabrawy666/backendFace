from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy.orm import validates
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import Float
import re
import json
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime  # Added for AttendanceSession

bcrypt = Bcrypt()
db = SQLAlchemy()

# Association table for student-course registration
student_courses = db.Table(
    'student_courses',
    db.Column('student_id', db.String(11), db.ForeignKey('student.student_id'), primary_key=True),
    db.Column('course_id', db.Integer, db.ForeignKey('course.course_id'), primary_key=True)
)

# Student model
class Student(db.Model):
    student_id = db.Column(db.String(11), primary_key=True, unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    _password = db.Column("password", db.String(255), nullable=False)
    face_encoding = db.Column(ARRAY(Float), nullable=True)
    email = db.Column(db.String(255), unique=True, nullable=False)

    # Many-to-many relationship with Course
    courses = db.relationship('Course', secondary=student_courses, backref='students', lazy='dynamic')

    @property
    def password(self):
        raise AttributeError("Password is not readable.")

    @password.setter
    def password(self, plaintext_password):
        if len(plaintext_password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in plaintext_password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in plaintext_password):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in plaintext_password):
            raise ValueError("Password must contain at least one digit")
        
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(plaintext_password.encode('utf-8'), salt)
        self._password = hashed_password.decode('utf-8')

    def check_password(self, plaintext_password):
        return bcrypt.checkpw(plaintext_password.encode('utf-8'), self._password.encode('utf-8'))

    @validates("email")
    def validate_email(self, key, email):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Invalid email format")
        return email
    
    @validates("student_id")
    def validate_student_id(self, key, student_id):
        if not re.match(r"^\d{11}$", student_id):
            raise ValueError("Student ID must be exactly 11 digits long.")
        return student_id

# Teacher model
class Teacher(db.Model):
    teacher_id = db.Column(db.String(11), primary_key=True, unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    _password = db.Column("password", db.String(255), nullable=False)

    # One-to-many relationship: a teacher can teach many courses
    courses = db.relationship('Course', back_populates='teacher')

    @property
    def password(self):
        raise AttributeError("Password is not readable.")

    @password.setter
    def password(self, plaintext_password):
        if len(plaintext_password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in plaintext_password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in plaintext_password):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in plaintext_password):
            raise ValueError("Password must contain at least one digit")
        
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(plaintext_password.encode('utf-8'), salt)
        self._password = hashed_password.decode('utf-8')

    def check_password(self, plaintext_password):
        return bcrypt.checkpw(plaintext_password.encode('utf-8'), self._password.encode('utf-8'))

# Course model
class Course(db.Model):
    course_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    course_name = db.Column(db.String(255), nullable=False)
    sessions = db.Column(db.Integer)

    # Add foreign key to link each course to a teacher
    teacher_id = db.Column(db.String(11), db.ForeignKey('teacher.teacher_id'), nullable=True)
    teacher = db.relationship('Teacher', back_populates='courses')

# Attendancelog model
class Attendancelog(db.Model):
    __tablename__ = 'attendancelog'

    student_id = db.Column(db.String(11), db.ForeignKey('student.student_id'), primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.course_id'), primary_key=True)
    session_id = db.Column(db.Integer, primary_key=True)  # Session number within the course
    connection_strength = db.Column(db.String(10), nullable=False)
    teacher_id = db.Column(db.String(11), db.ForeignKey('teacher.teacher_id'), nullable=False)
    date = db.Column(db.Date)
    time = db.Column(db.Time)
    status = db.Column(db.String(10), default='Absent')

    course = db.relationship('Course', backref=db.backref('attendancelog', lazy=True))
    teacher = db.relationship('Teacher', backref=db.backref('attendancelog', lazy=True))
    student = db.relationship('Student', backref=db.backref('attendancelog', lazy=True))

# AttendanceSession model
class AttendanceSession(db.Model):
    __tablename__ = 'attendance_session'

    id = db.Column(db.Integer, primary_key=True)
    session_number = db.Column(db.Integer)  # New column for session number
    teacher_id = db.Column(db.String(11), db.ForeignKey('teacher.teacher_id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.course_id'), nullable=False)
    ip_address = db.Column(db.String, nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)

    teacher = db.relationship('Teacher', backref='attendance_sessions')
    course = db.relationship('Course', backref='attendance_sessions')
