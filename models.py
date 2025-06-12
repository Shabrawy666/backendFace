from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy.orm import validates
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import Float
import re
import json
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime

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

class Attendancelog(db.Model):
    __tablename__ = 'attendancelog'

    # Primary key fields
    student_id = db.Column(db.String(11), db.ForeignKey('student.student_id'), primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.course_id'), primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_session.id'), primary_key=True)
    
    # Required fields
    teacher_id = db.Column(db.String(11), db.ForeignKey('teacher.teacher_id'), nullable=False)
    connection_strength = db.Column(db.String(20), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(10), default='absent')

    # Verification fields
    marking_ip = db.Column(db.String(45), nullable=True)
    verification_score = db.Column(db.Float, nullable=True)
    liveness_score = db.Column(db.Float, nullable=True)
    verification_method = db.Column(db.String(20), default='face')  # 'face', 'manual', 'system'
    verification_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Additional tracking
    attempts_count = db.Column(db.Integer, default=1)
    last_attempt = db.Column(db.DateTime, nullable=True)
    verification_details = db.Column(JSON, nullable=True)

    # Relationships
    teacher = db.relationship('Teacher', backref=db.backref('attendancelog', lazy=True))
    course = db.relationship('Course', backref=db.backref('attendancelog', lazy=True))
    student = db.relationship('Student', backref=db.backref('attendancelog', lazy=True))
    session = db.relationship('AttendanceSession', backref=db.backref('attendancelog', lazy=True))

    @classmethod
    def exists(cls, student_id, session_id, course_id):
        """Check if an attendance record exists"""
        return cls.query.filter_by(
            student_id=student_id,
            session_id=session_id,
            course_id=course_id
        ).first() is not None

    @property
    def is_verified(self):
        """Check if attendance is properly verified"""
        return (
            self.verification_score is not None and 
            self.verification_score > 0.8 and 
            self.liveness_score is not None and 
            self.liveness_score > 0.8
        )

    @property
    def verification_summary(self):
        """Get comprehensive verification details"""
        return {
            'method': self.verification_method,
            'score': self.verification_score,
            'liveness_score': self.liveness_score,
            'connection_strength': self.connection_strength,
            'is_verified': self.is_verified,
            'timestamp': self.verification_timestamp.isoformat() if self.verification_timestamp else None,
            'attempts': self.attempts_count,
            'last_attempt': self.last_attempt.isoformat() if self.last_attempt else None,
            'details': self.verification_details
        }

    def update_verification(self, verification_score, liveness_score, method='face'):
        """Update verification details"""
        self.verification_score = verification_score
        self.liveness_score = liveness_score
        self.verification_method = method
        self.verification_timestamp = datetime.utcnow()
        self.attempts_count += 1
        self.last_attempt = datetime.utcnow()

    def __init__(self, **kwargs):
        super(Attendancelog, self).__init__(**kwargs)
        if 'date' not in kwargs:
            self.date = datetime.utcnow().date()
        if 'time' not in kwargs:
            self.time = datetime.utcnow().time()
        if 'status' not in kwargs:
            self.status = 'absent'
        if 'verification_details' not in kwargs:
            self.verification_details = {}


class AttendanceSession(db.Model):
    __tablename__ = 'attendance_session'

    id = db.Column(db.Integer, primary_key=True)
    session_number = db.Column(db.Integer, nullable=False)
    teacher_id = db.Column(db.String(11), db.ForeignKey('teacher.teacher_id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.course_id'), nullable=False)
    ip_address = db.Column(db.String(45), nullable=False)
    
    # Session timestamps
    start_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)
    
    # Session status and tracking
    is_active = db.Column(db.Boolean, default=True)
    status = db.Column(db.String(20), default='ongoing')  # 'ongoing', 'completed', 'cancelled'
    verification_required = db.Column(db.Boolean, default=True)
    liveness_check_required = db.Column(db.Boolean, default=True)
    
    # Session metrics
    total_attempts = db.Column(db.Integer, default=0)
    successful_verifications = db.Column(db.Integer, default=0)
    failed_verifications = db.Column(db.Integer, default=0)
    
    # Relationships
    teacher = db.relationship('Teacher', backref='attendance_sessions')
    course = db.relationship('Course', backref='attendance_sessions')

    @property
    def duration(self):
        """Get session duration in minutes"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return None

    @property
    def session_stats(self):
        """Get comprehensive session statistics"""
        total_students = len(self.course.students.all())
        attendance_logs = list(self.attendancelog)
        present_count = len([log for log in attendance_logs if log.status == 'present'])
        verified_count = len([log for log in attendance_logs if log.is_verified])
        
        return {
            'total_students': total_students,
            'present_count': present_count,
            'absent_count': total_students - present_count,
            'verified_count': verified_count,
            'attendance_rate': (present_count / total_students * 100) if total_students > 0 else 0,
            'verification_rate': (verified_count / present_count * 100) if present_count > 0 else 0,
            'verification_metrics': {
                'total_attempts': self.total_attempts,
                'successful': self.successful_verifications,
                'failed': self.failed_verifications,
                'success_rate': (self.successful_verifications / self.total_attempts * 100) 
                               if self.total_attempts > 0 else 0
            }
        }

    def increment_verification_metrics(self, success: bool):
        """Update verification metrics"""
        self.total_attempts += 1
        if success:
            self.successful_verifications += 1
        else:
            self.failed_verifications += 1

    def __init__(self, **kwargs):
        super(AttendanceSession, self).__init__(**kwargs)
        if not self.start_time:
            self.start_time = datetime.utcnow()
