from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy.orm import validates
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import Float
import re
import json
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='models.log'
)
logger = logging.getLogger(__name__)

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
    enrolled_courses = db.relationship(
        'Course',
        secondary=student_courses,
        backref=db.backref('enrolled_students', lazy='dynamic'),
        lazy='dynamic'
    )

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

class Course(db.Model):
    course_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    course_name = db.Column(db.String(255), nullable=False)
    sessions = db.Column(db.Integer)

    # Add foreign key to link each course to a teacher
    teacher_id = db.Column(db.String(11), db.ForeignKey('teacher.teacher_id'), nullable=True)
    
    # Relationship with teacher
    teacher = db.relationship('Teacher', back_populates='courses')

    def get_student_count(self):
        """Get total number of students in the course"""
        return self.enrolled_students.count()  # Changed from students to enrolled_students

    def get_all_students(self):
        """Get list of all students in the course"""
        return self.enrolled_students.all()  # Changed from students to enrolled_students

    def is_student_enrolled(self, student_id):
        """Check if a student is enrolled in the course"""
        return self.enrolled_students.filter_by(student_id=student_id).first() is not None  # Changed from students to enrolled_students

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
    status = db.Column(db.String(10), nullable=False)  # Remove default here

    # Verification fields
    marking_ip = db.Column(db.String(45), nullable=True)
    verification_score = db.Column(db.Float, nullable=True)
    liveness_score = db.Column(db.Float, nullable=True)
    verification_method = db.Column(db.String(20), default='face')
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
        result = db.session.query(cls).filter_by(
            student_id=student_id,
            session_id=session_id,
            course_id=course_id
        ).first()
        return bool(result)

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
        now = datetime.utcnow()
        
        # Set default values
        defaults = {
            'date': now.date(),
            'time': now.time(),
            'status': 'present',  # Default to present for new records
            'verification_details': {},
            'verification_timestamp': now,
            'last_attempt': now,
            'attempts_count': 1
        }
        
        # Update defaults with provided values
        defaults.update(kwargs)

        # Ensure status is valid
        if 'status' not in defaults:
            defaults['status'] = 'present'
        elif defaults['status'] not in ['present', 'absent']:
            defaults['status'] = 'present'
        
        # Initialize with updated values
        super(Attendancelog, self).__init__(**defaults)

    @classmethod
    def create_attendance(cls, student_id, course_id, session_id, teacher_id, **kwargs):
        """Factory method to create attendance record"""
        now = datetime.utcnow()
        
        # Set default values for attendance creation
        attendance_data = {
            'student_id': student_id,
            'course_id': course_id,
            'session_id': session_id,
            'teacher_id': teacher_id,
            'date': now.date(),
            'time': now.time(),
            'status': 'present',
            'verification_timestamp': now,
            'last_attempt': now,
            'attempts_count': 1,
            'verification_details': {}
        }
        
        # Update with provided values, but ensure status remains 'present'
        kwargs['status'] = 'present'  # Force status to be present
        attendance_data.update(kwargs)
        
        # Create new attendance record
        return cls(**attendance_data)


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
    
    # Session status tracking
    is_active = db.Column(db.Boolean, default=True)
    status = db.Column(db.String(20), default='ongoing')  # 'ongoing', 'completed', 'cancelled'
    
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
        """Get session statistics"""
        try:
            # Get total students directly from the course relationship
            total_students = self.course.enrolled_students.count()
            
            # Get attendance records
            attendance_records = db.session.query(Attendancelog).filter_by(
                session_id=self.id
            ).all()
            
            present_count = sum(1 for log in attendance_records if log.status == 'present')
            verified_count = sum(1 for log in attendance_records if log.connection_strength == 'strong')
            
            return {
                'total_students': total_students,
                'present_count': present_count,
                'absent_count': total_students - present_count,
                'verified_count': verified_count,
                'attendance_rate': float(present_count / total_students * 100) if total_students > 0 else 0.0,
                'verification_rate': float(verified_count / present_count * 100) if present_count > 0 else 0.0
            }
        except Exception as e:
            logger.error(f"Error calculating session stats: {str(e)}")
            return {
                'total_students': 0,
                'present_count': 0,
                'absent_count': 0,
                'verified_count': 0,
                'attendance_rate': 0.0,
                'verification_rate': 0.0
            }

    def __init__(self, **kwargs):
        super(AttendanceSession, self).__init__(**kwargs)
        if not self.start_time:
            self.start_time = datetime.utcnow()
