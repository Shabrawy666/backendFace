from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from datetime import datetime
from enum import Enum

@dataclass
class RecognitionResult:
    """Enhanced recognition result structure"""
    def __init__(
        self,
        success: bool,
        confidence_score: float = None,
        verification_time: float = None,
        verification_type: str = None,
        error_message: str = None,
        data: dict = None
    ):
        self.success = success
        self.confidence_score = confidence_score
        self.verification_time = verification_time
        self.verification_type = verification_type
        self.error_message = error_message
        self.data = data or {}

    def to_dict(self) -> dict:
        """Convert RecognitionResult to dictionary"""
        return {
            "success": bool(self.success),  # Ensure it's a Python bool
            "confidence_score": float(self.confidence_score) if self.confidence_score is not None else None,
            "verification_time": float(self.verification_time) if self.verification_time is not None else None,
            "verification_type": str(self.verification_type) if self.verification_type else None,
            "error_message": str(self.error_message) if self.error_message else None,
            "data": dict(self.data) if self.data else {}
        }

class UserRole(Enum):
    """Defines user roles in the system"""
    TEACHER = "teacher"
    STUDENT = "student"

class AttendanceStatus(Enum):
    """Defines possible attendance statuses"""
    PRESENT = "present"
    ABSENT = "absent"
    PENDING = "pending"
    MANUALLY_MARKED = "manually_marked"

class SessionStatus(Enum):
    """Defines possible session statuses"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"

@dataclass
class Session:
    """Attendance session details"""
    teacher_id: str
    hall_id: str
    start_time: str
    teacher_ip: str
    status: SessionStatus = SessionStatus.ACTIVE
    is_active: bool = True
    wifi_ssid: Optional[str] = None
    rssi_threshold: Optional[float] = None
    course_id: Optional[str] = None
    id: Optional[str] = None
    end_time: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: Optional[str] = None
    connected_students: Set[str] = field(default_factory=set)
    attendance_records: Dict[str, Dict] = field(default_factory=dict)

@dataclass
class AttendanceRecord:
    """Individual attendance record"""
    id: str
    session_id: str
    student_id: str
    timestamp: str
    status: AttendanceStatus
    verification_details: Dict = field(default_factory=dict)
    modified_by: Optional[str] = None
    modification_reason: Optional[str] = None
    notification_sent: bool = False

@dataclass
class WifiSession:
    """Stores WiFi session information for attendance verification"""
    session_id: str
    teacher_id: str
    hall_id: str
    wifi_ssid: str  # Network name
    start_time: datetime
    is_active: bool = True
    connected_students: Set[str] = field(default_factory=set)  # Store connected student IDs