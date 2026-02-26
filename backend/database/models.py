import uuid
import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Text, ForeignKey, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from database.connection import Base


class UserRole(str, enum.Enum):
    viewer = "viewer"
    analyst = "analyst"
    admin = "admin"


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class User(Base):
    """Registered user. Anonymous users are tracked via Session only."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=True)
    name = Column(String(100), nullable=True)
    role = Column(SAEnum(UserRole), default=UserRole.analyst, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="user")

    def __repr__(self):
        return f"<User id={self.id} email={self.email}>"


class Session(Base):
    """Browser session — created for every visitor, linked to User if authenticated."""
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    user = relationship("User", back_populates="sessions")
    jobs = relationship("Job", back_populates="session")

    def __repr__(self):
        return f"<Session id={self.id} user_id={self.user_id}>"


class Job(Base):
    """One analysis run. Linked to a Session (anonymous) or User (authenticated)."""
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    file_size_mb = Column(Float, nullable=True)
    status = Column(SAEnum(JobStatus), default=JobStatus.queued, nullable=False, index=True)
    progress = Column(Integer, default=0, nullable=False)
    progress_message = Column(String(255), default="Queued for processing")
    error = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="jobs")
    session = relationship("Session", back_populates="jobs")
    result = relationship("Result", back_populates="job", uselist=False, cascade="all, delete-orphan")
    metadata_ = relationship("DatasetMetadata", back_populates="job", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Job id={self.id} filename={self.filename} status={self.status}>"

    def to_dict(self):
        return {
            "job_id": str(self.id),
            "filename": self.filename,
            "file_size_mb": self.file_size_mb,
            "status": self.status.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "error": self.error,
            "error_type": self.error_type,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Result(Base):
    """Full analysis output stored as JSONB. One result per completed job."""
    __tablename__ = "results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), unique=True, nullable=False)
    structural_analysis = Column(JSONB, nullable=True)
    statistical_analysis = Column(JSONB, nullable=True)
    model_recommendations = Column(JSONB, nullable=True)
    insights = Column(JSONB, nullable=True)
    dataset_info = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    job = relationship("Job", back_populates="result")

    def to_dict(self):
        return {
            "structural_analysis": self.structural_analysis,
            "statistical_analysis": self.statistical_analysis,
            "model_recommendations": self.model_recommendations,
            "insights": self.insights,
            "dataset_info": self.dataset_info,
        }


class DatasetMetadata(Base):
    """Quick-access summary — avoids loading heavy JSONB for list/history views."""
    __tablename__ = "dataset_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), unique=True, nullable=False)
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    missing_percentage = Column(Float, nullable=True)
    duplicate_rows = Column(Integer, nullable=True)
    data_structure_type = Column(String(50), nullable=True)
    critical_issues_count = Column(Integer, default=0)
    high_issues_count = Column(Integer, default=0)
    medium_issues_count = Column(Integer, default=0)
    primary_model_recommendation = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    job = relationship("Job", back_populates="metadata_")