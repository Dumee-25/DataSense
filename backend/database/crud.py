import uuid
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import desc

from database.models import User, Session, Job, Result, DatasetMetadata, JobStatus


# ─── Session CRUD ─────────────────────────────────────────────────────────────

def create_session(db: DBSession, user_id=None, hours: int = 168) -> Session:
    """Create a browser session. Default expiry: 7 days."""
    session = Session(
        user_id=user_id,
        expires_at=datetime.utcnow() + timedelta(hours=hours)
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: DBSession, session_id: str) -> Optional[Session]:
    """Get a session by ID. Returns None if expired or not found."""
    try:
        sid = uuid.UUID(session_id)
    except (ValueError, AttributeError):
        return None
    session = db.query(Session).filter(Session.id == sid).first()
    if session and session.expires_at < datetime.utcnow():
        return None
    return session


def get_or_create_session(db: DBSession, session_id: Optional[str]) -> Session:
    """Get existing session or create a new anonymous one."""
    if session_id:
        session = get_session(db, session_id)
        if session:
            return session
    return create_session(db)


# ─── Job CRUD ─────────────────────────────────────────────────────────────────

def create_job(
    db: DBSession,
    job_id: str,
    filename: str,
    session_id=None,
    user_id=None,
    file_size_mb: Optional[float] = None
) -> Job:
    """Create a new job record when a CSV is uploaded."""
    job = Job(
        id=uuid.UUID(job_id),
        session_id=session_id,
        user_id=user_id,
        filename=filename,
        file_size_mb=file_size_mb,
        status=JobStatus.queued,
        progress=0,
        progress_message="Queued for processing"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: DBSession, job_id: str) -> Optional[Job]:
    """Get a job by ID."""
    try:
        jid = uuid.UUID(job_id)
    except (ValueError, AttributeError):
        return None
    return db.query(Job).filter(Job.id == jid).first()


def update_job_progress(db: DBSession, job_id: str, progress: int, message: str):
    """Update progress % and status message."""
    try:
        jid = uuid.UUID(job_id)
    except (ValueError, AttributeError):
        return
    db.query(Job).filter(Job.id == jid).update({
        "progress": progress,
        "progress_message": message,
        "updated_at": datetime.utcnow()
    })
    db.commit()


def update_job_status(db: DBSession, job_id: str, status: JobStatus, **kwargs):
    """Update job status plus any extra fields."""
    try:
        jid = uuid.UUID(job_id)
    except (ValueError, AttributeError):
        return
    updates = {"status": status, "updated_at": datetime.utcnow()}
    updates.update(kwargs)
    db.query(Job).filter(Job.id == jid).update(updates)
    db.commit()


def complete_job(db: DBSession, job_id: str, processing_time_seconds: float):
    """Mark job as completed."""
    update_job_status(
        db, job_id,
        status=JobStatus.completed,
        progress=100,
        progress_message="Analysis complete",
        completed_at=datetime.utcnow(),
        processing_time_seconds=processing_time_seconds
    )


def fail_job(db: DBSession, job_id: str, error: str, error_type: str):
    """Mark job as failed with error details."""
    update_job_status(
        db, job_id,
        status=JobStatus.failed,
        error=error,
        error_type=error_type
    )


def cancel_job(db: DBSession, job_id: str) -> bool:
    """Cancel a job. Returns False if already in a terminal state."""
    job = get_job(db, job_id)
    if not job:
        return False
    if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
        return False
    update_job_status(db, job_id, status=JobStatus.cancelled)
    return True


def delete_job(db: DBSession, job_id: str) -> bool:
    """Delete job and all associated data via cascade."""
    job = get_job(db, job_id)
    if not job:
        return False
    db.delete(job)
    db.commit()
    return True


def get_jobs_for_session(
    db: DBSession,
    session_id: str,
    limit: int = 50,
    status: Optional[str] = None
) -> List[Job]:
    """List jobs for a session, newest first."""
    try:
        sid = uuid.UUID(session_id)
    except (ValueError, AttributeError):
        return []
    query = db.query(Job).filter(Job.session_id == sid)
    if status:
        query = query.filter(Job.status == status)
    return query.order_by(desc(Job.created_at)).limit(limit).all()


def get_jobs_for_user(
    db: DBSession,
    user_id: str,
    limit: int = 50,
    status: Optional[str] = None
) -> List[Job]:
    """List jobs for a user, newest first."""
    try:
        uid = uuid.UUID(user_id)
    except (ValueError, AttributeError):
        return []
    query = db.query(Job).filter(Job.user_id == uid)
    if status:
        query = query.filter(Job.status == status)
    return query.order_by(desc(Job.created_at)).limit(limit).all()


def cleanup_expired_jobs(db: DBSession, older_than_hours: int = 48) -> int:
    """Delete jobs older than retention period. Returns count deleted."""
    cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
    jobs = db.query(Job).filter(Job.created_at < cutoff).all()
    count = len(jobs)
    for job in jobs:
        db.delete(job)
    db.commit()
    return count


# ─── Result CRUD ──────────────────────────────────────────────────────────────

def save_result(
    db: DBSession,
    job_id: str,
    structural_analysis: dict,
    statistical_analysis: dict,
    model_recommendations: dict,
    insights: dict,
    dataset_info: dict
) -> Result:
    """Save full analysis results for a completed job."""
    result = Result(
        job_id=uuid.UUID(job_id),
        structural_analysis=structural_analysis,
        statistical_analysis=statistical_analysis,
        model_recommendations=model_recommendations,
        insights=insights,
        dataset_info=dataset_info
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_result(db: DBSession, job_id: str) -> Optional[Result]:
    """Get full results for a job."""
    try:
        jid = uuid.UUID(job_id)
    except (ValueError, AttributeError):
        return None
    return db.query(Result).filter(Result.job_id == jid).first()


# ─── DatasetMetadata CRUD ─────────────────────────────────────────────────────

def save_metadata(
    db: DBSession,
    job_id: str,
    blueprint: dict,
    insights: dict,
    recommendations: dict
) -> DatasetMetadata:
    """Extract and save quick-access metadata from full results."""
    basic_info = blueprint.get("basic_info", {})
    data_structure = blueprint.get("data_structure", {})

    metadata = DatasetMetadata(
        job_id=uuid.UUID(job_id),
        row_count=basic_info.get("rows"),
        column_count=basic_info.get("columns"),
        missing_percentage=basic_info.get("missing_percentage"),
        duplicate_rows=basic_info.get("duplicate_rows"),
        data_structure_type=data_structure.get("type"),
        critical_issues_count=len(insights.get("critical_insights", [])),
        high_issues_count=len(insights.get("high_priority_insights", [])),
        medium_issues_count=len(insights.get("medium_priority_insights", [])),
        primary_model_recommendation=recommendations.get("primary_model")
    )
    db.add(metadata)
    db.commit()
    db.refresh(metadata)
    return metadata