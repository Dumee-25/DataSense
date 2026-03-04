import os
import uuid
import math
import shutil
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Query, Depends, Cookie, Response
from sqlalchemy.orm import Session as DBSession

from database.connection import get_db
from database import crud
from database.models import JobStatus
from core import (StructuralAnalyzer, StatisticalEngine, ModelRecommender,
                  InsightGenerator, DeterministicSummary, AggregationEngine, RelevanceFilter,
                  ChartEngine)
from utils.data_validator import DataValidator

from fastapi.responses import StreamingResponse
import io
import base64
from core.pdf_generator import PDFReportGenerator
logger = logging.getLogger(__name__)
router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
JOB_RETENTION_HOURS = int(os.getenv("JOB_RETENTION_HOURS", "48"))
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

_active_jobs = 0  # In-memory concurrency counter


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_for_json(obj):
    """Recursively replace nan/inf/-inf with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# ── Background Pipeline ───────────────────────────────────────────────────────

def run_pipeline(job_id: str, file_path: str, user_context: str = "",
                  explicit_target: str = ""):
    """
    Runs the full analysis pipeline in a background thread.
    Persists all results to PostgreSQL. Cleans up temp file when done.

    Args:
        job_id: Unique job identifier
        file_path: Path to the uploaded CSV
        user_context: Optional data dictionary / domain context from the user
        explicit_target: Optional user-specified target column
    """
    global _active_jobs
    from database.connection import SessionLocal

    db = SessionLocal()
    start_time = datetime.utcnow()
    _active_jobs += 1

    try:
        logger.info(f"[{job_id}] Pipeline started: {file_path}")

        # ── Guard: check if cancelled before starting ──────────────────────
        job = crud.get_job(db, job_id)
        if not job or job.status == JobStatus.cancelled:
            logger.info(f"[{job_id}] Job was cancelled before pipeline started")
            return

        # ── Step 1: Validate (5%) ──────────────────────────────────────────
        crud.update_job_progress(db, job_id, 5, "Validating file...")
        validator = DataValidator(max_file_size_mb=MAX_FILE_SIZE_MB)
        val = validator.validate(file_path)
        if not val['valid']:
            crud.fail_job(db, job_id, val['error'], 'validation_error')
            return

        # ── Step 2: Load (10%) ────────────────────────────────────────────
        crud.update_job_progress(db, job_id, 10, "Loading dataset...")
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')

        if df.empty:
            crud.fail_job(db, job_id, "The CSV file is empty.", 'empty_file')
            return

        dataset_info = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 3),
            'column_names': df.columns.tolist(),
        }

        # Update file size on job record
        crud.update_job_status(
            db, job_id, JobStatus.processing,
            file_size_mb=dataset_info['size_mb']
        )

        # ── Step 3: Structural Analysis (30%) ─────────────────────────────
        if _is_cancelled(db, job_id): return
        crud.update_job_progress(db, job_id, 30, "Analyzing dataset structure...")
        try:
            blueprint = StructuralAnalyzer().analyze(
                df, explicit_target=explicit_target or None
            )
            logger.info(f"[{job_id}] Structural analysis complete")
        except Exception as e:
            logger.error(f"[{job_id}] Structural analysis failed: {e}", exc_info=True)
            crud.fail_job(db, job_id, str(e), 'structural_analysis_error')
            return

        # ── Step 4: Statistical Engine (55%) ──────────────────────────────
        if _is_cancelled(db, job_id): return
        crud.update_job_progress(db, job_id, 55, "Running statistical tests...")
        try:
            stats = StatisticalEngine().analyze(df, blueprint)
            logger.info(f"[{job_id}] Statistical analysis complete")
        except Exception as e:
            logger.error(f"[{job_id}] Statistical analysis failed: {e}", exc_info=True)
            crud.fail_job(db, job_id, str(e), 'statistical_analysis_error')
            return

        # ── Step 5: Model Recommendations (75%) ───────────────────────────
        if _is_cancelled(db, job_id): return
        crud.update_job_progress(db, job_id, 75, "Generating model recommendations...")
        try:
            recommendations = ModelRecommender().recommend(blueprint, stats)
            logger.info(f"[{job_id}] Model recommendations complete")
        except Exception as e:
            logger.error(f"[{job_id}] Model recommendations failed: {e}", exc_info=True)
            crud.fail_job(db, job_id, str(e), 'recommendation_error')
            return

        # ── Step 5b: Build deterministic summary (LLM firewall) ─────────
        #   The DeterministicSummary ensures the LLM only sees pre-calculated
        #   facts — it never touches raw data or does its own math.
        det_summary = DeterministicSummary().build(
            blueprint, stats, recommendations, user_context=user_context or None
        )
        logger.info(f"[{job_id}] Deterministic summary built (LLM firewall active)")

        # ── Step 6: Insights (90%) ────────────────────────────────────────
        if _is_cancelled(db, job_id): return
        crud.update_job_progress(db, job_id, 90, "Generating insights...")
        try:
            insights = InsightGenerator(use_llm=USE_LLM).generate_insights(
                blueprint, stats, recommendations,
                user_context=user_context, deterministic_summary=det_summary
            )
            logger.info(f"[{job_id}] Insights complete")
        except Exception as e:
            logger.error(f"[{job_id}] Insight generation failed: {e}", exc_info=True)
            # Non-fatal — use empty insights rather than failing the whole job
            insights = {
                'executive_summary': 'Analysis complete. Insight generation encountered an error.',
                'critical_insights': [],
                'high_priority_insights': [],
                'medium_priority_insights': [],
                'model_guidance': {},
                'quick_wins': [],
                'total_insights': 0,
                'severity_breakdown': {'critical': 0, 'high': 0, 'medium': 0},
            }

        # ── Step 7: Save to DB (100%) ─────────────────────────────────────
        crud.update_job_progress(db, job_id, 98, "Saving results...")

        clean_blueprint = sanitize_for_json(blueprint)
        clean_stats = sanitize_for_json(stats)
        clean_recs = sanitize_for_json(recommendations)
        clean_insights = sanitize_for_json(insights)
        clean_info = sanitize_for_json(dataset_info)

        crud.save_result(
            db,
            job_id=job_id,
            structural_analysis=clean_blueprint,
            statistical_analysis=clean_stats,
            model_recommendations=clean_recs,
            insights=clean_insights,
            dataset_info=clean_info,
        )

        crud.save_metadata(
            db,
            job_id=job_id,
            blueprint=clean_blueprint,
            insights=clean_insights,
            recommendations=clean_recs,
        )

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        crud.complete_job(db, job_id, elapsed)
        logger.info(f"[{job_id}] Completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"[{job_id}] Unexpected pipeline error: {e}", exc_info=True)
        try:
            crud.fail_job(db, job_id, f"Unexpected error: {str(e)}", 'unexpected_error')
        except Exception:
            pass

    finally:
        _active_jobs -= 1
        db.close()
        # Always clean up temp file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[{job_id}] Temp file removed")
        except Exception as e:
            logger.warning(f"[{job_id}] Could not remove temp file: {e}")


def _is_cancelled(db: DBSession, job_id: str) -> bool:
    """Check if a job was cancelled mid-pipeline."""
    job = crud.get_job(db, job_id)
    if job and job.status == JobStatus.cancelled:
        logger.info(f"[{job_id}] Job cancelled mid-pipeline")
        return True
    return False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/analyze")
async def upload_and_analyze(
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    context: Optional[str] = Form(default=None),
    target_column: Optional[str] = Form(default=None),
    db: DBSession = Depends(get_db),
    datasense_session: Optional[str] = Cookie(default=None),
):
    """
    Upload a CSV file and start analysis in the background.
    Returns a job_id immediately — poll /api/status/{job_id} for progress.

    Optional form fields:
      - context: A data dictionary or domain context string injected into the LLM prompt
      - target_column: Explicit target column name (overrides heuristic detection)
    """
    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Check concurrency limit
    if _active_jobs >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429,
            detail=f"Server busy ({MAX_CONCURRENT_JOBS} jobs running). Please try again shortly."
        )

    # Get or create browser session (tracks history without login)
    session = crud.get_or_create_session(db, datasense_session)

    # Set session cookie — 7 days, httponly
    response.set_cookie(
        key="datasense_session",
        value=str(session.id),
        max_age=60 * 60 * 24 * 7,
        httponly=True,
        samesite="lax",
    )

    # Create job record in DB
    job_id = str(uuid.uuid4())
    crud.create_job(
        db,
        job_id=job_id,
        filename=file.filename,
        session_id=session.id,
        user_id=session.user_id,
    )

    # Save uploaded file to temp directory
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_UPLOAD_DIR, f"{job_id}.csv")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        crud.delete_job(db, job_id)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Kick off background pipeline
    background_tasks.add_task(
        run_pipeline, job_id, file_path,
        user_context=context or "",
        explicit_target=target_column or ""
    )
    logger.info(f"[{job_id}] Job created for file: {file.filename}")

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "queued",
        "message": "Analysis started. Poll /api/status/{job_id} for updates.",
    }


@router.get("/status/{job_id}")
def get_status(job_id: str, db: DBSession = Depends(get_db)):
    """
    Poll this endpoint every 2 seconds to track job progress.
    Returns progress 0-100 and a human-readable status message.
    """
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "job_id": str(job.id),
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "progress_message": job.progress_message,
        "created_at": job.created_at.isoformat(),
    }

    if job.status == JobStatus.completed:
        response["completed_at"] = job.completed_at.isoformat() if job.completed_at else None
        response["processing_time_seconds"] = job.processing_time_seconds
        # Attach quick metadata so frontend can preview without full results fetch
        if job.metadata_:
            response["preview"] = {
                "rows": job.metadata_.row_count,
                "columns": job.metadata_.column_count,
                "critical_issues": job.metadata_.critical_issues_count,
                "high_issues": job.metadata_.high_issues_count,
                "primary_model": job.metadata_.primary_model_recommendation,
            }

    if job.status == JobStatus.failed:
        response["error"] = job.error
        response["error_type"] = job.error_type

    return response


@router.get("/results/{job_id}")
def get_results(job_id: str, db: DBSession = Depends(get_db)):
    """
    Fetch the full analysis results for a completed job.
    """
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status != JobStatus.completed:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet. Current status: {job.status.value}"
        )

    result = crud.get_result(db, job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found.")

    return {
        "job_id": job_id,
        "filename": job.filename,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "processing_time_seconds": job.processing_time_seconds,
        "results": {
            "dataset_info": result.dataset_info,
            "structural_analysis": result.structural_analysis,
            "statistical_analysis": result.statistical_analysis,
            "model_recommendations": result.model_recommendations,
            "insights": result.insights,
        }
    }


@router.delete("/cancel/{job_id}")
def cancel_job(job_id: str, db: DBSession = Depends(get_db)):
    """Cancel a queued or running job."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    cancelled = crud.cancel_job(db, job_id)
    if not cancelled:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status.value}"
        )
    return {"job_id": job_id, "status": "cancelled"}


@router.get("/jobs")
def list_jobs(
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: DBSession = Depends(get_db),
    datasense_session: Optional[str] = Cookie(default=None),
):
    """
    List all jobs for the current browser session.
    Powers the History page — no login required.
    """
    if not datasense_session:
        return {"total": 0, "jobs": []}

    jobs = crud.get_jobs_for_session(db, datasense_session, limit=limit, status=status)

    result = []
    for job in jobs:
        item = job.to_dict()
        if job.metadata_:
            item["metadata"] = {
                "rows": job.metadata_.row_count,
                "columns": job.metadata_.column_count,
                "missing_pct": job.metadata_.missing_percentage,
                "duplicate_rows": job.metadata_.duplicate_rows,
                "data_structure": job.metadata_.data_structure_type,
                "critical_issues": job.metadata_.critical_issues_count,
                "high_issues": job.metadata_.high_issues_count,
                "medium_issues": job.metadata_.medium_issues_count,
                "primary_model": job.metadata_.primary_model_recommendation,
            }
        result.append(item)

    return {"total": len(result), "jobs": result}


@router.delete("/jobs/{job_id}")
def delete_job(
    job_id: str,
    db: DBSession = Depends(get_db),
    datasense_session: Optional[str] = Cookie(default=None),
):
    """Delete a job and all its results. Only the owning session can delete."""
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Ownership check
    if datasense_session and str(job.session_id) != datasense_session:
        raise HTTPException(status_code=403, detail="Not authorized to delete this job.")

    crud.delete_job(db, job_id)
    return {"job_id": job_id, "deleted": True}


@router.delete("/jobs")
def cleanup_jobs(db: DBSession = Depends(get_db)):
    """Remove jobs older than the retention period."""
    removed = crud.cleanup_expired_jobs(db, older_than_hours=JOB_RETENTION_HOURS)
    return {"removed": removed, "message": f"Removed {removed} expired job(s)."}

@router.get("/results/{job_id}/export/pdf")
def export_pdf(job_id: str, db: DBSession = Depends(get_db)):
    """
    Generate and stream a PDF report for a completed job.
    The browser will trigger a file download automatically.
    """
    from fastapi.responses import StreamingResponse
    import io
    from core.pdf_generator import PDFReportGenerator

    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status != JobStatus.completed:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet. Status: {job.status.value}"
        )

    result = crud.get_result(db, job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found.")

    # Build the results dict the generator expects
    results_payload = {
        "filename": job.filename,
        "completed_at": job.completed_at.isoformat() if job.completed_at else "",
        "processing_time_seconds": job.processing_time_seconds,
        "results": {
            "dataset_info": result.dataset_info,
            "structural_analysis": result.structural_analysis,
            "statistical_analysis": result.statistical_analysis,
            "model_recommendations": result.model_recommendations,
            "insights": result.insights,
        }
    }

    try:
        buf = io.BytesIO()
        PDFReportGenerator().generate_to_buffer(results_payload, buf)
        buf.seek(0)
    except Exception as e:
        logger.error(f"PDF generation failed for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    safe_filename = job.filename.replace('.csv', '').replace(' ', '_')
    filename = f"datasense_report_{safe_filename}.pdf"

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/results/{job_id}/charts")
def get_charts(job_id: str, db: DBSession = Depends(get_db)):
    """
    Generate analysis charts for a completed job.
    Returns a dict of chart_name → base64-encoded PNG string (or null if skipped).
    """
    job = crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status != JobStatus.completed:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet. Status: {job.status.value}"
        )

    result = crud.get_result(db, job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Results not found.")

    results_payload = {
        "results": {
            "dataset_info": result.dataset_info,
            "structural_analysis": result.structural_analysis,
            "statistical_analysis": result.statistical_analysis,
            "model_recommendations": result.model_recommendations,
            "insights": result.insights,
        }
    }

    try:
        raw_charts = ChartEngine().generate_all(results_payload)
        encoded = {}
        for name, png_bytes in raw_charts.items():
            if png_bytes is not None:
                encoded[name] = base64.b64encode(png_bytes).decode("ascii")
            else:
                encoded[name] = None
        return {"job_id": job_id, "charts": encoded}
    except Exception as e:
        logger.error(f"Chart generation failed for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")
