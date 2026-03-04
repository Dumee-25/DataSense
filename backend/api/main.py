import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("DataSense API starting up...")

    # Ensure temp upload directory exists
    temp_dir = os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Temp upload dir ready: {temp_dir}")

    # Test DB connection on startup
    try:
        from database.connection import engine
        with engine.connect() as conn:
            logger.info("PostgreSQL connection OK")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}")

    # Recover orphaned jobs from previous crash/restart (#23)
    try:
        from database.connection import SessionLocal
        from database.models import Job, JobStatus
        db = SessionLocal()
        try:
            orphaned = db.query(Job).filter(
                Job.status.in_([JobStatus.queued, JobStatus.processing])
            ).all()
            for job in orphaned:
                job.status = JobStatus.failed
                job.error = "Server restarted during processing."
                job.error_type = "server_restart"
            if orphaned:
                db.commit()
                logger.info(f"Recovered {len(orphaned)} orphaned job(s)")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Orphaned job recovery failed: {e}")

    yield

    logger.info("DataSense API shutting down...")


app = FastAPI(
    title="DataSense API",
    description="AI-powered data analysis and consulting platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSRF protection — require custom header on mutation requests (#20)
ENABLE_CSRF = os.getenv("ENABLE_CSRF", "true").lower() == "true"

@app.middleware("http")
async def csrf_protection(request, call_next):
    """Require X-DataSense-Request header on mutations to prevent CSRF."""
    if ENABLE_CSRF and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        if not request.headers.get("x-datasense-request"):
            from starlette.responses import JSONResponse
            return JSONResponse(status_code=403, content={"detail": "CSRF validation failed."})
    return await call_next(request)

# Register routes
from api.routes import router
from api.auth_routes import auth_router
app.include_router(router, prefix="/api")
app.include_router(auth_router, prefix="/api/auth")

@app.get("/")
def root():
    return {
        "name": "DataSense API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
