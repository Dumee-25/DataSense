import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from utils.logging_config import setup_logging
setup_logging()

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

# Rate limiting — must be added after CORS so headers are present
from utils.rate_limiter import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

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
