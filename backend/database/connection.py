import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    logger.warning("DATABASE_URL not set — using development default. DO NOT USE IN PRODUCTION.")
    DATABASE_URL = "postgresql://datasense_user:datasense123@localhost:5432/datasense"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and closes it after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()