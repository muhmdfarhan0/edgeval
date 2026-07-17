from datetime import datetime, timezone

from sqlalchemy import create_engine, text, Column, String, Float, Integer, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# Ordered pipeline stages surfaced to the client for progress display.
PIPELINE_STAGES = [
    "queued",
    "loading_dataset",
    "loading_models",
    "evaluating_baseline",
    "evaluating_candidate",
    "benchmarking_latency",
    "building_report",
    "generating_insights",
    "complete",
]


class EvaluationJob(Base):
    __tablename__ = "evaluation_jobs"

    id = Column(String, primary_key=True)
    status = Column(String, default="pending")  # pending | running | complete | failed
    stage = Column(String, default="queued")  # see PIPELINE_STAGES
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    baseline_filename = Column(String, nullable=True)
    candidate_filename = Column(String, nullable=True)
    num_images = Column(Integer, nullable=True)
    num_classes = Column(Integer, nullable=True)

    # Full structured result, see services/metrics.py for shape.
    result = Column(JSON, nullable=True)


def init_db():
    Base.metadata.create_all(engine)
    # Lightweight migration for the `stage` column on DBs created before it existed.
    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(evaluation_jobs)"))}
        if "stage" not in cols:
            conn.execute(text("ALTER TABLE evaluation_jobs ADD COLUMN stage TEXT"))
            conn.commit()


def get_session():
    return SessionLocal()
