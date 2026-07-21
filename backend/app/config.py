import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "edgeval.db"

DATA_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

# IoU threshold used to match predicted boxes to ground-truth boxes for AP/precision/recall.
IOU_MATCH_THRESHOLD = 0.5

# Confidence threshold below which predictions are discarded before matching.
CONF_THRESHOLD = 0.25

# Latency benchmarking: warm-up runs are discarded so first-inference JIT/cache
# costs don't skew the measured mean; the timed runs repeat on the same image
# so both models are compared under identical input conditions.
LATENCY_WARMUP_RUNS = 5
LATENCY_TIMED_RUNS = 50

# Per-class AP drop (candidate - baseline) beyond which a class is flagged.
AP_DROP_WARNING_THRESHOLD = 0.05
AP_DROP_CRITICAL_THRESHOLD = 0.15

# Optional: LLM-generated narrative summary layered on top of the report.
# The deploy_recommended pass/fail verdict stays rule-based/deterministic
# (see diff_engine.py) regardless of whether this is configured — the LLM
# only explains the numbers in plain English, it never decides the gate.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Remote demo asset URLs for a hosted demo dataset and model pair.
# If configured, /evaluate/demo will download and cache these once.
DEMO_BASELINE_URL = os.environ.get("DEMO_BASELINE_URL")
DEMO_CANDIDATE_URL = os.environ.get("DEMO_CANDIDATE_URL")
DEMO_DATASET_URL = os.environ.get("DEMO_DATASET_URL")
DEMO_CACHE_DIR = DATA_DIR / "demo_assets"
DEMO_CACHE_DIR.mkdir(exist_ok=True)

# Comma-separated list of origins allowed to call this API (e.g. your Vercel
# frontend's URL). Defaults to "*" for local development/testing — restrict
# this in production via the ALLOWED_ORIGINS env var.
_allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_raw.split(",")] if _allowed_origins_raw != "*" else ["*"]
