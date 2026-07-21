from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.formparsers import MultiPartParser

# Starlette spools any multipart file part over 1MB (its default) to a real
# temp file during parsing — meaning our model/dataset uploads (all >1MB)
# already get written to disk once before _save_upload() below writes them
# again to their final destination. Raising this lets moderate-sized uploads
# stay in memory through parsing, cutting that redundant disk write. Bounded
# deliberately (not raised further) so a very large upload still falls back
# to disk-backed spooling instead of risking OOM on a memory-constrained hosts.
MultiPartParser.spool_max_size = 25 * 1024 * 1024  # 25MB

from app.config import ALLOWED_ORIGINS, CONF_THRESHOLD, JOBS_DIR
from app.db import PIPELINE_STAGES, EvaluationJob, get_session, init_db
from app.security import enforce_rate_limit, require_api_key
from app.services.dataset_loader import extract_dataset_zip
from app.services.diff_engine import build_report
from app.services.inference import run_model_over_dataset
from app.services.latency import benchmark_latency
from app.services.llm_insights import generate_insights
from app.services.model_loader import load_model

app = FastAPI(title="EdgeEval API", version="0.1.0")

# The frontend is a separately-deployed static site (Vercel), so every
# request here is cross-origin — without this, the browser blocks the
# frontend's fetch() calls before they even reach these routes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def index():
    return {"service": "EdgeEval API", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _metrics_to_dict(metrics) -> dict:
    return {
        "overall_map": metrics.overall_map,
        "per_class": {
            name: {
                "precision": m.precision,
                "recall": m.recall,
                "ap": m.ap,
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "num_ground_truth": m.num_ground_truth,
            }
            for name, m in metrics.per_class.items()
        },
        "inference_times_ms": metrics.inference_times_ms,
        "mean_inference_ms": (
            sum(metrics.inference_times_ms) / len(metrics.inference_times_ms)
            if metrics.inference_times_ms
            else 0.0
        ),
    }


def _run_evaluation_job(
    job_id: str,
    job_dir: Path,
    baseline_path: Path,
    candidate_path: Path,
    dataset_zip_path: Path,
    baseline_filename: str,
    candidate_filename: str,
) -> None:
    """Runs the full evaluation pipeline in the background so the client
    doesn't block an HTTP request on a multi-minute model evaluation."""
    session = get_session()
    job = session.get(EvaluationJob, job_id)
    job.status = "running"
    job.stage = "loading_dataset"
    session.commit()

    try:
        dataset = extract_dataset_zip(dataset_zip_path, job_dir / "dataset")
        job.num_images = len(dataset.images)
        job.stage = "loading_models"
        session.commit()

        baseline = load_model(baseline_path, CONF_THRESHOLD)
        class_names = baseline.names if baseline.names else dataset.class_names
        candidate = load_model(candidate_path, CONF_THRESHOLD, fallback_names=class_names)
        job.num_classes = len(class_names)
        job.stage = "evaluating_baseline"
        session.commit()

        baseline_metrics = run_model_over_dataset(baseline, dataset, class_names)
        job.stage = "evaluating_candidate"
        session.commit()

        candidate_metrics = run_model_over_dataset(candidate, dataset, class_names)
        job.stage = "benchmarking_latency"
        session.commit()

        benchmark_image = dataset.images[0].image_path
        baseline_latency = benchmark_latency(baseline, benchmark_image)
        candidate_latency = benchmark_latency(candidate, benchmark_image)
        job.stage = "building_report"
        session.commit()

        baseline_dict = {"model": "baseline", "filename": baseline_filename, **_metrics_to_dict(baseline_metrics)}
        candidate_dict = {"model": "candidate", "filename": candidate_filename, **_metrics_to_dict(candidate_metrics)}

        report = build_report(baseline_dict, candidate_dict, baseline_latency, candidate_latency)
        job.stage = "generating_insights"
        session.commit()

        # Best-effort: the deploy_recommended verdict above is already final
        # and deterministic. This just adds a plain-English explanation on
        # top — if it's unavailable or fails, the report stands on its own.
        report["llm_insights"] = generate_insights(report, len(dataset.images), len(class_names))

        result = {
            "job_id": job_id,
            "num_images": len(dataset.images),
            "num_classes": len(class_names),
            "class_names": class_names,
            "baseline": baseline_dict,
            "candidate": candidate_dict,
            "latency_benchmark": {
                "baseline": baseline_latency,
                "candidate": candidate_latency,
            },
            "report": report,
        }

        job.status = "complete"
        job.stage = "complete"
        job.result = result
        session.commit()

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        session.commit()
    finally:
        session.close()


@app.post("/evaluate", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
async def evaluate(request: Request, background_tasks: BackgroundTasks):
    # FastAPI's automatic File(...) injection caps each part at 1MB by default
    # (Starlette's MultiPartParser max_part_size) — far too small for model
    # weights, so the form is parsed manually here with a much higher limit.
    form = await request.form(max_part_size=500 * 1024 * 1024)

    baseline_model = form.get("baseline_model")
    candidate_model = form.get("candidate_model")
    validation_dataset = form.get("validation_dataset")
    if not (
        isinstance(baseline_model, StarletteUploadFile)
        and isinstance(candidate_model, StarletteUploadFile)
        and isinstance(validation_dataset, StarletteUploadFile)
    ):
        raise HTTPException(
            status_code=422,
            detail="baseline_model, candidate_model, and validation_dataset must all be uploaded files.",
        )

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = _save_upload(baseline_model, job_dir / f"baseline{Path(baseline_model.filename).suffix}")
    candidate_path = _save_upload(candidate_model, job_dir / f"candidate{Path(candidate_model.filename).suffix}")
    dataset_zip_path = _save_upload(validation_dataset, job_dir / "dataset.zip")

    session = get_session()
    try:
        job = EvaluationJob(
            id=job_id,
            status="pending",
            stage="queued",
            baseline_filename=baseline_model.filename,
            candidate_filename=candidate_model.filename,
        )
        session.add(job)
        session.commit()
    finally:
        session.close()

    background_tasks.add_task(
        _run_evaluation_job,
        job_id,
        job_dir,
        baseline_path,
        candidate_path,
        dataset_zip_path,
        baseline_model.filename,
        candidate_model.filename,
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/evaluate/{job_id}/status")
def get_status(job_id: str):
    session = get_session()
    try:
        job = session.get(EvaluationJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "job_id": job.id,
            "status": job.status,
            "stage": job.stage,
            "stage_index": PIPELINE_STAGES.index(job.stage) if job.stage in PIPELINE_STAGES else 0,
            "total_stages": len(PIPELINE_STAGES),
            "error": job.error,
            "num_images": job.num_images,
            "num_classes": job.num_classes,
        }
    finally:
        session.close()


@app.get("/evaluate/{job_id}")
def get_job(job_id: str):
    session = get_session()
    try:
        job = session.get(EvaluationJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "job_id": job.id,
            "status": job.status,
            "error": job.error,
            "num_images": job.num_images,
            "num_classes": job.num_classes,
            "result": job.result,
        }
    finally:
        session.close()


@app.get("/evaluate/{job_id}/report")
def get_report(job_id: str):
    session = get_session()
    try:
        job = session.get(EvaluationJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "complete":
            raise HTTPException(status_code=409, detail=f"job is {job.status}, not complete")

        return {
            "job_id": job.id,
            "num_images": job.num_images,
            "num_classes": job.num_classes,
            "baseline_overall_map": job.result["baseline"]["overall_map"],
            "candidate_overall_map": job.result["candidate"]["overall_map"],
            **job.result["report"],
        }
    finally:
        session.close()
