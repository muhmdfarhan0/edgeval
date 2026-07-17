# EdgeEval

Catch computer vision model regressions before you ship them to the edge.
Upload a baseline model and a quantized/optimized candidate (YOLO, ONNX,
INT8) plus a validation set — get per-class mAP/precision/recall deltas,
a latency comparison, an AI-generated explanation of what changed, and a
rule-based deployment recommendation.

## Structure

```
backend/    FastAPI API — inference, metrics, diff engine, LLM insights
frontend/   Static site (vanilla HTML/CSS/JS) — upload UI + report view
```

The two are deployed separately and talk over HTTP:

- **Backend** → a container host with persistent disk (Railway, Render, Fly.io,
  or a Hugging Face Space with the Docker SDK). Not deployable to serverless
  platforms (Vercel functions, AWS Lambda) — the ML dependencies (torch,
  onnxruntime, opencv) and multi-minute job runtimes don't fit that model.
- **Frontend** → Vercel (or any static host). Set `API_BASE` near the top of
  the `<script>` block in `frontend/index.html` to your deployed backend's URL.

## Local development

```bash
cd backend
python -m venv venv
venv/Scripts/activate   # or source venv/bin/activate on macOS/Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/prepare_models.py        # downloads a baseline + exports an INT8 ONNX candidate
python scripts/make_sample_dataset.py   # builds a small YOLO-format validation set

uvicorn app.main:app --reload
```

The API is now at `http://127.0.0.1:8000`. To try the frontend against it
locally, open `frontend/index.html` directly in a browser (or serve the
`frontend/` folder with any static server) with `API_BASE` set to
`http://127.0.0.1:8000`.

## Configuration

See `backend/.env.example` for all environment variables — API key auth,
rate limiting, allowed CORS origins, and the optional Groq API key that
powers the AI Insights section (the pass/fail verdict itself always stays
rule-based, regardless of whether this is configured).

## Deploying

1. Deploy `backend/` (it has its own `Dockerfile`) to Railway/Render/Fly.io/
   a Docker-SDK Hugging Face Space. Mount a persistent volume at the path
   `DATA_DIR` points to, and set `ALLOWED_ORIGINS` to your frontend's URL.
2. Deploy `frontend/` to Vercel as a static site, with `API_BASE` in
   `index.html` pointed at the backend's public URL.
