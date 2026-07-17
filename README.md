# EdgeEval

**[Live demo →](https://edgeval.vercel.app)**

Catch per-class accuracy regressions in quantized computer vision models —
YOLO, ONNX, INT8 — before you deploy them to edge hardware. There's no
"Promptfoo for computer vision": every eval tool on the market (Promptfoo,
DeepEval, LangSmith, Braintrust) evaluates LLM/text output, not CV models.
EdgeEval fills that gap.

Upload a baseline model and a quantized/optimized candidate (`.pt` or
`.onnx`) plus a YOLO-format validation set — get IoU-matched per-class
mAP@0.5, precision/recall deltas, a latency benchmark, an AI-generated
plain-English explanation of what changed, and a rule-based deployment
recommendation. Per-class breakdown is the differentiator: aggregate
mAP can barely move while one class silently regresses 20+ points —
exactly the kind of failure that breaks a model in production.

## Structure

```
backend/    FastAPI API — inference, metrics, diff engine, LLM insights
frontend/   Static site (vanilla HTML/CSS/JS) — upload UI + report view
```

The two are deployed separately and talk over HTTP:

- **Backend** → Render, via the `render.yaml` Blueprint at the repo root
  (Docker-based container host, not serverless — the ML dependencies (torch,
  onnxruntime, opencv) and multi-minute job runtimes don't fit a functions
  platform like Vercel's or Lambda's).
- **Frontend** → Vercel, with the `frontend/` folder set as the project root.
  Set `API_BASE` near the top of the `<script>` block in `frontend/index.html`
  to the Render service's URL once it's live.

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

### Backend → Render

1. In the Render dashboard: **New > Blueprint**, connect this GitHub repo.
   Render reads `render.yaml` and configures the Docker web service rooted at
   `backend/` automatically.
2. When prompted, paste in `GROQ_API_KEY` (deliberately excluded from
   `render.yaml`/git — see `backend/.env.example`).
3. Deploy. The first build installs torch/onnxruntime/opencv and takes a
   few minutes.
4. Once live, note the service URL (`https://edgeval-api-xxxx.onrender.com`).
5. **Free-tier caveat:** Render's free plan has no persistent disk and spins
   the service down after inactivity — the SQLite DB and uploaded models/
   datasets don't survive a restart, and the first request after idle will be
   slow (cold start + model load). Fine for a demo; upgrade to a paid plan
   with a disk mounted at `DATA_DIR` (`/app/data`) if job history needs to
   persist.
6. Once you have a final frontend URL (see below), update `ALLOWED_ORIGINS`
   in the Render service's environment settings from `*` to that exact URL.

### Frontend → Vercel

1. Import this GitHub repo in Vercel, set the **root directory** to `frontend`.
2. Before or after deploying, set `API_BASE` near the top of the `<script>`
   block in `frontend/index.html` to the Render URL from above, then push.
3. Once Vercel assigns a domain, go back and lock down the backend's
   `ALLOWED_ORIGINS` to that domain (step 6 above).
