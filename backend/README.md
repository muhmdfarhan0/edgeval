# EdgeEval API

FastAPI backend for EdgeEval — evaluates quantized/optimized computer vision
models (YOLO, ONNX, INT8) against a baseline: per-class mAP, precision/recall
deltas, latency benchmarking, and a rule-based deployment recommendation with
an optional LLM-generated narrative summary.

This is the API only. The upload/report frontend is deployed separately
(Vercel) and talks to this service over HTTP — see `/docs` for the
interactive API reference once it's running.

## Deploying on Render

This repo includes `render.yaml` at the repo root — in the Render dashboard,
choose **New > Blueprint**, point it at this GitHub repo, and it will pick up
the Docker build (rooted at `backend/`) automatically. You'll be prompted to
paste in `GROQ_API_KEY` (kept out of git on purpose). See the root `README.md`
for the full deployment checklist.
