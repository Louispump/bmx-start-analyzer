"""
BMX Start Analyzer — Web App
FastAPI server: upload vidéo → analyse → résultats avec vidéo annotée

Usage:
  uvicorn app:app --reload --port 8000
  Ouvrir http://localhost:8000 dans le navigateur
"""

import os
os.environ["MPLBACKEND"] = "agg"  # DOIT être avant tout import matplotlib/ultralytics

import uuid
import json
import shutil
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader

# Importer les fonctions d'analyse
from analyze import main as analyze_main, detect_beeps_audio, detect_gate_drop

# ── Dossiers ──────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="BMX Start Analyzer")

app.mount("/static",  StaticFiles(directory="static"),  name="static")
app.mount("/output",  StaticFiles(directory="output"),  name="output")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# cache_size=0 : contournement bug Python 3.14 + Jinja2 LRUCache
_jinja_env = Environment(loader=FileSystemLoader("templates"), cache_size=0)
templates  = Jinja2Templates(env=_jinja_env)

# ── Job store (in-memory, suffit pour le dev) ─────────────────────────────────
# Structure: { job_id: { status, filename, progress, results, error } }
jobs: dict = {}


# ── Traitement en arrière-plan ────────────────────────────────────────────────
def run_analysis(job_id: str, video_path: Path):
    """Pipeline complet: détection gate → YOLO → résultats."""
    try:
        jobs[job_id]["status"]   = "processing"
        jobs[job_id]["progress"] = "Détection du gate..."

        # 1. Tentative audio
        beep_times, gate_time_audio, audio_conf = detect_beeps_audio(str(video_path))
        bip1_time   = None
        gate_method = "visual"

        if beep_times and audio_conf >= 0.40:
            gate_drop   = gate_time_audio
            bip1_time   = beep_times[0]
            gate_method = "audio"
            jobs[job_id]["progress"] = f"Gate détecté (audio, {audio_conf:.0%}) — analyse YOLO..."
        else:
            gate_frame, gate_time_visual, visual_conf, _ = detect_gate_drop(str(video_path))
            gate_drop = gate_time_visual
            jobs[job_id]["progress"] = f"Gate détecté (visuel, {visual_conf:.0%}) — analyse YOLO..."

        if gate_drop is None:
            raise ValueError("Impossible de détecter le gate drop automatiquement.")

        # 2. Analyse principale
        jobs[job_id]["progress"] = "Pose estimation + tracking..."
        results = analyze_main(str(video_path), gate_drop=gate_drop, bip1_time=bip1_time)

        if results is None:
            raise ValueError("L'analyse n'a pas retourné de résultats.")

        results["gate_method"] = gate_method
        results["bip_times"]   = beep_times if beep_times else []

        # 3. Sauvegarder le JSON
        stem         = video_path.stem.replace(f"{job_id}_", "")
        results_path = OUTPUT_DIR / f"{stem}_results.json"
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        jobs[job_id].update({
            "status":    "done",
            "progress":  "Terminé",
            "stem":      stem,
            "results":   results,
        })

    except Exception as e:
        jobs[job_id].update({
            "status":  "error",
            "error":   str(e),
            "detail":  traceback.format_exc(),
        })


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Reçoit la vidéo, démarre l'analyse en arrière-plan, retourne un job_id."""
    job_id     = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status":   "queued",
        "filename": file.filename,
        "progress": "En attente...",
    }

    background_tasks.add_task(run_analysis, job_id, video_path)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Polling: retourne le statut du job (queued / processing / done / error)."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {
        "status":   job["status"],
        "progress": job.get("progress", ""),
        "error":    job.get("error", ""),
    }


@app.get("/result/{job_id}")
async def result(request: Request, job_id: str):
    """Page de résultats."""
    job = jobs.get(job_id)
    if not job:
        return templates.TemplateResponse(request, "index.html",
                                          {"error": "Job introuvable."})
    if job["status"] != "done":
        return templates.TemplateResponse(request, "index.html",
                                          {"error": "Analyse pas encore terminée."})
    return templates.TemplateResponse(request, "result.html", {
        "job_id":  job_id,
        "results": job["results"],
    })
