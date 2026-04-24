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
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader

from analyze import main as analyze_main, detect_beeps_audio, detect_gate_drop

# ── Dossiers ──────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
PROS_DB    = OUTPUT_DIR / "pros_db.json"   # persistance de la banque de pros
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="BMX Start Analyzer")

app.mount("/static",  StaticFiles(directory="static"),  name="static")
app.mount("/output",  StaticFiles(directory="output"),  name="output")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

_jinja_env = Environment(loader=FileSystemLoader("templates"), cache_size=0)
templates  = Jinja2Templates(env=_jinja_env)

# ── Stores ────────────────────────────────────────────────────────────────────
jobs: dict = {}   # analyses en cours / terminées (in-memory)

# Banque de pros — chargée depuis le fichier JSON au démarrage
def load_pros() -> dict:
    if PROS_DB.exists():
        try:
            return json.loads(PROS_DB.read_text())
        except Exception:
            pass
    return {}

def save_pros(pros: dict):
    PROS_DB.write_text(json.dumps(pros, indent=2, ensure_ascii=False))

pros: dict = load_pros()


# ── Analyse en arrière-plan ───────────────────────────────────────────────────
def run_analysis(job_id: str, video_path: Path):
    """Pipeline complet: détection gate → YOLO → résultats."""
    try:
        jobs[job_id]["status"]   = "processing"
        jobs[job_id]["progress"] = "Détection du gate..."

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

        jobs[job_id]["progress"] = "Pose estimation + tracking..."
        results = analyze_main(str(video_path), gate_drop=gate_drop, bip1_time=bip1_time)

        if results is None:
            raise ValueError("L'analyse n'a pas retourné de résultats.")

        results["gate_method"] = gate_method
        results["bip_times"]   = beep_times if beep_times else []

        stem         = video_path.stem.replace(f"{job_id}_", "")
        results_path = OUTPUT_DIR / f"{stem}_results.json"
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        jobs[job_id].update({
            "status":   "done",
            "progress": "Terminé",
            "stem":     stem,
            "results":  results,
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "error":  str(e),
            "detail": traceback.format_exc(),
        })


def run_pro_analysis(pro_id: str, video_path: Path):
    """Analyse un pro : génère uniquement la vidéo annotée (squelette)."""
    try:
        pros[pro_id]["status"]   = "processing"
        pros[pro_id]["progress"] = "Détection du gate..."
        save_pros(pros)

        beep_times, gate_time_audio, audio_conf = detect_beeps_audio(str(video_path))
        bip1_time = None

        if beep_times and audio_conf >= 0.40:
            gate_drop = gate_time_audio
            bip1_time = beep_times[0]
            pros[pro_id]["progress"] = "Gate détecté (audio) — génération squelette..."
        else:
            gate_frame, gate_time_visual, visual_conf, _ = detect_gate_drop(str(video_path))
            gate_drop = gate_time_visual
            pros[pro_id]["progress"] = "Gate détecté (visuel) — génération squelette..."

        if gate_drop is None:
            raise ValueError("Impossible de détecter le gate drop.")

        pros[pro_id]["progress"] = "Génération du squelette..."
        save_pros(pros)

        results = analyze_main(str(video_path), gate_drop=gate_drop, bip1_time=bip1_time)
        if results is None:
            raise ValueError("L'analyse n'a pas retourné de résultats.")

        # On ne garde que ce qu'il faut pour la banque de référence
        pros[pro_id].update({
            "status":        "done",
            "progress":      "Terminé",
            "video_file":    results["files"]["annotated_video"],
            "duration_s":    results["duration_s"],
            "fps":           results["fps"],
        })
        save_pros(pros)

    except Exception as e:
        pros[pro_id].update({
            "status": "error",
            "error":  str(e),
        })
        save_pros(pros)


# ── Routes principales ────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id     = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[job_id] = {"status": "queued", "filename": file.filename, "progress": "En attente..."}
    background_tasks.add_task(run_analysis, job_id, video_path)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {"status": job["status"], "progress": job.get("progress", ""), "error": job.get("error", "")}


@app.get("/result/{job_id}")
async def result(request: Request, job_id: str):
    job = jobs.get(job_id)
    if not job:
        return templates.TemplateResponse(request, "index.html", {"error": "Job introuvable."})
    if job["status"] != "done":
        return templates.TemplateResponse(request, "index.html", {"error": "Analyse pas encore terminée."})
    return templates.TemplateResponse(request, "result.html", {
        "job_id":  job_id,
        "results": job["results"],
    })


# ── Banque de pros ────────────────────────────────────────────────────────────
@app.get("/pros")
async def pros_page(request: Request):
    """Page d'administration de la banque de pros."""
    return templates.TemplateResponse(request, "pros.html", {
        "pros": list(pros.values()),
    })


@app.post("/pros/add")
async def add_pro(background_tasks: BackgroundTasks,
                  file: UploadFile = File(...),
                  name: str = Form(...)):
    """Upload une vidéo de pro → génère le squelette → sauvegarde dans la banque."""
    pro_id     = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"pro_{pro_id}_{file.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pros[pro_id] = {
        "id":       pro_id,
        "name":     name.strip() or file.filename,
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status":   "queued",
        "progress": "En attente...",
        "video_file": None,
    }
    save_pros(pros)
    background_tasks.add_task(run_pro_analysis, pro_id, video_path)
    return {"pro_id": pro_id}


@app.get("/pros/status/{pro_id}")
async def pro_status(pro_id: str):
    p = pros.get(pro_id)
    if not p:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {"status": p["status"], "progress": p.get("progress", ""), "error": p.get("error", "")}


@app.delete("/pros/{pro_id}")
async def delete_pro(pro_id: str):
    if pro_id in pros:
        del pros[pro_id]
        save_pros(pros)
        return {"ok": True}
    return JSONResponse({"error": "Pro introuvable."}, status_code=404)


# ── Comparaison côte à côte ───────────────────────────────────────────────────
@app.get("/compare/{job_id}")
async def compare(request: Request, job_id: str):
    """Page de comparaison: rider analysé vs un pro de la banque."""
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return templates.TemplateResponse(request, "index.html", {"error": "Job introuvable."})

    pros_done = [p for p in pros.values() if p.get("status") == "done"]
    return templates.TemplateResponse(request, "compare.html", {
        "job_id":    job_id,
        "rider":     job["results"],
        "pros_list": pros_done,
    })
