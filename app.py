"""
BMX Start Analyzer — Web App

Usage:
  uvicorn app:app --reload --port 8000
"""

import os
os.environ["MPLBACKEND"] = "agg"

import uuid
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime

import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader

from analyze import main as analyze_main
from audio_gate import detect_gate_drop
from mahieu import analyze_video as mahieu_analyze, render_debug_frame as mahieu_render

# ── Dossiers ──────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
PROS_DB    = OUTPUT_DIR / "pros_db.json"
JOBS_DB    = OUTPUT_DIR / "jobs_db.json"
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
def load_pros() -> dict:
    if PROS_DB.exists():
        try:
            return json.loads(PROS_DB.read_text())
        except Exception:
            pass
    return {}

def save_pros(pros: dict):
    PROS_DB.write_text(json.dumps(pros, indent=2, ensure_ascii=False))

def load_jobs() -> dict:
    if JOBS_DB.exists():
        try:
            return json.loads(JOBS_DB.read_text())
        except Exception:
            pass
    return {}

def save_jobs(jobs: dict):
    # On ne persiste que les jobs terminés (status == done)
    done = {k: v for k, v in jobs.items() if v.get("status") == "done"}
    JOBS_DB.write_text(json.dumps(done, indent=2, ensure_ascii=False))

pros: dict = load_pros()
jobs: dict = load_jobs()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_video_info(video_path: Path) -> dict:
    """Retourne fps et durée d'une vidéo."""
    cap      = cv2.VideoCapture(str(video_path))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": round(fps, 3), "n_frames": n_frames,
            "duration_s": round(n_frames / fps, 2)}


# ── Analyse en arrière-plan ───────────────────────────────────────────────────
def run_analysis(job_id: str, video_path: Path, gate_drop: float):
    try:
        jobs[job_id]["status"]   = "processing"
        jobs[job_id]["progress"] = "Pose estimation + tracking..."

        results = analyze_main(str(video_path), gate_drop=gate_drop)

        if results is None:
            raise ValueError("L'analyse n'a pas retourné de résultats.")

        results["gate_method"] = "manual"
        results["job_id"]      = job_id

        stem         = video_path.stem.replace(f"{job_id}_", "")
        results_path = OUTPUT_DIR / f"{job_id}_results.json"
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        jobs[job_id].update({
            "status":   "done",
            "progress": "Terminé",
            "stem":     stem,
            "results":  results,
        })
        save_jobs(jobs)
    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "error":  str(e),
            "detail": traceback.format_exc(),
        })


def run_pro_analysis(pro_id: str, video_path: Path, gate_drop: float):
    try:
        pros[pro_id]["status"]   = "processing"
        pros[pro_id]["progress"] = "Génération du squelette..."
        save_pros(pros)

        results = analyze_main(str(video_path), gate_drop=gate_drop)
        if results is None:
            raise ValueError("L'analyse n'a pas retourné de résultats.")

        pros[pro_id].update({
            "status":      "done",
            "progress":    "Terminé",
            "video_file":  results["files"]["annotated_video"],
            "gate_drop_t": results["gate_drop_t"],
            "duration_s":  results["duration_s"],
            "fps":         results["fps"],
        })
        save_pros(pros)
    except Exception as e:
        pros[pro_id].update({"status": "error", "error": str(e)})
        save_pros(pros)


# ── Routes principales ────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Sauvegarde la vidéo et retourne les infos (fps, durée) pour la sélection du gate."""
    job_id     = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    info = get_video_info(video_path)
    jobs[job_id] = {
        "status":     "pending_gate",
        "filename":   file.filename,
        "video_path": str(video_path),
        "video_url":  f"/uploads/{job_id}_{file.filename}",
        **info,
    }
    return {
        "job_id":    job_id,
        "video_url": jobs[job_id]["video_url"],
        "fps":       info["fps"],
        "n_frames":  info["n_frames"],
    }


@app.post("/detect_gate/{job_id}")
async def detect_gate(job_id: str):
    """Tente de détecter le gate drop via la cadence audio UCI (4 beeps)."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    res = detect_gate_drop(Path(job["video_path"]))
    if res.get("detected"):
        gate_frame = int(round(res["gate_t"] * job["fps"]))
        gate_frame = max(0, min(gate_frame, job["n_frames"] - 1))
        return {
            "detected":         True,
            "gate_frame":       gate_frame,
            "gate_time":        res["gate_t"],
            "beeps_t":          res["beeps_t"],
            "mean_interval_ms": res["mean_interval_ms"],
            "confidence":       res["confidence"],
        }
    return {"detected": False, "reason": res.get("reason", "unknown")}


@app.post("/start/{job_id}")
async def start_analysis(background_tasks: BackgroundTasks,
                         job_id: str,
                         gate_frame: int = Form(...)):
    """Démarre l'analyse avec le gate frame choisi manuellement."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)

    fps       = job["fps"]
    gate_drop = gate_frame / fps
    video_path = Path(job["video_path"])

    jobs[job_id]["status"]    = "queued"
    jobs[job_id]["progress"]  = "En attente..."
    jobs[job_id]["gate_drop"] = gate_drop

    background_tasks.add_task(run_analysis, job_id, video_path, gate_drop)
    return {"ok": True}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {"status": job["status"], "progress": job.get("progress", ""),
            "error": job.get("error", "")}



def get_job_or_recover(job_id: str) -> dict | None:
    """Retourne le job depuis la mémoire ou le recharge depuis le fichier disque."""
    if job_id in jobs and jobs[job_id].get("status") == "done":
        return jobs[job_id]
    results_path = OUTPUT_DIR / f"{job_id}_results.json"
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
            jobs[job_id] = {"status": "done", "results": results}
            return jobs[job_id]
        except Exception:
            pass
    return None


@app.get("/result/{job_id}")
async def result(request: Request, job_id: str):
    job = get_job_or_recover(job_id)
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


# ── Banque de pros ────────────────────────────────────────────────────────────
@app.get("/pros")
async def pros_page(request: Request):
    return templates.TemplateResponse(request, "pros.html",
                                      {"pros": list(pros.values())})


@app.post("/pros/upload")
async def pros_upload(file: UploadFile = File(...), name: str = Form(...)):
    """Sauvegarde la vidéo du pro et retourne les infos pour sélection du gate."""
    pro_id     = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"pro_{pro_id}_{file.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    info = get_video_info(video_path)
    pros[pro_id] = {
        "id":         pro_id,
        "name":       name.strip() or file.filename,
        "added_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status":     "pending_gate",
        "video_path": str(video_path),
        "video_url":  f"/uploads/pro_{pro_id}_{file.filename}",
        "video_file": None,
        **info,
    }
    save_pros(pros)
    return {
        "pro_id":    pro_id,
        "video_url": pros[pro_id]["video_url"],
        "fps":       info["fps"],
        "n_frames":  info["n_frames"],
    }


@app.post("/pros/detect_gate/{pro_id}")
async def pros_detect_gate(pro_id: str):
    """Tente de détecter le gate drop via la cadence audio UCI (4 beeps)."""
    p = pros.get(pro_id)
    if not p:
        return JSONResponse({"error": "Pro introuvable."}, status_code=404)
    res = detect_gate_drop(Path(p["video_path"]))
    if res.get("detected"):
        gate_frame = int(round(res["gate_t"] * p["fps"]))
        gate_frame = max(0, min(gate_frame, p["n_frames"] - 1))
        return {
            "detected":         True,
            "gate_frame":       gate_frame,
            "gate_time":        res["gate_t"],
            "beeps_t":          res["beeps_t"],
            "mean_interval_ms": res["mean_interval_ms"],
            "confidence":       res["confidence"],
        }
    return {"detected": False, "reason": res.get("reason", "unknown")}


@app.post("/pros/start/{pro_id}")
async def pros_start(background_tasks: BackgroundTasks,
                     pro_id: str,
                     gate_frame: int = Form(...)):
    """Démarre la génération du squelette du pro avec le gate frame choisi."""
    p = pros.get(pro_id)
    if not p:
        return JSONResponse({"error": "Pro introuvable."}, status_code=404)

    fps        = p["fps"]
    gate_drop  = gate_frame / fps
    video_path = Path(p["video_path"])

    pros[pro_id]["status"]   = "queued"
    pros[pro_id]["progress"] = "En attente..."
    save_pros(pros)

    background_tasks.add_task(run_pro_analysis, pro_id, video_path, gate_drop)
    return {"ok": True}


@app.get("/pros/status/{pro_id}")
async def pro_status(pro_id: str):
    p = pros.get(pro_id)
    if not p:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {"status": p["status"], "progress": p.get("progress", ""),
            "error": p.get("error", "")}


@app.delete("/pros/{pro_id}")
async def delete_pro(pro_id: str):
    if pro_id in pros:
        del pros[pro_id]
        save_pros(pros)
        return {"ok": True}
    return JSONResponse({"error": "Pro introuvable."}, status_code=404)


# ── Plateforme de test détection audio ────────────────────────────────────────
AUDIO_CACHE = OUTPUT_DIR / "audio_detection_db.json"

def _load_audio_cache() -> dict:
    if AUDIO_CACHE.exists():
        try:
            return json.loads(AUDIO_CACHE.read_text())
        except Exception:
            pass
    return {}


@app.get("/test/audio")
async def test_audio(request: Request):
    """Page de vérification visuelle de la détection audio sur toutes les vidéos
    du dossier uploads/. Résultats cachés sur disque (clé = nom + mtime)."""
    cache = _load_audio_cache()
    videos = []
    dirty  = False
    for p in sorted(UPLOAD_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv"):
            continue
        info = get_video_info(p)
        key  = f"{p.name}::{int(p.stat().st_mtime)}"
        cached = cache.get(key)
        if cached:
            detect = cached
        else:
            detect = detect_gate_drop(p)
            cache[key] = detect
            dirty = True
        gate_frame = None
        if detect.get("detected"):
            gate_frame = int(round(detect["gate_t"] * info["fps"]))
            gate_frame = max(0, min(gate_frame, info["n_frames"] - 1))
        videos.append({
            "filename":   p.name,
            "url":        f"/uploads/{p.name}",
            "fps":        info["fps"],
            "duration_s": info["duration_s"],
            "n_frames":   info["n_frames"],
            "detect":     detect,
            "gate_frame": gate_frame,
        })
    if dirty:
        AUDIO_CACHE.write_text(json.dumps(cache, indent=2, default=str))
    return templates.TemplateResponse(request, "test_audio.html",
                                      {"videos": videos})


# ── Plateforme de test alignement Mahieu ──────────────────────────────────────
MAHIEU_CACHE = OUTPUT_DIR / "mahieu_db.json"

def _load_mahieu_cache() -> dict:
    if MAHIEU_CACHE.exists():
        try:
            return json.loads(MAHIEU_CACHE.read_text())
        except Exception:
            pass
    return {}


@app.get("/test/mahieu")
async def test_mahieu(request: Request):
    """Vérification visuelle de l'alignement Mahieu sur les vidéos d'uploads/."""
    debug_dir = OUTPUT_DIR / "mahieu_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_mahieu_cache()
    dirty = False
    videos = []
    for p in sorted(UPLOAD_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv"):
            continue
        info = get_video_info(p)
        key  = f"{p.name}::{int(p.stat().st_mtime)}"
        cached = cache.get(key)
        if cached:
            entry = cached
        else:
            res = mahieu_analyze(p)
            debug_url = None
            if res.get("detected") and res.get("best_frame"):
                bf  = res["best_frame"]
                png = debug_dir / f"{p.stem}.jpg"
                if mahieu_render(p, bf["frame"], bf["metric"], bf["side"], png):
                    debug_url = f"/output/mahieu_debug/{png.name}"
            entry = {"res": res, "debug_url": debug_url}
            cache[key] = entry
            dirty = True
        videos.append({
            "filename":   p.name,
            "url":        f"/uploads/{p.name}",
            "fps":        info["fps"],
            "duration_s": info["duration_s"],
            "n_frames":   info["n_frames"],
            "res":        entry["res"],
            "debug_url":  entry["debug_url"],
        })
    if dirty:
        MAHIEU_CACHE.write_text(json.dumps(cache, indent=2, default=str))
    return templates.TemplateResponse(request, "test_mahieu.html",
                                      {"videos": videos})


@app.post("/test/mahieu/clear")
async def clear_mahieu_cache():
    if MAHIEU_CACHE.exists():
        MAHIEU_CACHE.unlink()
    return {"ok": True}


# ── Comparaison ───────────────────────────────────────────────────────────────
@app.get("/compare/{job_id}")
async def compare(request: Request, job_id: str):
    job = get_job_or_recover(job_id)
    if not job or job["status"] != "done":
        return templates.TemplateResponse(request, "index.html",
                                          {"error": "Job introuvable."})
    pros_done = [p for p in pros.values() if p.get("status") == "done"]
    return templates.TemplateResponse(request, "compare.html", {
        "job_id":    job_id,
        "rider":     job["results"],
        "pros_list": pros_done,
    })

