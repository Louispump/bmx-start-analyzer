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

import numpy as np
import pandas as pd

from analyze import main as analyze_main, calculate_angle
from audio_gate import detect_gate_drop
from mahieu import analyze_video as mahieu_analyze, render_debug_frame as mahieu_render

# ── Dossiers ──────────────────────────────────────────────────────────────────
UPLOAD_DIR    = Path("uploads")
OUTPUT_DIR    = Path("output")
PROS_DB       = OUTPUT_DIR / "pros_db.json"
JOBS_DB       = OUTPUT_DIR / "jobs_db.json"
ATHLETES_DB   = OUTPUT_DIR / "athletes_db.json"
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

def load_athletes() -> dict:
    if ATHLETES_DB.exists():
        try:
            return json.loads(ATHLETES_DB.read_text())
        except Exception:
            pass
    return {}

def save_athletes(athletes: dict):
    ATHLETES_DB.write_text(json.dumps(athletes, indent=2, ensure_ascii=False))

pros:     dict = load_pros()
jobs:     dict = load_jobs()
athletes: dict = load_athletes()


def _athlete_jobs(athlete_id: str) -> list[dict]:
    """Retourne tous les jobs (terminés) liés à cet athlète, triés du plus récent au plus ancien."""
    out = []
    for jid, j in jobs.items():
        if j.get("status") != "done":
            continue
        if j.get("athlete_id") != athlete_id:
            continue
        out.append({
            "job_id":      jid,
            "video_name":  j.get("results", {}).get("video_name", "—"),
            "added_at":    j.get("added_at", ""),
            "fps":         j.get("results", {}).get("fps"),
            "duration_s":  j.get("results", {}).get("duration_s"),
            "reaction":    j.get("results", {}).get("reaction", {}),
            "gate_drop_t": j.get("results", {}).get("gate_drop_t"),
        })
    out.sort(key=lambda x: x.get("added_at", ""), reverse=True)
    return out


def _athlete_stats(athlete_jobs: list[dict]) -> dict:
    """Mini dashboard : nb de vidéos, meilleur / moyen temps de réaction (excluant les faux départs)."""
    reactions_ms = []
    for j in athlete_jobs:
        r = j.get("reaction") or {}
        if r.get("type") == "false_start":
            continue
        if r.get("type") == "bip" and r.get("from_bip1_ms") is not None:
            reactions_ms.append(r["from_bip1_ms"])
        elif r.get("from_gate_ms") is not None:
            reactions_ms.append(r["from_gate_ms"])
    return {
        "n_videos":  len(athlete_jobs),
        "best_ms":   min(reactions_ms) if reactions_ms else None,
        "avg_ms":    round(sum(reactions_ms) / len(reactions_ms), 1) if reactions_ms else None,
        "false_starts": sum(1 for j in athlete_jobs
                            if (j.get("reaction") or {}).get("type") == "false_start"),
        "history":   reactions_ms,  # pour le mini-graphique
    }


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
            "status":        "done",
            "progress":      "Terminé",
            "video_file":    results["files"]["annotated_video"],
            "landmarks_csv": results["files"]["landmarks_csv"],
            "front_foot":    results.get("front_foot"),
            "gate_drop_t":   results["gate_drop_t"],
            "duration_s":    results["duration_s"],
            "fps":           results["fps"],
        })
        save_pros(pros)
    except Exception as e:
        pros[pro_id].update({"status": "error", "error": str(e)})
        save_pros(pros)


# ── Routes principales ────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    athletes_list = sorted(athletes.values(), key=lambda a: a.get("name", "").lower())
    return templates.TemplateResponse(request, "index.html",
                                      {"athletes_list": athletes_list})


@app.post("/upload")
async def upload(file: UploadFile = File(...),
                 athlete_id: str = Form("")):
    """Sauvegarde la vidéo et retourne les infos (fps, durée) pour la sélection du gate.
    `athlete_id` est optionnel : la vidéo peut exister sans athlète."""
    job_id     = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    info = get_video_info(video_path)
    aid = athlete_id.strip() or None
    if aid and aid not in athletes:
        aid = None  # athlète invalide → on ignore
    jobs[job_id] = {
        "status":     "pending_gate",
        "filename":   file.filename,
        "video_path": str(video_path),
        "video_url":  f"/uploads/{job_id}_{file.filename}",
        "athlete_id": aid,
        "added_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
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
    aid = job.get("athlete_id")
    athlete = athletes.get(aid) if aid else None
    return templates.TemplateResponse(request, "result.html", {
        "job_id":  job_id,
        "results": job["results"],
        "athlete": athlete,
    })


# ── Athlètes (dossiers) ───────────────────────────────────────────────────────
@app.get("/athletes")
async def athletes_page(request: Request):
    """Liste de tous les athlètes avec un compte de vidéos par dossier."""
    items = []
    for a in athletes.values():
        a_jobs = _athlete_jobs(a["id"])
        last = a_jobs[0]["added_at"] if a_jobs else None
        items.append({**a, "n_videos": len(a_jobs), "last_at": last})
    items.sort(key=lambda x: x.get("name", "").lower())
    return templates.TemplateResponse(request, "athletes.html",
                                      {"athletes_list": items})


@app.post("/athletes")
async def athletes_create(name: str = Form(...), notes: str = Form("")):
    """Crée un nouvel athlète. Retourne l'id pour usage immédiat (ex : depuis l'upload)."""
    name = name.strip()
    if not name:
        return JSONResponse({"error": "Nom requis."}, status_code=400)
    aid = str(uuid.uuid4())[:8]
    athletes[aid] = {
        "id":         aid,
        "name":       name,
        "notes":      notes.strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_athletes(athletes)
    return {"id": aid, "name": name}


@app.get("/athletes/{athlete_id}")
async def athlete_detail(request: Request, athlete_id: str):
    a = athletes.get(athlete_id)
    if not a:
        return templates.TemplateResponse(request, "athletes.html",
                                          {"athletes_list": [],
                                           "error": "Athlète introuvable."})
    a_jobs = _athlete_jobs(athlete_id)
    stats  = _athlete_stats(a_jobs)
    return templates.TemplateResponse(request, "athlete_detail.html", {
        "athlete":      a,
        "athlete_jobs": a_jobs,
        "stats":        stats,
    })


@app.delete("/athletes/{athlete_id}")
async def athlete_delete(athlete_id: str):
    """Supprime l'athlète. Les vidéos analysées restent (athlete_id mis à None)."""
    if athlete_id not in athletes:
        return JSONResponse({"error": "Athlète introuvable."}, status_code=404)
    del athletes[athlete_id]
    save_athletes(athletes)
    # Désassigne les jobs liés
    dirty = False
    for j in jobs.values():
        if j.get("athlete_id") == athlete_id:
            j["athlete_id"] = None
            dirty = True
    if dirty:
        save_jobs(jobs)
    return {"ok": True}


# ── Banque de pros ────────────────────────────────────────────────────────────
@app.get("/pros")
async def pros_page(request: Request):
    return templates.TemplateResponse(request, "pros.html",
                                      {"pros": list(pros.values())})


@app.get("/pros/{pro_id}/view")
async def pro_view(request: Request, pro_id: str):
    """Page de visionnage standalone d'un pro (pour montrer aux athlètes)."""
    p = pros.get(pro_id)
    if not p or p.get("status") != "done":
        return templates.TemplateResponse(request, "pros.html",
                                          {"pros": list(pros.values()),
                                           "error": "Pro introuvable ou non analysé."})
    return templates.TemplateResponse(request, "pro_view.html", {"pro": p})


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


# ── Comparaison angles à un instant T (rider vs pro) ──────────────────────────
def _infer_front_foot_from_csv(df: pd.DataFrame) -> str:
    """Fallback pour les pros déjà stockés sans `front_foot` : on choisit le côté
    dont la confiance moyenne (épaule + hanche + genou + cheville) est la plus
    haute sur l'ensemble de la vidéo."""
    cols_L = ["L_shoulder_conf", "L_hip_conf", "L_knee_conf", "L_ankle_conf"]
    cols_R = ["R_shoulder_conf", "R_hip_conf", "R_knee_conf", "R_ankle_conf"]
    if all(c in df.columns for c in cols_L + cols_R):
        L_conf = df[cols_L].mean().mean()
        R_conf = df[cols_R].mean().mean()
        return "L" if L_conf >= R_conf else "R"
    return "L"


def _normalized_skeleton(row, side: str, direction: int) -> dict | None:
    """Normalise le squelette pour overlay : hanche à l'origine, échelle = 1.0
    sur la longueur hanche→épaule, mirror si le rider est tourné à gauche
    (pour que les 2 squelettes soient comparables face à droite).

    Retourne dict {nom_articulation: [x, y]} ou None si l'ancrage est invalide.
    """
    sh_x = row.get(f"{side}_shoulder_x", np.nan)
    sh_y = row.get(f"{side}_shoulder_y", np.nan)
    hi_x = row.get(f"{side}_hip_x",      np.nan)
    hi_y = row.get(f"{side}_hip_y",      np.nan)
    if np.isnan(sh_x) or np.isnan(hi_x) or np.isnan(sh_y) or np.isnan(hi_y):
        return None
    scale = float(np.hypot(sh_x - hi_x, sh_y - hi_y))
    if scale < 1.0:
        return None

    other = "R" if side == "L" else "L"

    def _n(col_x: str, col_y: str):
        x = row.get(col_x, np.nan)
        y = row.get(col_y, np.nan)
        if np.isnan(x) or np.isnan(y):
            return None
        nx = (x - hi_x) / scale
        ny = (y - hi_y) / scale
        if direction < 0:
            nx = -nx   # mirror : tout le monde tourné à droite
        return [round(float(nx), 3), round(float(ny), 3)]

    return {
        "nose":          _n("nose_x",                "nose_y"),
        "shoulder":      _n(f"{side}_shoulder_x",    f"{side}_shoulder_y"),
        "elbow":         _n(f"{side}_elbow_x",       f"{side}_elbow_y"),
        "wrist":         _n(f"{side}_wrist_x",       f"{side}_wrist_y"),
        "hip":           _n(f"{side}_hip_x",         f"{side}_hip_y"),    # → [0, 0]
        "knee":          _n(f"{side}_knee_x",        f"{side}_knee_y"),
        "ankle":         _n(f"{side}_ankle_x",       f"{side}_ankle_y"),
        "back_shoulder": _n(f"{other}_shoulder_x",   f"{other}_shoulder_y"),
        "back_hip":      _n(f"{other}_hip_x",        f"{other}_hip_y"),
    }


def _compute_angles_at_time(csv_path: Path, t: float, side: str) -> dict | None:
    """Lit les keypoints au frame le plus proche de l'instant `t` dans le CSV
    landmarks et calcule 6 angles : knee, hip, ankle, shoulder, elbow, trunk.

    `side` ∈ {'L','R'} = côté du rider (pied avant). Le tronc est signé selon
    la direction du rider (positif = penché en avant). Le squelette normalisé
    pour l'overlay est inclus dans la réponse."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "time" not in df.columns or len(df) == 0:
        return None
    idx = int((df["time"] - t).abs().idxmin())
    row = df.loc[idx]

    # Direction (rider face droite ou gauche) à partir du nez vs centre des hanches
    nose_x  = row.get("nose_x",   np.nan)
    L_hi_x  = row.get("L_hip_x",  np.nan)
    R_hi_x  = row.get("R_hip_x",  np.nan)
    if not (np.isnan(nose_x) or np.isnan(L_hi_x) or np.isnan(R_hi_x)):
        direction = 1 if nose_x > (L_hi_x + R_hi_x) / 2 else -1
    else:
        direction = 1

    def _pt(part: str):
        x = row.get(f"{side}_{part}_x", np.nan)
        y = row.get(f"{side}_{part}_y", np.nan)
        return (float(x), float(y))

    sh = _pt("shoulder")
    el = _pt("elbow")
    wr = _pt("wrist")
    hi = _pt("hip")
    kn = _pt("knee")
    an = _pt("ankle")

    def _safe_angle(p1, p2, p3):
        if any(np.isnan(p[0]) or np.isnan(p[1]) for p in (p1, p2, p3)):
            return None
        a = float(calculate_angle(p1, p2, p3))
        return round(a, 1) if not np.isnan(a) else None

    knee_a     = _safe_angle(hi, kn, an)
    hip_a      = _safe_angle(sh, hi, kn)
    elbow_a    = _safe_angle(sh, el, wr)
    shoulder_a = _safe_angle(hi, sh, el)
    # Cheville : proxy pointe pied (+30 px dans direction du rider)
    ankle_a = None
    if not (np.isnan(an[0]) or np.isnan(kn[0])):
        toe_proxy = (an[0] + 30 * direction, an[1])
        ankle_a = _safe_angle(kn, an, toe_proxy)

    # Tronc : angle (hip→shoulder) vs verticale, signé selon direction
    # Convention : positif = penché en AVANT du rider, 0 = vertical, négatif = en arrière
    trunk_a = None
    if not (np.isnan(sh[0]) or np.isnan(hi[0])):
        dx = sh[0] - hi[0]
        dy = sh[1] - hi[1]   # négatif = épaule au-dessus (image y vers le bas)
        trunk_a = round(float(np.degrees(np.arctan2(direction * dx, -dy))), 1)

    return {
        "frame":    int(idx),
        "t":        round(float(row["time"]), 3),
        "side":     side,
        "knee":     knee_a,
        "hip":      hip_a,
        "ankle":    ankle_a,
        "shoulder": shoulder_a,
        "elbow":    elbow_a,
        "trunk":    trunk_a,
        "skeleton": _normalized_skeleton(row, side, direction),
    }


@app.post("/compare_angles/{job_id}")
async def compare_angles(job_id: str,
                         pro_id:    str   = Form(...),
                         rider_t:   float = Form(...),
                         pro_t:     float = Form(...)):
    """Calcule et compare les 6 angles articulaires (rider vs pro) aux instants
    spécifiés. Lit les CSV landmarks déjà générés par l'analyse — pas de
    ré-analyse ni de re-pose-detection."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    pro = pros.get(pro_id)
    if not pro or pro.get("status") != "done":
        return JSONResponse({"error": "Pro introuvable ou non analysé."},
                            status_code=404)

    rider_results = job["results"]
    rider_csv     = OUTPUT_DIR / rider_results["files"]["landmarks_csv"]
    rider_side    = rider_results.get("front_foot") or "L"

    pro_csv_name  = pro.get("landmarks_csv")
    if not pro_csv_name:
        # Pro analysé avant l'ajout du champ — on dérive depuis la vidéo annotée
        pro_csv_name = pro["video_file"].replace("_annotated.mp4", "_landmarks.csv")
    pro_csv = OUTPUT_DIR / pro_csv_name

    # Fallback : déduire le côté pied avant si non stocké côté pro
    pro_side = pro.get("front_foot")
    if not pro_side and pro_csv.exists():
        pro_side = _infer_front_foot_from_csv(pd.read_csv(pro_csv))
    pro_side = pro_side or "L"

    rider_angles = _compute_angles_at_time(rider_csv, rider_t, rider_side)
    pro_angles   = _compute_angles_at_time(pro_csv,   pro_t,   pro_side)

    if rider_angles is None or pro_angles is None:
        return JSONResponse(
            {"error": f"Lecture CSV impossible (rider={rider_csv.exists()}, pro={pro_csv.exists()})"},
            status_code=500,
        )

    return {
        "rider":   rider_angles,
        "pro":     pro_angles,
        "pro_name": pro.get("name", "Pro"),
    }


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

