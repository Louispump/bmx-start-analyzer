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
from datetime import datetime, timedelta

import cv2
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, Response
from jinja2 import Environment, FileSystemLoader

import numpy as np
import pandas as pd

from analyze import main as analyze_main, calculate_angle, \
    segment_phases as analyze_segment_phases, PHASE_COLORS
from audio_gate import detect_gate_drop
from mahieu import analyze_video as mahieu_analyze, render_debug_frame as mahieu_render

# ── Constantes ────────────────────────────────────────────────────────────────
# Cadence UCI BMX officielle (Random Cadence 2007, doc rad-gate) :
#   beep 1 (60ms) · pause 60ms · beep 2 (60ms) · pause 60ms · beep 3 (60ms)
#   · pause 60ms · beep 4 / gate drop
# Soit 3 × 60 + 3 × 60 = 360 ms entre le DÉBUT du beep 1 et le gate drop.
# Utilisé pour dériver le bip 1 quand l'audio n'a pas pu être détecté et que
# le user a marqué le gate drop manuellement.
UCI_BIP1_TO_GATE_S = 0.360

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


# Empêche le cache navigateur sur les pages HTML et les CSS/JS — sinon Safari
# (surtout iPad) garde l'ancienne version pendant plusieurs minutes après une
# mise à jour du code, ce qui donne l'impression que rien n'a changé.
@app.middleware("http")
async def no_cache_for_html(request, call_next):
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if ct.startswith("text/html") or ct.startswith("text/css") \
       or ct.startswith("application/javascript"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"]        = "no-cache"
        response.headers["Expires"]       = "0"
    return response


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

# ── Sauvegardes automatiques ─────────────────────────────────────────────────
# Tout write d'une DB JSON :
#   1. Atomic : write dans un .tmp puis os.replace (jamais de fichier corrompu
#      mi-écrit même si Python crashe en plein milieu).
#   2. Snapshot du résultat dans output/backups/YYYY-MM-DD/<filename>. Un seul
#      snapshot par jour (overwrite intra-jour), retention 14 jours. Si tu
#      découvres demain qu'un job a été pourri, t'as la version d'hier.
BACKUP_DIR             = OUTPUT_DIR / "backups"
BACKUP_RETENTION_DAYS  = 14

def _atomic_write_text(path: Path, content: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)

def _snapshot_db(path: Path):
    if not path.exists(): return
    today = datetime.now().strftime("%Y-%m-%d")
    dest_dir = BACKUP_DIR / today
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest_dir / path.name)

def _prune_old_backups():
    if not BACKUP_DIR.exists(): return
    cutoff = datetime.now() - timedelta(days=BACKUP_RETENTION_DAYS)
    for d in BACKUP_DIR.iterdir():
        if not d.is_dir(): continue
        try:
            dt = datetime.strptime(d.name, "%Y-%m-%d")
            if dt < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except ValueError:
            pass  # nom de dossier inattendu → on touche pas

def _save_db_with_backup(path: Path, content: str):
    _atomic_write_text(path, content)
    try:
        _snapshot_db(path)
        _prune_old_backups()
    except Exception as e:
        # Une erreur de snapshot ne doit JAMAIS empêcher le save principal
        print(f"[backup] snapshot failed for {path.name}: {e}")

def _backup_status() -> dict:
    if not BACKUP_DIR.exists():
        return {"snapshot_count": 0, "last_snapshot": None, "total_kb": 0}
    folders = sorted([d.name for d in BACKUP_DIR.iterdir() if d.is_dir()])
    total = 0
    for f in BACKUP_DIR.rglob("*"):
        if f.is_file():
            try: total += f.stat().st_size
            except Exception: pass
    return {
        "snapshot_count": len(folders),
        "last_snapshot":  folders[-1] if folders else None,
        "total_kb":       total // 1024,
        "retention_days": BACKUP_RETENTION_DAYS,
    }


def save_pros(pros: dict):
    _save_db_with_backup(PROS_DB, json.dumps(pros, indent=2, ensure_ascii=False))

def load_jobs() -> dict:
    if JOBS_DB.exists():
        try:
            return json.loads(JOBS_DB.read_text())
        except Exception:
            pass
    return {}

def save_jobs(jobs: dict):
    # On ne persiste que les jobs terminés ET rattachés à un athlète.
    # Les jobs sans athlete_id sont éphémères : ils disparaissent au prochain
    # redémarrage du serveur (mais leurs fichiers sur disque restent jusqu'à
    # la purge manuelle depuis /settings).
    done = {k: v for k, v in jobs.items()
            if v.get("status") == "done" and v.get("athlete_id")}
    _save_db_with_backup(JOBS_DB, json.dumps(done, indent=2, ensure_ascii=False))

def load_athletes() -> dict:
    if ATHLETES_DB.exists():
        try:
            return json.loads(ATHLETES_DB.read_text())
        except Exception:
            pass
    return {}

def save_athletes(athletes: dict):
    _save_db_with_backup(ATHLETES_DB, json.dumps(athletes, indent=2, ensure_ascii=False))

pros:     dict = load_pros()
jobs:     dict = load_jobs()
athletes: dict = load_athletes()


# ── Nettoyage des jobs orphelins (sans athlète) ──────────────────────────────
def _delete_job_artifacts(job_id: str, job: dict | None = None) -> int:
    """Supprime tous les fichiers sur disque liés à ce job.
    Retourne le nb de fichiers supprimés. Idempotent."""
    n = 0
    # Fichier source uploadé
    if job:
        vp = job.get("video_path")
        if vp:
            p = Path(vp)
            try:
                if p.exists(): p.unlink(); n += 1
            except Exception:
                pass
    # Tous les artefacts en sortie commencent par "{job_id}_"
    for d in (UPLOAD_DIR, OUTPUT_DIR):
        if not d.exists(): continue
        for f in d.glob(f"{job_id}_*"):
            try:
                f.unlink(); n += 1
            except Exception:
                pass
    return n


def _orphan_job_ids() -> list[str]:
    """Jobs en mémoire sans athlete_id (status=done)."""
    return [jid for jid, j in jobs.items()
            if j.get("status") == "done" and not j.get("athlete_id")]


def _orphan_files_on_disk() -> list[Path]:
    """Fichiers dont le préfixe ne correspond à aucun job/pro encore vivant.
    Règles :
      - `pro_<id>_*`     → garder si `<id>` est dans pros
      - `manual_<id>_*`  → JAMAIS purgé (uploads /manual_reaction, pas de job assoc)
      - `<job_id>_*`     → garder si `<job_id>` est dans jobs
      - `*_db.json`, `.DS_Store`, debug images → ignorés
      - autres préfixes inconnus → orphelins
    """
    valid_jobs = set(jobs.keys())
    valid_pros = set(pros.keys())

    def is_orphan(name: str) -> bool:
        if name == ".DS_Store":             return False
        if name.endswith("_db.json"):       return False
        if name.startswith("manual_"):      return False  # uploads manual_reaction
        if name.startswith("debug_"):       return True   # diagnostics → purgeables
        if name.startswith("pro_"):
            # format attendu : pro_<id>_<rest>
            rest = name[4:]
            if "_" not in rest: return True  # malformé
            pid = rest.split("_", 1)[0]
            return pid not in valid_pros
        # Sinon : préfixe = tout avant le 1er "_"
        prefix = name.split("_", 1)[0]
        # Si pas de "_" du tout, c'est un fichier sans préfixe → orphelin
        if "_" not in name: return True
        return prefix not in valid_jobs

    out: list[Path] = []
    for d in (OUTPUT_DIR, UPLOAD_DIR):
        if not d.exists(): continue
        for f in d.iterdir():
            if not f.is_file(): continue
            if is_orphan(f.name):
                out.append(f)
    return out


def purge_orphans() -> dict:
    """Purge les jobs orphelins en mémoire ET les fichiers orphelins sur disque.
    Retourne un récap pour l'UI."""
    n_jobs   = 0
    n_files  = 0
    for jid in _orphan_job_ids():
        j = jobs.pop(jid, None)
        if j: n_jobs += 1
        n_files += _delete_job_artifacts(jid, j)
    for f in _orphan_files_on_disk():
        try:
            f.unlink(); n_files += 1
        except Exception:
            pass
    # On ne persiste plus les orphelins de toute façon, mais on garde
    # l'appel pour rester cohérent si autre chose a changé.
    save_jobs(jobs)
    return {"jobs_removed": n_jobs, "files_removed": n_files}


_MONTHS_FR = ["", "janv.", "févr.", "mars", "avr.", "mai", "juin",
              "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def _format_date_fr(added_at: str) -> str:
    """'2026-05-21 14:32' → '21 mai 2026'. Fallback : renvoie added_at brut."""
    date_part = (added_at or "").split(" ")[0]
    try:
        Y, M, D = date_part.split("-")
        return f"{int(D)} {_MONTHS_FR[int(M)]} {int(Y)}"
    except Exception:
        return date_part


def _display_name(job_id: str, job: dict) -> str:
    """Nom d'affichage : `Athlete — 5 juin 2026 — 04`.
    `04` = position chronologique de la vidéo pour cet athlète dans la journée.
    Si `custom_name` est défini sur le job, c'est lui qui est utilisé.
    Sans athlète, on retombe sur le nom de fichier original."""
    custom = (job.get("custom_name") or "").strip()
    if custom:
        return custom
    aid = job.get("athlete_id")
    if not aid:
        return job.get("results", {}).get("video_name") \
            or job.get("filename") or "—"
    athlete = athletes.get(aid)
    name = athlete.get("name") if athlete else "Athlète inconnu"
    added_at = job.get("added_at", "")
    if not added_at:
        return name
    date_part = added_at.split(" ")[0]
    date_fr   = _format_date_fr(added_at)
    # Numéro dans la journée
    same_day: list[tuple[str, str]] = []
    for jid, j in jobs.items():
        if j.get("status") != "done": continue
        if j.get("athlete_id") != aid: continue
        if (j.get("added_at") or "").split(" ")[0] != date_part: continue
        same_day.append((j.get("added_at", ""), jid))
    same_day.sort()
    seq = next((i + 1 for i, (_, jid) in enumerate(same_day) if jid == job_id), 1)
    return f"{name} — {date_fr} — {seq:02d}"


# ── Tags + notes par session ─────────────────────────────────────────────────
# 4 tags fixes (un seul par job). Stockés en lowercase, affichés via TAG_LABELS.
VALID_TAGS  = {"course", "entrainement", "warmup", "pb"}
TAG_LABELS  = {
    "course":       "Course",
    "entrainement": "Entraînement",
    "warmup":       "Warmup",
    "pb":           "PB",
}


def _job_compare_entry(jid: str, j: dict, athlete_name: str | None = None) -> dict:
    """Format compact d'un job pour le sélecteur "Comparer avec" sur la page
    Compare. Retourne tout ce qu'il faut pour piloter le panneau référence."""
    r = j.get("results", {})
    return {
        "id":               jid,
        "video_name":       r.get("video_name", j.get("filename", "—")),
        "display_name":     _display_name(jid, j),
        "annotated_video":  r.get("files", {}).get("annotated_video", ""),
        "landmarks_csv":    r.get("files", {}).get("landmarks_csv", ""),
        "fps":              r.get("fps"),
        "gate_drop_t":      r.get("gate_drop_t"),
        "front_foot":       r.get("front_foot"),
        "added_at":         j.get("added_at", ""),
        "athlete_id":       j.get("athlete_id"),
        "athlete_name":     athlete_name,
        "tag":              j.get("tag"),
        "tag_label":        TAG_LABELS.get(j.get("tag") or "", ""),
    }


def _athlete_jobs(athlete_id: str, filter_tag: str | None = None) -> list[dict]:
    """Retourne tous les jobs (terminés) liés à cet athlète, triés du plus récent
    au plus ancien. Si `filter_tag` est défini, ne garde que les jobs avec ce tag."""
    out = []
    for jid, j in jobs.items():
        if j.get("status") != "done":
            continue
        if j.get("athlete_id") != athlete_id:
            continue
        if filter_tag is not None and j.get("tag") != filter_tag:
            continue
        out.append({
            "job_id":       jid,
            "video_name":   j.get("results", {}).get("video_name", "—"),
            "display_name": _display_name(jid, j),
            "added_at":     j.get("added_at", ""),
            "fps":          j.get("results", {}).get("fps"),
            "duration_s":   j.get("results", {}).get("duration_s"),
            "reaction":     j.get("results", {}).get("reaction", {}),
            "gate_drop_t":  j.get("results", {}).get("gate_drop_t"),
            "excluded":     bool(j.get("excluded_from_stats", False)),
            "tag":          j.get("tag"),
            "tag_label":    TAG_LABELS.get(j.get("tag") or "", ""),
            "notes":        j.get("notes", ""),
        })
    out.sort(key=lambda x: x.get("added_at", ""), reverse=True)
    return out


def _athlete_tag_counts(athlete_id: str) -> dict:
    """Compteur de jobs par tag pour cet athlète (utilisé par les pills de filtre)."""
    counts = {"total": 0, "untagged": 0}
    for tag in VALID_TAGS:
        counts[tag] = 0
    for jid, j in jobs.items():
        if j.get("status") != "done": continue
        if j.get("athlete_id") != athlete_id: continue
        counts["total"] += 1
        t = j.get("tag")
        if t in VALID_TAGS:
            counts[t] += 1
        else:
            counts["untagged"] += 1
    return counts


def _athlete_stats(athlete_jobs: list[dict]) -> dict:
    """Mini dashboard : nb de vidéos, meilleur / moyen temps de réaction (excluant
    les faux départs ET les sessions cochées comme exclues par l'utilisateur).
    Métrique = `from_bip1_ms` (bip 1 → premier mouvement) si l'audio a été détecté,
    sinon fallback sur `from_gate_ms`."""
    reactions_ms = []
    n_excluded   = 0
    for j in athlete_jobs:
        if j.get("excluded"):
            n_excluded += 1
            continue
        r = j.get("reaction") or {}
        if r.get("type") == "false_start":
            continue
        if r.get("from_bip1_ms") is not None:
            reactions_ms.append(r["from_bip1_ms"])
        elif r.get("from_gate_ms") is not None:
            reactions_ms.append(r["from_gate_ms"])
    return {
        "n_videos":  len(athlete_jobs),
        "n_excluded": n_excluded,
        "best_ms":   min(reactions_ms) if reactions_ms else None,
        "avg_ms":    round(sum(reactions_ms) / len(reactions_ms), 1) if reactions_ms else None,
        "false_starts": sum(1 for j in athlete_jobs
                            if not j.get("excluded")
                            and (j.get("reaction") or {}).get("type") == "false_start"),
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
def run_analysis(job_id: str, video_path: Path, gate_drop: float,
                 bip1_time: float | None = None):
    try:
        jobs[job_id]["status"]   = "processing"
        jobs[job_id]["progress"] = "Pose estimation + tracking..."

        results = analyze_main(str(video_path), gate_drop=gate_drop,
                                bip1_time=bip1_time)

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


def run_pro_analysis(pro_id: str, video_path: Path, gate_drop: float,
                     bip1_time: float | None = None):
    try:
        pros[pro_id]["status"]   = "processing"
        pros[pro_id]["progress"] = "Génération du squelette..."
        save_pros(pros)

        results = analyze_main(str(video_path), gate_drop=gate_drop,
                                bip1_time=bip1_time)
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
                         gate_frame: int = Form(...),
                         bip1_time: float = Form(-1.0)):
    """Démarre l'analyse avec le gate frame choisi et (optionnel) le temps du
    1er bip pour ancrer le temps de réaction depuis le bip 1."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)

    fps       = job["fps"]
    gate_drop = gate_frame / fps
    video_path = Path(job["video_path"])
    # Si l'audio n'a pas détecté de bip 1, on le dérive à partir du gate drop
    # et de la cadence UCI fixe (360 ms entre bip 1 et gate drop) — comme ça
    # le temps de réaction "depuis bip 1" reste calculé même sans audio.
    if bip1_time > 0:
        bip1 = bip1_time
    else:
        bip1 = max(0.0, gate_drop - UCI_BIP1_TO_GATE_S)

    jobs[job_id]["status"]    = "queued"
    jobs[job_id]["progress"]  = "En attente..."
    jobs[job_id]["gate_drop"] = gate_drop
    jobs[job_id]["bip1_time"] = bip1

    background_tasks.add_task(run_analysis, job_id, video_path, gate_drop, bip1)
    return {"ok": True}


@app.post("/result/{job_id}/adjust_gate")
async def adjust_gate(job_id: str, gate_frame: int = Form(...)):
    """Recalcule les phases / temps de réaction avec un nouveau gate drop,
    sans repasser par la pose detection. Utilise le CSV landmarks déjà
    généré. < 1 seconde au lieu de 1-3 min pour une ré-analyse complète."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."}, status_code=404)
    results  = job.get("results", {})
    csv_name = results.get("files", {}).get("landmarks_csv")
    if not csv_name:
        return JSONResponse({"error": "CSV landmarks introuvable."}, status_code=500)
    csv_path = OUTPUT_DIR / csv_name
    if not csv_path.exists():
        return JSONResponse({"error": f"Fichier CSV manquant : {csv_name}"}, status_code=500)

    fps      = float(results.get("fps") or job.get("fps") or 30)
    new_gate = gate_frame / fps

    # bip1 : on garde la valeur audio si elle existe (détection fiable),
    # sinon on dérive selon la cadence UCI fixe (gate − 360 ms).
    audio_detected = results.get("gate_method") == "audio"
    old_bip1       = job.get("bip1_time")
    if audio_detected and old_bip1 is not None:
        new_bip1 = old_bip1
    else:
        new_bip1 = max(0.0, new_gate - UCI_BIP1_TO_GATE_S)

    df   = pd.read_csv(csv_path)
    side = results.get("front_foot") or "L"
    ankle_col = f"{side}_ankle_y"

    try:
        phases, first_move_idx, reaction_type = analyze_segment_phases(
            df, new_gate, ankle_col, bip1_time=new_bip1
        )
    except Exception as e:
        return JSONResponse({"error": f"Erreur recalcul phases : {e}"}, status_code=500)

    t_move          = float(df.loc[first_move_idx, "time"])
    react_from_gate = t_move - new_gate
    react_from_bip  = (t_move - new_bip1) if new_bip1 else None

    phases_list = []
    for phase_name, (start, end) in phases.items():
        t_start = float(df.loc[start, "time"])
        t_end   = float(df.loc[end,   "time"])
        phases_list.append({
            "name":        phase_name,
            "start_t":     round(t_start, 3),
            "end_t":       round(t_end,   3),
            "duration_ms": round((t_end - t_start) * 1000),
            "color":       PHASE_COLORS.get(phase_name, "#eeeeee"),
        })

    # MAJ du job
    results["gate_drop_t"] = round(float(new_gate), 3)
    results["reaction"]    = {
        "type":         reaction_type,
        "first_move_t": round(t_move, 3),
        "from_gate_ms": round(react_from_gate * 1000),
        "from_bip1_ms": round(react_from_bip * 1000) if react_from_bip is not None else None,
    }
    results["phases"]   = phases_list
    job["gate_drop"]    = new_gate
    job["bip1_time"]    = new_bip1
    save_jobs(jobs)
    return {"ok": True, "gate_drop_t": new_gate, "first_move_t": t_move}


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
    # Liste triée des athlètes pour le picker "Assigner / changer"
    athletes_options = sorted(
        [{"id": a["id"], "name": a["name"]} for a in athletes.values()],
        key=lambda x: x["name"].lower(),
    )
    return templates.TemplateResponse(request, "result.html", {
        "job_id":           job_id,
        "results":          job["results"],
        "athlete":          athlete,
        "athletes_options": athletes_options,
        "display_name":     _display_name(job_id, job),
        "custom_name":      job.get("custom_name", ""),
        "auto_name":        _display_name(job_id, {**job, "custom_name": ""}),
        "current_tag":      job.get("tag", ""),
        "current_notes":    job.get("notes", ""),
        "tag_options":      [{"id": k, "label": v} for k, v in TAG_LABELS.items()],
    })


@app.post("/jobs/{job_id}/name")
async def jobs_set_name(job_id: str, custom_name: str = Form("")):
    """Définit un nom personnalisé pour la vidéo. Chaîne vide = retour au nom auto."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    name = (custom_name or "").strip()
    if len(name) > 120:
        return JSONResponse({"error": "Trop long (max 120 caractères)."}, status_code=400)
    if name:
        job["custom_name"] = name
    else:
        job.pop("custom_name", None)
    save_jobs(jobs)
    return {
        "ok":           True,
        "display_name": _display_name(job_id, job),
        "auto_name":    _display_name(job_id, {**job, "custom_name": ""}),
        "custom_name":  job.get("custom_name", ""),
    }


@app.post("/jobs/{job_id}/exclude")
async def jobs_set_excluded(job_id: str, excluded: str = Form("0")):
    """Marque une session comme exclue des stats / graphique. 1 = exclue, 0 = incluse."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    job["excluded_from_stats"] = (excluded == "1")
    save_jobs(jobs)
    return {"ok": True, "excluded": job["excluded_from_stats"]}


@app.post("/jobs/{job_id}/athlete")
async def jobs_set_athlete(job_id: str, athlete_id: str = Form("")):
    """Assigne / change / retire l'athlète d'un job. Assigner persiste le job
    (il sortira de l'état éphémère). Retirer le rendra à nouveau éphémère."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    aid = athlete_id.strip() or None
    if aid and aid not in athletes:
        return JSONResponse({"error": "Athlète introuvable."}, status_code=400)
    job["athlete_id"] = aid
    save_jobs(jobs)
    return {
        "ok":         True,
        "athlete_id": aid,
        "athlete":    athletes.get(aid) if aid else None,
    }


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
async def athlete_detail(request: Request, athlete_id: str, tag: str = ""):
    a = athletes.get(athlete_id)
    if not a:
        return templates.TemplateResponse(request, "athletes.html",
                                          {"athletes_list": [],
                                           "error": "Athlète introuvable."})
    # Validation du filtre tag
    active_tag = tag.lower() if tag.lower() in VALID_TAGS else None
    a_jobs = _athlete_jobs(athlete_id, filter_tag=active_tag)
    stats  = _athlete_stats(a_jobs)
    counts = _athlete_tag_counts(athlete_id)
    return templates.TemplateResponse(request, "athlete_detail.html", {
        "athlete":      a,
        "athlete_jobs": a_jobs,
        "stats":        stats,
        "active_tag":   active_tag,
        "tag_counts":   counts,
        "tag_labels":   TAG_LABELS,
    })


@app.post("/jobs/{job_id}/tag")
async def jobs_set_tag(job_id: str, tag: str = Form("")):
    """Définit le tag d'un job. Chaîne vide / invalide = retire le tag."""
    j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    t = (tag or "").strip().lower()
    if t and t in VALID_TAGS:
        j["tag"] = t
    else:
        j.pop("tag", None)
    save_jobs(jobs)
    return {
        "ok":        True,
        "tag":       j.get("tag"),
        "tag_label": TAG_LABELS.get(j.get("tag") or "", ""),
    }


@app.post("/jobs/{job_id}/notes")
async def jobs_set_notes(job_id: str, notes: str = Form("")):
    """Définit / met à jour les notes libres d'un job. Vide = retire les notes."""
    j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    text = (notes or "").strip()
    if len(text) > 2000:
        return JSONResponse({"error": "Trop long (max 2000 caractères)."}, status_code=400)
    if text:
        j["notes"] = text
    else:
        j.pop("notes", None)
    save_jobs(jobs)
    return {"ok": True, "notes": j.get("notes", "")}


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


# ── Mesure manuelle du temps de réaction ─────────────────────────────────────
@app.get("/manual_reaction")
async def manual_reaction_page(request: Request):
    """Page de mesure manuelle : on liste les vidéos existantes (jobs + pros)
    et on laisse aussi l'option d'en uploader une nouvelle."""
    videos = []
    for jid, j in jobs.items():
        if j.get("status") != "done":
            continue
        r = j.get("results", {})
        ann = r.get("files", {}).get("annotated_video")
        if not ann:
            continue
        ath_id   = j.get("athlete_id")
        ath_name = athletes.get(ath_id, {}).get("name") if ath_id else None
        date     = j.get("added_at", "").split(" ")[0] if j.get("added_at") else ""
        label    = ath_name or r.get("video_name", jid)
        if date:
            label = f"{label} — {date}"
        videos.append({
            "label": label,
            "url":   f"/output/{ann}",
            "fps":   r.get("fps", 30),
        })
    for pid, p in pros.items():
        if p.get("status") != "done":
            continue
        vf = p.get("video_file")
        if not vf:
            continue
        videos.append({
            "label": (p.get("name") or pid) + " (pro)",
            "url":   f"/output/{vf}",
            "fps":   p.get("fps", 30),
        })
    videos.sort(key=lambda v: v["label"].lower())
    return templates.TemplateResponse(request, "manual_reaction.html",
                                      {"videos_list": videos})


@app.post("/manual_reaction/upload")
async def manual_reaction_upload(file: UploadFile = File(...)):
    """Sauve une vidéo pour mesure manuelle, sans déclencher l'analyse complète."""
    vid        = str(uuid.uuid4())[:8]
    safe_name  = file.filename.replace("/", "_")
    video_path = UPLOAD_DIR / f"manual_{vid}_{safe_name}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    info = get_video_info(video_path)
    return {
        "video_url": f"/uploads/manual_{vid}_{safe_name}",
        "fps":       info["fps"],
    }


# ── Paramètres (thème, etc.) ─────────────────────────────────────────────────
@app.get("/settings")
async def settings_page(request: Request):
    # Compteurs pour la carte "Nettoyage" (orphelins = sans athlète)
    n_orphan_jobs  = len(_orphan_job_ids())
    n_orphan_files = len(_orphan_files_on_disk())
    return templates.TemplateResponse(request, "settings.html", {
        "n_orphan_jobs":  n_orphan_jobs,
        "n_orphan_files": n_orphan_files,
        "backup":         _backup_status(),
    })


# ── Calibration du gate drop ─────────────────────────────────────────────────
def _get_or_run_audio_detection(jid: str, j: dict, force: bool = False) -> dict:
    """Cache la détection audio sur le job. Run une seule fois par job sauf force.
    Persiste dans jobs_db (snapshot auto)."""
    cached = j.get("audio_detection")
    if cached and not force:
        return cached
    vp_str = j.get("video_path")
    if not vp_str:
        return {"detected": False, "reason": "no video_path"}
    vp = Path(vp_str)
    if not vp.exists():
        return {"detected": False, "reason": "video file missing"}
    try:
        res = detect_gate_drop(vp)
    except Exception as e:
        res = {"detected": False, "reason": f"audio error: {e}"}
    # Stocke seulement les champs sérialisables utiles
    summary = {
        "detected":         bool(res.get("detected")),
        "gate_t":           res.get("gate_t"),
        "beeps_t":          res.get("beeps_t"),
        "mean_interval_ms": res.get("mean_interval_ms"),
        "confidence":       res.get("confidence"),
        "reason":           res.get("reason"),
    }
    j["audio_detection"] = summary
    save_jobs(jobs)
    return summary


@app.get("/gate_calibration")
async def gate_calibration_list(request: Request):
    """Liste des jobs avec leur statut de calibration et stats agrégées.
    Run audio detection pour chaque job (cache après 1er run)."""
    import statistics
    rows = []
    for jid, j in jobs.items():
        if j.get("status") != "done": continue
        r     = j.get("results", {})
        cal   = j.get("gate_calibration") or {}
        audio = _get_or_run_audio_detection(jid, j)
        rows.append({
            "job_id":         jid,
            "display_name":   _display_name(jid, j),
            "added_at":       j.get("added_at", ""),
            "fps":            r.get("fps"),
            "stored_gate_t":  r.get("gate_drop_t"),
            "audio_detected": audio.get("detected"),
            "audio_t":        audio.get("gate_t") if audio.get("detected") else None,
            "audio_reason":   audio.get("reason"),
            "audio_conf":     audio.get("confidence"),
            "calibrated":     bool(cal),
            "true_t":         cal.get("true_gate_t"),
            "diff_ms":        cal.get("diff_t_ms"),
            "diff_frames":    cal.get("diff_frames"),
            "calibrated_at":  cal.get("calibrated_at"),
        })
    rows.sort(key=lambda r: r["added_at"], reverse=True)
    # Stats globales sur les calibrés
    diffs_ms = [r["diff_ms"] for r in rows if r["calibrated"] and r["diff_ms"] is not None]
    stats = None
    if diffs_ms:
        stats = {
            "count":       len(diffs_ms),
            "median_ms":   round(statistics.median(diffs_ms), 1),
            "mean_ms":     round(sum(diffs_ms) / len(diffs_ms), 1),
            "stdev_ms":    round(statistics.stdev(diffs_ms) if len(diffs_ms) > 1 else 0.0, 1),
            "abs_mean_ms": round(sum(abs(d) for d in diffs_ms) / len(diffs_ms), 1),
            "min_ms":      round(min(diffs_ms), 1),
            "max_ms":      round(max(diffs_ms), 1),
        }
    return templates.TemplateResponse(request, "gate_calibration.html", {
        "rows": rows, "stats": stats,
    })


@app.get("/gate_calibration/{job_id}")
async def gate_calibration_detail(request: Request, job_id: str):
    """Vue détaillée : vidéo + marker audio + marker user pour 1 job."""
    j = get_job_or_recover(job_id)
    if not j or j.get("status") != "done":
        return templates.TemplateResponse(request, "index.html",
                                          {"error": "Job introuvable."})
    audio_info = _get_or_run_audio_detection(job_id, j)
    return templates.TemplateResponse(request, "gate_calibration_detail.html", {
        "job_id":        job_id,
        "display_name":  _display_name(job_id, j),
        "results":       j["results"],
        "video_url":     j.get("video_url"),
        "audio_info":    audio_info or {"detected": False},
        "calibration":   j.get("gate_calibration") or {},
    })


@app.post("/jobs/{job_id}/rerun_audio")
async def jobs_rerun_audio(job_id: str):
    """Force un re-run de la détection audio (vide le cache et recalcule)."""
    j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    j.pop("audio_detection", None)
    audio = _get_or_run_audio_detection(job_id, j, force=True)
    return {"ok": True, "audio": audio}


@app.post("/jobs/{job_id}/gate_calibration")
async def jobs_set_gate_calibration(job_id: str,
                                     true_gate_t: float = Form(...),
                                     audio_t: float    = Form(-1.0)):
    """Sauve la frame "vraie" marquée visuellement par l'user + le diff avec audio."""
    j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    fps = j.get("results", {}).get("fps") or 30.0
    diff_t = (audio_t - true_gate_t) if audio_t >= 0 else None
    j["gate_calibration"] = {
        "true_gate_t":     float(true_gate_t),
        "true_gate_frame": int(round(true_gate_t * fps)),
        "audio_t":         float(audio_t) if audio_t >= 0 else None,
        "audio_frame":     int(round(audio_t * fps)) if audio_t >= 0 else None,
        "diff_t_ms":       round(diff_t * 1000, 1) if diff_t is not None else None,
        "diff_frames":     int(round(diff_t * fps)) if diff_t is not None else None,
        "calibrated_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_jobs(jobs)
    return {"ok": True, "calibration": j["gate_calibration"]}


@app.delete("/jobs/{job_id}/gate_calibration")
async def jobs_delete_gate_calibration(job_id: str):
    j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    if "gate_calibration" in j:
        del j["gate_calibration"]
        save_jobs(jobs)
    return {"ok": True}


@app.post("/jobs/purge_orphans")
async def jobs_purge_orphans():
    """Supprime tous les jobs en mémoire sans athlète + les fichiers orphelins
    sur disque (préfixe job_id qui ne correspond à aucun job/pro connu)."""
    return purge_orphans()


@app.get("/settings/backup.zip")
async def settings_backup_zip():
    """ZIP contenant :
       - current/  → les 3 DBs JSON actuelles
       - history/  → tous les snapshots quotidiens (jusqu'à 14 jours)
    À télécharger régulièrement et garder dans iCloud / Drive / clé USB."""
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in (PROS_DB, JOBS_DB, ATHLETES_DB):
            if p.exists():
                zf.write(p, arcname=f"current/{p.name}")
        if BACKUP_DIR.exists():
            for f in BACKUP_DIR.rglob("*.json"):
                arc = "history/" + f.relative_to(BACKUP_DIR).as_posix()
                zf.write(f, arcname=arc)
    fname = f"bmx-backup-{datetime.now().strftime('%Y-%m-%d_%H%M')}.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.post("/settings/backup_snapshot")
async def settings_backup_snapshot():
    """Force un snapshot immédiat des 3 DBs (utile avant une opération risquée)."""
    try:
        for p in (PROS_DB, JOBS_DB, ATHLETES_DB):
            _snapshot_db(p)
        _prune_old_backups()
        return {"ok": True, "status": _backup_status()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Outils (calculateurs gear / pression) ────────────────────────────────────
@app.get("/tools/gear")
async def tools_gear(request: Request):
    return templates.TemplateResponse(request, "tools_gear.html", {})


@app.get("/tools/pressure")
async def tools_pressure(request: Request):
    return templates.TemplateResponse(request, "tools_pressure.html", {})


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
                     gate_frame: int = Form(...),
                     bip1_time: float = Form(-1.0)):
    """Démarre la génération du squelette du pro avec gate frame + (optionnel) bip1."""
    p = pros.get(pro_id)
    if not p:
        return JSONResponse({"error": "Pro introuvable."}, status_code=404)

    fps        = p["fps"]
    gate_drop  = gate_frame / fps
    video_path = Path(p["video_path"])
    if bip1_time > 0:
        bip1 = bip1_time
    else:
        bip1 = max(0.0, gate_drop - UCI_BIP1_TO_GATE_S)

    pros[pro_id]["status"]   = "queued"
    pros[pro_id]["progress"] = "En attente..."
    pros[pro_id]["bip1_time"] = bip1
    save_pros(pros)

    background_tasks.add_task(run_pro_analysis, pro_id, video_path, gate_drop, bip1)
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


def _normalized_skeleton_anchored(row, side: str, direction: int,
                                  hip0_x: float, hip0_y: float,
                                  scale0: float) -> dict | None:
    """Comme _normalized_skeleton mais ancré sur (hip0, scale0) fixés à un
    instant de référence (T = −0.5s par défaut). La hanche n'est PAS remise à
    (0,0) à chaque frame ; elle bouge avec le rider depuis sa position
    initiale, ce qui révèle le déplacement réel pendant la séquence."""
    if not (np.isfinite(hip0_x) and np.isfinite(hip0_y) and scale0 > 1.0):
        return None
    other = "R" if side == "L" else "L"

    def _n(col_x: str, col_y: str):
        x = row.get(col_x, np.nan)
        y = row.get(col_y, np.nan)
        if np.isnan(x) or np.isnan(y):
            return None
        nx = (x - hip0_x) / scale0
        ny = (y - hip0_y) / scale0
        if direction < 0:
            nx = -nx
        return [round(float(nx), 3), round(float(ny), 3)]

    return {
        "nose":          _n("nose_x",                "nose_y"),
        "shoulder":      _n(f"{side}_shoulder_x",    f"{side}_shoulder_y"),
        "elbow":         _n(f"{side}_elbow_x",       f"{side}_elbow_y"),
        "wrist":         _n(f"{side}_wrist_x",       f"{side}_wrist_y"),
        "hip":           _n(f"{side}_hip_x",         f"{side}_hip_y"),
        "knee":          _n(f"{side}_knee_x",        f"{side}_knee_y"),
        "ankle":         _n(f"{side}_ankle_x",       f"{side}_ankle_y"),
        "back_shoulder": _n(f"{other}_shoulder_x",   f"{other}_shoulder_y"),
        "back_elbow":    _n(f"{other}_elbow_x",      f"{other}_elbow_y"),
        "back_wrist":    _n(f"{other}_wrist_x",      f"{other}_wrist_y"),
        "back_hip":      _n(f"{other}_hip_x",        f"{other}_hip_y"),
    }


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
        "back_elbow":    _n(f"{other}_elbow_x",      f"{other}_elbow_y"),
        "back_wrist":    _n(f"{other}_wrist_x",      f"{other}_wrist_y"),
        "back_hip":      _n(f"{other}_hip_x",        f"{other}_hip_y"),
    }


def _compute_angles_at_time(csv_path: Path, t: float, side: str,
                            gate_t: float | None = None,
                            anchor_T: float = -0.5) -> dict | None:
    """Lit les keypoints au frame le plus proche de l'instant `t` dans le CSV
    landmarks et calcule 6 angles : knee, hip, ankle, shoulder, elbow, trunk.

    `side` ∈ {'L','R'} = côté du rider (pied avant). Le tronc est signé selon
    la direction du rider (positif = penché en avant). Le squelette normalisé
    pour l'overlay est inclus dans la réponse.

    Si `gate_t` est fourni, on calcule aussi `skeleton_free` ancré à
    `gate_t + anchor_T` (par défaut T=−0.5s) — utile pour la photo statique
    avec le toggle "Mouvement libre depuis T=−0.5s"."""
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

    # Mode "Mouvement libre depuis T=−0.5s" : si on connaît le gate_t, on
    # trouve la frame à gate_t + anchor_T et on en tire hip0, scale0 fixes.
    skeleton_free = None
    if gate_t is not None:
        anchor_t_abs = gate_t + anchor_T
        anchor_idx   = int((df["time"] - anchor_t_abs).abs().idxmin())
        anchor_row   = df.loc[anchor_idx]
        hip0_x = float(anchor_row.get(f"{side}_hip_x",      np.nan))
        hip0_y = float(anchor_row.get(f"{side}_hip_y",      np.nan))
        sh0_x  = float(anchor_row.get(f"{side}_shoulder_x", np.nan))
        sh0_y  = float(anchor_row.get(f"{side}_shoulder_y", np.nan))
        scale0 = float(np.hypot(sh0_x - hip0_x, sh0_y - hip0_y)) \
                 if np.isfinite(hip0_x) and np.isfinite(sh0_x) else 0.0
        skeleton_free = _normalized_skeleton_anchored(
            row, side, direction, hip0_x, hip0_y, scale0)

    return {
        "frame":         int(idx),
        "t":             round(float(row["time"]), 3),
        "side":          side,
        "knee":          knee_a,
        "hip":           hip_a,
        "ankle":         ankle_a,
        "shoulder":      shoulder_a,
        "elbow":         elbow_a,
        "trunk":         trunk_a,
        "skeleton":      _normalized_skeleton(row, side, direction),
        "skeleton_free": skeleton_free,
    }


def _resolve_reference(ref_type: str, ref_id: str) -> dict | None:
    """Résout une référence (pro ou autre job) en un dict commun :
    {csv_path, side, gate_t, name}. Retourne None si introuvable / invalide."""
    if ref_type == "pro":
        pro = pros.get(ref_id)
        if not pro or pro.get("status") != "done":
            return None
        csv_name = pro.get("landmarks_csv") \
                   or pro["video_file"].replace("_annotated.mp4", "_landmarks.csv")
        csv_path = OUTPUT_DIR / csv_name
        side = pro.get("front_foot")
        if not side and csv_path.exists():
            side = _infer_front_foot_from_csv(pd.read_csv(csv_path))
        return {
            "csv_path": csv_path,
            "side":     side or "L",
            "gate_t":   float(pro.get("gate_drop_t", 0.0)),
            "name":     pro.get("name", "Pro"),
        }
    if ref_type == "job":
        j = jobs.get(ref_id)
        if not j or j.get("status") != "done":
            return None
        r = j.get("results", {})
        csv_name = r.get("files", {}).get("landmarks_csv")
        if not csv_name:
            return None
        aid = j.get("athlete_id")
        ath = athletes.get(aid, {}).get("name") if aid else None
        date = j.get("added_at", "").split(" ")[0] if j.get("added_at") else ""
        name = f"{ath} — {date}" if ath else (r.get("video_name") or date or "Référence")
        return {
            "csv_path": OUTPUT_DIR / csv_name,
            "side":     r.get("front_foot") or "L",
            "gate_t":   float(r.get("gate_drop_t", 0.0)),
            "name":     name,
        }
    return None


@app.post("/compare_angles/{job_id}")
async def compare_angles(job_id:    str,
                         rider_t:   float = Form(...),
                         pro_t:     float = Form(...),
                         pro_id:    str   = Form(""),
                         ref_type:  str   = Form(""),
                         ref_id:    str   = Form("")):
    """Calcule les 6 angles articulaires (rider vs référence) aux instants
    spécifiés. Référence = pro de la banque OU autre job déjà analysé.
    Compatibilité : `pro_id` seul est encore accepté (= ref_type=pro)."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    if not ref_type and pro_id:
        ref_type, ref_id = "pro", pro_id
    ref = _resolve_reference(ref_type, ref_id)
    if not ref:
        return JSONResponse({"error": "Référence introuvable ou non analysée."},
                            status_code=404)

    rider_results = job["results"]
    rider_csv     = OUTPUT_DIR / rider_results["files"]["landmarks_csv"]
    rider_side    = rider_results.get("front_foot") or "L"
    rider_gate_t  = float(rider_results.get("gate_drop_t", 0.0))

    rider_angles = _compute_angles_at_time(rider_csv, rider_t, rider_side,
                                           gate_t=rider_gate_t)
    ref_angles   = _compute_angles_at_time(ref["csv_path"], pro_t, ref["side"],
                                           gate_t=ref["gate_t"])

    if rider_angles is None or ref_angles is None:
        return JSONResponse(
            {"error": f"Lecture CSV impossible (rider={rider_csv.exists()}, ref={ref['csv_path'].exists()})"},
            status_code=500,
        )

    return {
        "rider":    rider_angles,
        "pro":      ref_angles,    # clé conservée pour compat front
        "pro_name": ref["name"],
    }


# ── Séquence animée des angles (rider vs pro sur une fenêtre de temps) ───────
def _compute_sequence(csv_path: Path, gate_t: float, side: str,
                      t_start: float, t_end: float,
                      anchor_T: float = -0.5) -> list[dict]:
    """Retourne la liste {T, frame, t, angles, skeleton, skeleton_free} pour
    toutes les frames du CSV dont le temps est dans [gate_t + t_start,
    gate_t + t_end]. T = temps relatif au gate drop (T=0 = chute des grilles).

    `anchor_T` = instant de référence pour le mode "mouvement libre" : on fige
    la position des hanches et l'échelle hanche→épaule à cet instant (par
    défaut T = −0.5s, juste avant la chute des grilles → position statique
    parfaite comme référence). Les squelettes des deux ridens s'alignent
    exactement à anchor_T puis divergent en suivant leur mouvement réel."""
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if "time" not in df.columns or len(df) == 0:
        return []
    t_lo = gate_t + t_start
    t_hi = gate_t + t_end
    sub  = df[(df["time"] >= t_lo) & (df["time"] <= t_hi)]
    if sub.empty:
        return []

    # Direction (face droite/gauche) : on prend la médiane sur la fenêtre pour
    # éviter les flips ponctuels dus au bruit de pose detection.
    nose_x = sub.get("nose_x")
    L_hi_x = sub.get("L_hip_x")
    R_hi_x = sub.get("R_hip_x")
    direction = 1
    if nose_x is not None and L_hi_x is not None and R_hi_x is not None:
        center_hip = (L_hi_x + R_hi_x) / 2.0
        diff = (nose_x - center_hip).dropna()
        if len(diff) > 0:
            direction = 1 if float(diff.median()) > 0 else -1

    # Ancrage T = anchor_T : on cherche dans le CSV complet (pas seulement la
    # fenêtre) la frame la plus proche pour récupérer hip0 et scale0.
    anchor_t_abs = gate_t + anchor_T
    anchor_idx   = int((df["time"] - anchor_t_abs).abs().idxmin())
    anchor_row   = df.loc[anchor_idx]
    hip0_x  = float(anchor_row.get(f"{side}_hip_x",      np.nan))
    hip0_y  = float(anchor_row.get(f"{side}_hip_y",      np.nan))
    sh0_x   = float(anchor_row.get(f"{side}_shoulder_x", np.nan))
    sh0_y   = float(anchor_row.get(f"{side}_shoulder_y", np.nan))
    scale0  = float(np.hypot(sh0_x - hip0_x, sh0_y - hip0_y)) \
              if np.isfinite(hip0_x) and np.isfinite(sh0_x) else 0.0

    out: list[dict] = []
    for idx, row in sub.iterrows():
        t = float(row["time"])
        T = round(t - gate_t, 3)

        def _pt(part: str):
            x = row.get(f"{side}_{part}_x", np.nan)
            y = row.get(f"{side}_{part}_y", np.nan)
            return (float(x), float(y))

        sh = _pt("shoulder"); el = _pt("elbow"); wr = _pt("wrist")
        hi = _pt("hip");      kn = _pt("knee");  an = _pt("ankle")

        def _safe_angle(p1, p2, p3):
            if any(np.isnan(p[0]) or np.isnan(p[1]) for p in (p1, p2, p3)):
                return None
            a = float(calculate_angle(p1, p2, p3))
            return round(a, 1) if not np.isnan(a) else None

        knee_a     = _safe_angle(hi, kn, an)
        hip_a      = _safe_angle(sh, hi, kn)
        elbow_a    = _safe_angle(sh, el, wr)
        shoulder_a = _safe_angle(hi, sh, el)
        ankle_a = None
        if not (np.isnan(an[0]) or np.isnan(kn[0])):
            toe_proxy = (an[0] + 30 * direction, an[1])
            ankle_a   = _safe_angle(kn, an, toe_proxy)
        trunk_a = None
        if not (np.isnan(sh[0]) or np.isnan(hi[0])):
            dx = sh[0] - hi[0]
            dy = sh[1] - hi[1]
            trunk_a = round(float(np.degrees(np.arctan2(direction * dx, -dy))), 1)

        out.append({
            "T":             T,
            "t":             round(t, 3),
            "frame":         int(idx),
            "knee":          knee_a,
            "hip":           hip_a,
            "ankle":         ankle_a,
            "shoulder":      shoulder_a,
            "elbow":         elbow_a,
            "trunk":         trunk_a,
            "skeleton":      _normalized_skeleton(row, side, direction),
            "skeleton_free": _normalized_skeleton_anchored(
                                  row, side, direction, hip0_x, hip0_y, scale0),
        })
    return out


@app.post("/compare_angles_sequence/{job_id}")
async def compare_angles_sequence(job_id: str,
                                  pro_id:        str   = Form(""),
                                  ref_type:      str   = Form(""),
                                  ref_id:        str   = Form(""),
                                  rider_gate_t:  float = Form(-1.0),
                                  ref_gate_t:    float = Form(-1.0),
                                  t_start:       float = Form(-0.5),
                                  t_end:         float = Form(2.5)):
    """Séquence complète squelettes + angles, rider vs référence (pro ou autre
    job). Fenêtre [gate_t + t_start, gate_t + t_end] alignée sur le gate drop
    de chaque vidéo (T=0 = chute des grilles)."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    if not ref_type and pro_id:
        ref_type, ref_id = "pro", pro_id
    ref = _resolve_reference(ref_type, ref_id)
    if not ref:
        return JSONResponse({"error": "Référence introuvable ou non analysée."},
                            status_code=404)

    rider_results = job["results"]
    rider_csv     = OUTPUT_DIR / rider_results["files"]["landmarks_csv"]
    rider_side    = rider_results.get("front_foot") or "L"

    # Si le client a passé un gate calibré manuellement (via les flèches de
    # calage solo), on l'utilise comme T=0 plutôt que le gate stocké au
    # moment de l'analyse. -1 = pas de surcharge, on garde le gate d'origine.
    rider_t0 = rider_gate_t if rider_gate_t >= 0 \
               else float(rider_results.get("gate_drop_t", 0.0))
    ref_t0   = ref_gate_t   if ref_gate_t   >= 0 else ref["gate_t"]

    rider_seq = _compute_sequence(rider_csv, rider_t0, rider_side,
                                  t_start, t_end)
    ref_seq   = _compute_sequence(ref["csv_path"], ref_t0, ref["side"],
                                  t_start, t_end)

    if not rider_seq or not ref_seq:
        return JSONResponse(
            {"error": f"Lecture CSV impossible (rider={rider_csv.exists()}, ref={ref['csv_path'].exists()})"},
            status_code=500,
        )

    def _est_fps(seq):
        if len(seq) < 2: return 30.0
        dt = seq[-1]["t"] - seq[0]["t"]
        return round((len(seq) - 1) / dt, 2) if dt > 0 else 30.0

    return {
        "rider":      rider_seq,
        "pro":        ref_seq,    # clé conservée pour compat front
        "rider_fps":  _est_fps(rider_seq),
        "pro_fps":    _est_fps(ref_seq),
        "t_start":    t_start,
        "t_end":      t_end,
        "pro_name":   ref["name"],
    }


# ── Annotations dessinées sur la page Compare ─────────────────────────────────
MAX_ANNOTATIONS_PER_JOB = 500


@app.get("/compare/{job_id}/annotations")
async def get_annotations(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    return {"annotations": job.get("annotations", [])}


@app.get("/pros/{pro_id}/annotations")
async def get_pro_annotations(pro_id: str):
    pro = pros.get(pro_id)
    if not pro:
        return JSONResponse({"error": "Pro introuvable."}, status_code=404)
    return {"annotations": pro.get("annotations", [])}


@app.post("/pros/{pro_id}/annotations")
async def save_pro_annotations(pro_id: str, request: Request):
    pro = pros.get(pro_id)
    if not pro:
        return JSONResponse({"error": "Pro introuvable."}, status_code=404)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide."}, status_code=400)
    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        return JSONResponse({"error": "annotations doit être une liste."}, status_code=400)
    if len(annotations) > MAX_ANNOTATIONS_PER_JOB:
        return JSONResponse(
            {"error": f"Trop d'annotations (max {MAX_ANNOTATIONS_PER_JOB})."},
            status_code=400,
        )
    pro["annotations"] = annotations
    save_pros(pros)
    return {"ok": True, "count": len(annotations)}


@app.post("/compare/{job_id}/annotations")
async def save_annotations(job_id: str, request: Request):
    """Sauve la liste complète des annotations pour ce job (remplace l'ancienne)."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON invalide."}, status_code=400)
    annotations = data.get("annotations", [])
    if not isinstance(annotations, list):
        return JSONResponse({"error": "annotations doit être une liste."}, status_code=400)
    if len(annotations) > MAX_ANNOTATIONS_PER_JOB:
        return JSONResponse(
            {"error": f"Trop d'annotations (max {MAX_ANNOTATIONS_PER_JOB})."},
            status_code=400,
        )
    job["annotations"] = annotations
    save_jobs(jobs)
    return {"ok": True, "count": len(annotations)}


# ── Comparaison ───────────────────────────────────────────────────────────────
@app.get("/compare/{job_id}")
async def compare(request: Request, job_id: str):
    job = get_job_or_recover(job_id)
    if not job or job["status"] != "done":
        return templates.TemplateResponse(request, "index.html",
                                          {"error": "Job introuvable."})
    pros_done = [p for p in pros.values() if p.get("status") == "done"]

    # Sélecteur élargi : jobs du même athlète + jobs d'autres athlètes (ou sans athlète),
    # tous filtrés sur status=done et ayant un landmarks_csv exploitable.
    cur_athlete_id = job.get("athlete_id")
    same_athlete_jobs: list[dict] = []
    other_jobs: list[dict] = []
    for jid, j in jobs.items():
        if jid == job_id:
            continue
        if j.get("status") != "done":
            continue
        r = j.get("results", {})
        if not r.get("files", {}).get("landmarks_csv"):
            continue
        aid = j.get("athlete_id")
        athlete_name = athletes.get(aid, {}).get("name") if aid else None
        entry = _job_compare_entry(jid, j, athlete_name)
        if cur_athlete_id and aid == cur_athlete_id:
            same_athlete_jobs.append(entry)
        else:
            other_jobs.append(entry)
    same_athlete_jobs.sort(key=lambda x: x.get("added_at", ""), reverse=True)
    other_jobs.sort(key=lambda x: x.get("added_at", ""), reverse=True)

    return templates.TemplateResponse(request, "compare.html", {
        "job_id":             job_id,
        "rider":              job["results"],
        "rider_display_name": _display_name(job_id, job),
        "pros_list":          pros_done,
        "same_athlete_jobs":  same_athlete_jobs,
        "other_jobs":         other_jobs,
        "current_athlete":    athletes.get(cur_athlete_id, {}).get("name") if cur_athlete_id else None,
    })

