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
import shlex
import subprocess
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
from scipy.signal import savgol_filter, find_peaks

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
TRACKS_DB     = OUTPUT_DIR / "tracks_db.json"
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

def load_tracks() -> dict:
    if TRACKS_DB.exists():
        try:
            return json.loads(TRACKS_DB.read_text())
        except Exception:
            pass
    return {}

def save_tracks(tracks: dict):
    _save_db_with_backup(TRACKS_DB, json.dumps(tracks, indent=2, ensure_ascii=False))

pros:     dict = load_pros()
jobs:     dict = load_jobs()
athletes: dict = load_athletes()
tracks:   dict = load_tracks()


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


def _athlete_jobs(athlete_id: str,
                  filter_tag: str | None = None,
                  filter_track: str | None = None) -> list[dict]:
    """Retourne tous les jobs (terminés) liés à cet athlète, triés du plus récent
    au plus ancien. Filtres optionnels par tag et / ou par track_id."""
    out = []
    for jid, j in jobs.items():
        if j.get("status") != "done":
            continue
        if j.get("athlete_id") != athlete_id:
            continue
        if filter_tag is not None and j.get("tag") != filter_tag:
            continue
        if filter_track is not None and j.get("track_id") != filter_track:
            continue
        track_id = j.get("track_id")
        track_name = tracks.get(track_id, {}).get("name") if track_id else None
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
            "track_id":     track_id,
            "track_name":   track_name,
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


def _athlete_track_counts(athlete_id: str) -> dict:
    """Compteur de jobs par track pour cet athlète. Retourne dict track_id → count."""
    counts: dict = {}
    untracked = 0
    for jid, j in jobs.items():
        if j.get("status") != "done": continue
        if j.get("athlete_id") != athlete_id: continue
        tid = j.get("track_id")
        if tid and tid in tracks:
            counts[tid] = counts.get(tid, 0) + 1
        else:
            untracked += 1
    return {"by_track": counts, "untracked": untracked}


def _track_usage_counts() -> dict:
    """Combien de jobs utilisent chaque track (pour /settings)."""
    out: dict = {tid: 0 for tid in tracks.keys()}
    for j in jobs.values():
        if j.get("status") != "done": continue
        tid = j.get("track_id")
        if tid and tid in out:
            out[tid] += 1
    return out


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
        # Génération automatique du preview rapide en post-analyse
        _start_pro_preview_generation(pro_id)
    except Exception as e:
        pros[pro_id].update({"status": "error", "error": str(e)})
        save_pros(pros)


# ── Génération de previews H.264 720p pour les vidéos pro ────────────────────
# Le pipeline d'analyse génère la vidéo annotée avec OpenCV en MPEG-4 part 2
# brut (codec lent à décoder, débits aberrants). Sur du 4K 60fps comme Eddy
# Clerté, ça donne un fichier de 100+ MB qui rame sur iPad (chargement +
# scrubbing). On encode donc un preview compact en H.264 baseline 720p 30fps
# avec faststart, qu'on sert à la place pour l'affichage. Le fichier original
# annoté reste sur disque (intact) pour conserver la qualité d'analyse.

# Version du format de preview — change quand on touche aux options ffmpeg pour
# invalider les anciens previews.
PREVIEW_FORMAT_VERSION = "1.0"


def _preview_path_for(video_file: str) -> Path:
    """Retourne le chemin du preview attendu pour une vidéo annotée donnée.
    Ex: pro_70bdaf36_EDDIE_annotated.mp4 → pro_70bdaf36_EDDIE_annotated_preview.mp4
    """
    stem = Path(video_file).stem
    return OUTPUT_DIR / f"{stem}_preview.mp4"


def _make_pro_preview_sync(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """Encode `input_path` en H.264 720p baseline avec faststart. Synchrone.
    Retourne (success, message_d_erreur_ou_log_court)."""
    if not input_path.exists():
        return False, f"Source absente : {input_path}"
    # Box 1280×1280 avec aspect ratio préservé : la plus grande dimension passe
    # à 1280, l'autre est mise à l'échelle proportionnellement. Force la
    # dimension finale paire (-2) car H.264 exige des dimensions paires.
    vf = ("scale='if(gt(iw,ih),min(1280,iw),-2)':"
          "'if(gt(iw,ih),-2,min(1280,ih))':"
          "force_original_aspect_ratio=decrease,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-vf", vf,
        "-r", "30",                                  # cap 30 fps
        "-c:v", "libx264",
        "-profile:v", "baseline", "-level", "3.1",   # compat iPad max
        "-preset", "veryfast",
        "-crf", "24",
        "-g", "30", "-keyint_min", "30",             # keyframe chaque 1s → scrub fluide
        "-pix_fmt", "yuv420p",                       # compat Safari mobile
        "-an",                                       # pas d'audio dans le preview
        "-movflags", "+faststart",                   # démarrage lecture immédiat
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, (result.stderr or result.stdout or "ffmpeg failed")[-500:]
        if not output_path.exists() or output_path.stat().st_size == 0:
            return False, "Le fichier preview n'a pas été créé."
        return True, f"OK · {output_path.stat().st_size // 1024} KB"
    except FileNotFoundError:
        return False, "ffmpeg introuvable dans le PATH"
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timeout (>5 min)"
    except Exception as e:
        return False, f"Exception : {e}"


def _start_pro_preview_generation(pro_id: str):
    """Spawn la génération du preview en thread daemon — non-bloquant pour
    l'event loop FastAPI. Met à jour pros_db.json au fur et à mesure."""
    import threading
    p = pros.get(pro_id)
    if not p or p.get("status") != "done" or not p.get("video_file"):
        return
    # Évite les générations concurrentes
    if p.get("preview_status") == "generating":
        return
    pros[pro_id]["preview_status"]  = "generating"
    pros[pro_id]["preview_error"]   = None
    pros[pro_id].pop("preview_file",    None)
    save_pros(pros)

    def _worker():
        try:
            video_file  = p["video_file"]
            input_path  = OUTPUT_DIR / video_file
            output_path = _preview_path_for(video_file)
            ok, msg = _make_pro_preview_sync(input_path, output_path)
            if ok:
                pros[pro_id]["preview_status"]   = "ready"
                pros[pro_id]["preview_file"]     = output_path.name
                pros[pro_id]["preview_version"]  = PREVIEW_FORMAT_VERSION
                pros[pro_id]["preview_size_kb"]  = output_path.stat().st_size // 1024
                pros[pro_id]["preview_error"]    = None
            else:
                pros[pro_id]["preview_status"] = "error"
                pros[pro_id]["preview_error"]  = msg
            save_pros(pros)
        except Exception as e:
            pros[pro_id]["preview_status"] = "error"
            pros[pro_id]["preview_error"]  = f"Worker exception : {e}"
            save_pros(pros)

    threading.Thread(target=_worker, daemon=True,
                     name=f"preview-{pro_id}").start()


def _needs_preview(pro: dict) -> bool:
    """True si ce pro n'a pas de preview à jour."""
    if pro.get("status") != "done" or not pro.get("video_file"):
        return False
    if pro.get("preview_status") == "generating":
        return False
    if pro.get("preview_status") != "ready":
        return True
    if pro.get("preview_version") != PREVIEW_FORMAT_VERSION:
        return True
    # Vérifie que le fichier existe encore sur disque
    preview_file = pro.get("preview_file")
    if not preview_file or not (OUTPUT_DIR / preview_file).exists():
        return True
    return False


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
    current_track_id   = job.get("track_id")
    current_track_name = tracks.get(current_track_id, {}).get("name") if current_track_id else ""
    tracks_options = sorted(
        [{"id": t["id"], "name": t["name"]} for t in tracks.values()],
        key=lambda x: x["name"].lower(),
    )
    return templates.TemplateResponse(request, "result.html", {
        "job_id":              job_id,
        "results":             job["results"],
        "athlete":             athlete,
        "athletes_options":    athletes_options,
        "display_name":        _display_name(job_id, job),
        "custom_name":         job.get("custom_name", ""),
        "auto_name":           _display_name(job_id, {**job, "custom_name": ""}),
        "current_tag":         job.get("tag", ""),
        "current_notes":       job.get("notes", ""),
        "tag_options":         [{"id": k, "label": v} for k, v in TAG_LABELS.items()],
        "current_track_id":    current_track_id or "",
        "current_track_name":  current_track_name,
        "tracks_options":      tracks_options,
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
async def athlete_detail(request: Request, athlete_id: str,
                          tag: str = "", track: str = ""):
    a = athletes.get(athlete_id)
    if not a:
        return templates.TemplateResponse(request, "athletes.html",
                                          {"athletes_list": [],
                                           "error": "Athlète introuvable."})
    active_tag   = tag.lower() if tag.lower() in VALID_TAGS else None
    active_track = track if track in tracks else None
    a_jobs       = _athlete_jobs(athlete_id, filter_tag=active_tag, filter_track=active_track)
    stats        = _athlete_stats(a_jobs)
    counts       = _athlete_tag_counts(athlete_id)
    track_cnt    = _athlete_track_counts(athlete_id)
    # Tracks utilisés par cet athlète, triés par usage décroissant
    used_tracks = []
    for tid, n in sorted(track_cnt["by_track"].items(), key=lambda x: -x[1]):
        if tid in tracks:
            used_tracks.append({"id": tid, "name": tracks[tid]["name"], "count": n})
    return templates.TemplateResponse(request, "athlete_detail.html", {
        "athlete":         a,
        "athlete_jobs":    a_jobs,
        "stats":           stats,
        "active_tag":      active_tag,
        "active_track":    active_track,
        "tag_counts":      counts,
        "tag_labels":      TAG_LABELS,
        "used_tracks":     used_tracks,
        "untracked_count": track_cnt["untracked"],
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


# ── Pistes (tracks) — CRUD ─────────────────────────────────────────
@app.get("/tracks")
async def tracks_list():
    """Liste JSON des pistes triées alphabétiquement (pour les pickers)."""
    items = sorted(
        [{"id": t["id"], "name": t["name"]} for t in tracks.values()],
        key=lambda x: x["name"].lower(),
    )
    return {"tracks": items}


@app.post("/tracks")
async def tracks_create(name: str = Form(...)):
    """Crée une piste. Si une piste existe déjà avec ce nom (case-insensitive),
    retourne celle-là plutôt que de dupliquer."""
    name = (name or "").strip()
    if not name:
        return JSONResponse({"error": "Nom requis."}, status_code=400)
    if len(name) > 60:
        return JSONResponse({"error": "Trop long (max 60 caractères)."}, status_code=400)
    # Anti-doublon insensible à la casse
    for t in tracks.values():
        if t["name"].lower() == name.lower():
            return {"id": t["id"], "name": t["name"], "already_existed": True}
    tid = str(uuid.uuid4())[:8]
    tracks[tid] = {
        "id":         tid,
        "name":       name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_tracks(tracks)
    return {"id": tid, "name": name, "already_existed": False}


@app.patch("/tracks/{track_id}")
async def tracks_rename(track_id: str, name: str = Form(...)):
    t = tracks.get(track_id)
    if not t:
        return JSONResponse({"error": "Piste introuvable."}, status_code=404)
    new_name = (name or "").strip()
    if not new_name:
        return JSONResponse({"error": "Nom requis."}, status_code=400)
    if len(new_name) > 60:
        return JSONResponse({"error": "Trop long (max 60 caractères)."}, status_code=400)
    # Anti-doublon (en ignorant soi-même)
    for other_id, other in tracks.items():
        if other_id != track_id and other["name"].lower() == new_name.lower():
            return JSONResponse({"error": "Une autre piste a déjà ce nom."}, status_code=400)
    t["name"] = new_name
    save_tracks(tracks)
    return {"ok": True, "id": track_id, "name": new_name}


@app.delete("/tracks/{track_id}")
async def tracks_delete(track_id: str):
    """Supprime une piste. Les jobs qui l'utilisaient perdent leur référence
    (track_id mis à None) — la suppression est non-destructive côté analyses."""
    if track_id not in tracks:
        return JSONResponse({"error": "Piste introuvable."}, status_code=404)
    del tracks[track_id]
    save_tracks(tracks)
    dirty = False
    for j in jobs.values():
        if j.get("track_id") == track_id:
            j.pop("track_id", None)
            dirty = True
    if dirty:
        save_jobs(jobs)
    return {"ok": True}


@app.post("/jobs/{job_id}/track")
async def jobs_set_track(job_id: str, track_id: str = Form("")):
    """Assigne (ou retire si vide) une piste sur un job."""
    j = jobs.get(job_id)
    if not j:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    tid = (track_id or "").strip()
    if tid and tid not in tracks:
        return JSONResponse({"error": "Piste inconnue."}, status_code=400)
    if tid:
        j["track_id"] = tid
    else:
        j.pop("track_id", None)
    save_jobs(jobs)
    track_name = tracks.get(tid, {}).get("name") if tid else None
    return {"ok": True, "track_id": j.get("track_id"), "track_name": track_name}


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
        # Preview H.264 720p si dispo, sinon vidéo annotée originale
        display_file = p.get("preview_file") or vf
        videos.append({
            "label": (p.get("name") or pid) + " (pro)",
            "url":   f"/output/{display_file}",
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
    # Liste des pistes triée alpha + usage count
    usage = _track_usage_counts()
    tracks_list = sorted(
        [{"id": t["id"], "name": t["name"],
          "created_at": t.get("created_at", ""), "n_jobs": usage.get(t["id"], 0)}
         for t in tracks.values()],
        key=lambda x: x["name"].lower(),
    )
    return templates.TemplateResponse(request, "settings.html", {
        "n_orphan_jobs":  n_orphan_jobs,
        "n_orphan_files": n_orphan_files,
        "backup":         _backup_status(),
        "tracks_list":    tracks_list,
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
        for p in (PROS_DB, JOBS_DB, ATHLETES_DB, TRACKS_DB):
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
    """Force un snapshot immédiat des DBs (utile avant une opération risquée)."""
    try:
        for p in (PROS_DB, JOBS_DB, ATHLETES_DB, TRACKS_DB):
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


@app.post("/pros/{pro_id}/generate_preview")
async def pros_generate_preview(pro_id: str):
    """Lance la génération du preview H.264 720p en background. Retourne
    immédiatement avec le status courant."""
    p = pros.get(pro_id)
    if not p:
        return JSONResponse({"error": "Pro introuvable."}, status_code=404)
    if p.get("status") != "done" or not p.get("video_file"):
        return JSONResponse({"error": "Pro non analysé."}, status_code=400)
    _start_pro_preview_generation(pro_id)
    return {"ok": True, "pro_id": pro_id,
            "preview_status": pros[pro_id].get("preview_status")}


@app.post("/pros/regenerate_all_previews")
async def pros_regenerate_all_previews():
    """Lance la génération de tous les previews manquants ou obsolètes."""
    launched = []
    skipped  = []
    for pid, p in pros.items():
        if _needs_preview(p):
            _start_pro_preview_generation(pid)
            launched.append({"id": pid, "name": p.get("name", "")})
        else:
            skipped.append({"id": pid, "name": p.get("name", ""),
                            "preview_status": p.get("preview_status")})
    return {"launched": launched, "skipped": skipped,
            "n_launched": len(launched), "n_skipped": len(skipped)}


@app.get("/pros/preview_status")
async def pros_preview_status():
    """État de génération des previews pour TOUS les pros — sert au polling
    léger côté UI (bouton "Régénérer" + badges par pro)."""
    return {
        "pros": [
            {
                "id":             p.get("id", pid),
                "name":           p.get("name", ""),
                "preview_status": p.get("preview_status") or "missing",
                "preview_file":   p.get("preview_file"),
                "preview_error":  p.get("preview_error"),
                "preview_size_kb": p.get("preview_size_kb"),
                "needs_preview":  _needs_preview(p),
            }
            for pid, p in pros.items()
        ]
    }


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


# ── Module Explosivité & séquence proximale-distale ──────────────────────────
# Calcule la vitesse angulaire (ω = dθ/dt) pour hanche, genou, cheville sur la
# fenêtre [gate_drop, gate_drop + window_s] et l'ordre temporel des pics.
#
# Choix méthodologique (cf. ROADMAP §6 P1+P3) :
#   - Métrique intra-rider uniquement (pas de comparaison à un pro) → la
#     dérivée temporelle annule les biais constants liés à l'angle de caméra,
#     aux proportions corporelles et au cadrage, contrairement aux angles
#     absolus du module /compare_angles.
#   - Fenêtre [gate−0.3s, gate+1.0s] : large pour capter l'anticipation
#     élite (pic d'extension qui démarre AVANT gate drop) ET les pics
#     tardifs amateurs (>500ms). Permet aussi de détecter via find_peaks
#     les vrais maxima locaux plutôt que des artefacts de bord.
#   - Lissage Savitzky-Golay avant dérivation pour réduire l'amplification
#     du bruit YOLO-Pose par la différentiation.
#   - Détection de pic par scipy.signal.find_peaks (maxima locaux) ; si
#     aucun maximum local trouvé → fallback argmax + flag edge_peak qui
#     remonte à l'UI comme "Mesure incomplète, refaire avec vidéo plus
#     longue".
#   - Pas de cheville dans le Coordinating Index si fps < 60 : le proxy
#     pointe-de-pied (+30 px) combiné à la faible amplitude rend le pic
#     d'ankle peu fiable à basse cadence vidéo.
def _compute_kinematic_burst(csv_path: Path, gate_t: float, side: str,
                             window_pre: float = 0.3,
                             window_post: float = 1.0) -> dict | None:
    """Retourne ω_max (°/s) et t_peak (s, relatif au gate drop) pour hanche,
    genou, cheville côté pied avant + index de coordination proximale-distale.

    Convention : on cherche le pic de vitesse d'EXTENSION (angle qui
    augmente), donc ω positive. Si une articulation reste en flexion sur
    toute la fenêtre, on retourne None pour cette articulation.

    Retour : dict avec
      - fps_est       : float       (cadence vidéo estimée sur la fenêtre)
      - window_pre    : float       (durée pré-gate analysée, en s)
      - window_post   : float       (durée post-gate analysée, en s)
      - n_samples     : int         (nombre de frames analysées)
      - hip/knee/ankle: {omega_max, t_peak, series, edge_peak} ou None
        · edge_peak=True signifie qu'aucun maximum local n'a été trouvé →
          le pic remonté est en bord de fenêtre, donc potentiellement
          incomplet (mesure à refaire avec une vidéo plus longue).
      - ci_score      : int  -1/0/1/2  (voir _coordinating_index)
      - ci_verdict    : str  "proximal_distal" | "simultaneous" | "inverted" | "partial"
      - ci_reason     : str  texte humain décrivant la séquence
      - has_edge_warning: bool  True si au moins une articulation a edge_peak
    Retourne None si lecture CSV impossible ou fenêtre vide.
    """
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "time" not in df.columns or len(df) == 0:
        return None
    t_lo = gate_t - window_pre
    t_hi = gate_t + window_post
    sub  = df[(df["time"] >= t_lo) & (df["time"] <= t_hi)].reset_index(drop=True)
    if len(sub) < 5:
        return None

    # Direction (face droite/gauche) — médiane sur la fenêtre pour stabilité
    nose_x = sub.get("nose_x")
    L_hi_x = sub.get("L_hip_x")
    R_hi_x = sub.get("R_hip_x")
    direction = 1
    if nose_x is not None and L_hi_x is not None and R_hi_x is not None:
        center_hip = (L_hi_x + R_hi_x) / 2.0
        diff = (nose_x - center_hip).dropna()
        if len(diff) > 0:
            direction = 1 if float(diff.median()) > 0 else -1

    t_arr = sub["time"].to_numpy(dtype=float)
    # FPS estimée sur la fenêtre (peut différer légèrement du fps global si
    # frames droppées)
    dt_arr = np.diff(t_arr)
    fps_est = float(1.0 / np.median(dt_arr)) if len(dt_arr) and np.median(dt_arr) > 0 else 30.0

    def _col(part: str, axis: str):
        c = f"{side}_{part}_{axis}"
        return sub[c].to_numpy(dtype=float) if c in sub.columns else np.full(len(sub), np.nan)

    sh_x, sh_y = _col("shoulder", "x"), _col("shoulder", "y")
    hi_x, hi_y = _col("hip",      "x"), _col("hip",      "y")
    kn_x, kn_y = _col("knee",     "x"), _col("knee",     "y")
    an_x, an_y = _col("ankle",    "x"), _col("ankle",    "y")

    def _angle_series(p1x, p1y, p2x, p2y, p3x, p3y):
        """Angle (p1, p2, p3) en degrés, frame par frame, NaN si keypoint manquant."""
        v1x = p1x - p2x;  v1y = p1y - p2y
        v2x = p3x - p2x;  v2y = p3y - p2y
        dot   = v1x * v2x + v1y * v2y
        n1    = np.sqrt(v1x * v1x + v1y * v1y)
        n2    = np.sqrt(v2x * v2x + v2y * v2y)
        with np.errstate(invalid='ignore', divide='ignore'):
            cos = np.clip(dot / (n1 * n2), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    knee  = _angle_series(hi_x, hi_y, kn_x, kn_y, an_x, an_y)
    hip   = _angle_series(sh_x, sh_y, hi_x, hi_y, kn_x, kn_y)
    # Cheville : proxy pointe-pied (+30 px dans la direction du rider)
    toe_x = an_x + 30.0 * direction
    toe_y = an_y
    ankle = _angle_series(kn_x, kn_y, an_x, an_y, toe_x, toe_y)

    def _omega_peak(angle_arr, label):
        """Retourne (omega_max, t_peak_rel_gate, series_downsampled) ou None.
        omega_max = pic de ω positive (extension) en °/s.
        t_peak_rel_gate = timestamp du pic, relatif au gate drop (s).
        series_downsampled = liste {T, omega} pour le graphe (max ~30 points).
        """
        # Interpole les trous courts (≤3 frames) pour permettre savgol/gradient
        s = pd.Series(angle_arr).interpolate(method='linear', limit=3,
                                              limit_area='inside').to_numpy()
        valid = ~np.isnan(s)
        if valid.sum() < 5:
            return None
        # Pour savgol on a besoin d'un signal sans NaN. On comble les NaN
        # résiduels (typiquement aux bords) par forward/backward fill — la
        # sortie omega sera ensuite masquée sur `valid` pour invalider ces
        # zones extrapolées.
        s_for_filter = pd.Series(s).ffill().bfill().to_numpy()
        if np.all(np.isnan(s_for_filter)):
            return None
        n = len(s_for_filter)
        # window adaptatif (impair) ; polyorder=2 < window
        win = min(7, n if n % 2 == 1 else n - 1)
        if win >= 5:
            try:
                s_smooth = savgol_filter(s_for_filter, window_length=win,
                                          polyorder=2, mode='interp')
            except Exception:
                s_smooth = s_for_filter
        else:
            s_smooth = s_for_filter
        # Dérivée temporelle : °/s
        omega = np.gradient(s_smooth, t_arr)
        # On ne garde que les échantillons où l'angle est valide
        omega = np.where(valid, omega, np.nan)
        if np.all(np.isnan(omega)):
            return None
        # Pic de ω positive (extension) uniquement
        omega_pos = np.where(omega > 0, omega, np.nan)
        if np.all(np.isnan(omega_pos)):
            # Articulation reste en flexion ou statique → pas d'extension
            return None

        # Détection de pic robuste : on cherche un maximum local interne
        # (find_peaks). Un maximum local ne peut PAS être en bord, donc
        # s'il existe c'est forcément un vrai pic biomécanique. Si aucun
        # maximum local n'est trouvé → fallback sur argmax + flag edge_peak.
        edge_peak = False
        omega_filled = np.where(np.isnan(omega_pos), -np.inf, omega_pos)
        # prominence proportionnelle au signal pour éviter les micro-pics
        # dus au bruit résiduel post-savgol
        peak_thr = max(30.0, float(np.nanmax(omega_pos)) * 0.4)
        peaks, _ = find_peaks(omega_filled, prominence=peak_thr,
                              distance=max(2, int(0.05 * fps_est)))
        if len(peaks) > 0:
            # Plus haut maximum local
            idx_peak = int(peaks[np.argmax(omega_filled[peaks])])
        else:
            # Fallback : argmax. Si le pic tombe en bord, on flag.
            idx_peak = int(np.nanargmax(omega_pos))
            n_edge   = max(2, int(0.04 * fps_est))   # ~40ms de zone de bord
            if idx_peak < n_edge or idx_peak >= n - n_edge:
                edge_peak = True

        omega_max = float(omega[idx_peak])
        t_peak    = float(t_arr[idx_peak] - gate_t)

        # Downsample la série pour le graphe (max ~40 points)
        step = max(1, n // 40)
        series = [
            {"T": round(float(t_arr[i] - gate_t), 3),
             "omega": round(float(omega[i]), 1) if not np.isnan(omega[i]) else None}
            for i in range(0, n, step)
        ]
        return {
            "omega_max": round(omega_max, 1),
            "t_peak":    round(t_peak, 3),
            "series":    series,
            "edge_peak": edge_peak,
        }

    hip_r   = _omega_peak(hip,   "hip")
    knee_r  = _omega_peak(knee,  "knee")
    ankle_r = _omega_peak(ankle, "ankle")

    # Coordinating Index — séquence proximale → distale (hanche → genou → cheville)
    # Tolérance = 2 frames pour la simultanéité (pics à <2/fps secondes d'écart)
    tol = 2.0 / max(fps_est, 1.0)
    include_ankle = (fps_est >= 60.0) and (ankle_r is not None)
    ci_score, ci_verdict, ci_reason = _coordinating_index(
        hip_r, knee_r, ankle_r if include_ankle else None, tol)

    has_edge = any(j is not None and j.get("edge_peak") for j in (hip_r, knee_r, ankle_r))

    return {
        "fps_est":          round(fps_est, 1),
        "window_pre":       window_pre,
        "window_post":      window_post,
        "n_samples":        int(len(sub)),
        "hip":              hip_r,
        "knee":             knee_r,
        "ankle":            ankle_r,
        "ci_score":         ci_score,
        "ci_verdict":       ci_verdict,
        "ci_reason":        ci_reason,
        "ankle_in_ci":      include_ankle,
        "has_edge_warning": has_edge,
    }


def _coordinating_index(hip, knee, ankle, tol_s: float) -> tuple[int, str, str]:
    """Classe la séquence d'extension :
      score=2 → proximal-distal correct (hanche puis genou [puis cheville])
      score=1 → simultanée (tous les pics dans une fenêtre de ±tol)
      score=0 → inversée (genou avant hanche, ou cheville avant genou)
      score=-1→ données partielles (au moins une articulation manquante)
    Retourne (score, verdict, raison_humaine).
    """
    pts = []
    if hip   is not None: pts.append(("hanche",   hip["t_peak"]))
    if knee  is not None: pts.append(("genou",    knee["t_peak"]))
    if ankle is not None: pts.append(("cheville", ankle["t_peak"]))

    if len(pts) < 2:
        return -1, "partial", "Données insuffisantes pour évaluer la séquence."

    # Trie par instant de pic
    pts_sorted = sorted(pts, key=lambda x: x[1])
    # L'ordre attendu (proximal → distal)
    expected = ["hanche", "genou", "cheville"]
    got      = [name for name, _ in pts_sorted]

    # Vérifie simultanéité : tous les pics dans une fenêtre ±tol
    t_min = min(t for _, t in pts)
    t_max = max(t for _, t in pts)
    if (t_max - t_min) <= tol_s:
        names = " · ".join(f"{n} ({int(round(t*1000))}ms)" for n, t in pts)
        return 1, "simultaneous", f"Pics quasi-simultanés ({names})."

    # Vérifie si l'ordre observé est un sous-ordre de l'attendu
    expected_filtered = [n for n in expected if n in got]
    if got == expected_filtered:
        order_str = " → ".join(f"{n} ({int(round(t*1000))}ms)" for n, t in pts_sorted)
        return 2, "proximal_distal", f"Séquence proximale-distale : {order_str}."
    else:
        order_str = " → ".join(f"{n} ({int(round(t*1000))}ms)" for n, t in pts_sorted)
        return 0, "inverted", f"Séquence non-optimale : {order_str}."


# Version du schéma de cache burst. Incrémenter à chaque changement de
# fenêtre/algo qui invaliderait les anciennes valeurs persistées.
BURST_CACHE_VERSION = "1.1"


def _burst_diagnose(burst: dict, perso: dict | None = None) -> dict:
    """Moteur de règles → verdict + observations + cue coaching.

    Logique (toute documentée, pas de ML) :
      1. Détecte d'abord les cas non-significatifs / mesure incomplète.
      2. Note la séquence (CI) comme observation principale.
      3. Compare ω hanche / genou à la moyenne perso si disponible.
      4. Détecte des patterns techniques (délai hanche→genou anormal, etc.).
      5. Choisit UN cue d'entraînement basé sur le défaut dominant.

    Sources :
      - Gross 2017 — coordination proximale-distale, ω élites 400-700°/s
      - Cowell 2020 — séquencement hanche→genou→cheville
      - Grigg 2020 — back position, role du pull-back
      - Mahieu — alignement épaules-bassin pour transmission

    Retour : {verdict, verdict_label, headline, observations:[{type, text}], cue:{text, why}}.
    """
    obs: list[dict] = []
    hip   = burst.get("hip")
    knee  = burst.get("knee")
    ankle = burst.get("ankle")

    # ── Cas dégénérés ────────────────────────────────────────────────
    if not hip or not knee:
        return {
            "verdict": "incomplete",
            "verdict_label": "Données insuffisantes",
            "headline": "Le pipeline n'a pas extrait assez de keypoints sur cet essai.",
            "observations": [{"type": "warn",
                "text": "Au moins l'une des articulations principales (hanche, genou) "
                        "n'a pas de pic d'extension détecté."}],
            "cue": None,
        }

    hip_omega   = hip.get("omega_max") or 0
    knee_omega  = knee.get("omega_max") or 0
    hip_tpeak   = hip.get("t_peak")
    knee_tpeak  = knee.get("t_peak")

    # Essai non-significatif (warmup, drill statique, gate raté…)
    if hip_omega < 50 and knee_omega < 50:
        return {
            "verdict": "non_significant",
            "verdict_label": "Essai non-significatif",
            "headline": "Pas d'extension franche détectée — probablement un warmup, "
                        "un drill statique ou un départ avorté.",
            "observations": [
                {"type": "warn", "text": f"ω hanche {hip_omega:.0f}°/s, "
                                          f"ω genou {knee_omega:.0f}°/s — trop faible pour analyser."},
                {"type": "info", "text": "Cet essai est exclu du calcul du référentiel personnel."},
            ],
            "cue": None,
        }

    # Mesure en bord de fenêtre → fiabilité douteuse
    if burst.get("has_edge_warning"):
        edges = [name for name, j in (("hanche", hip), ("genou", knee), ("cheville", ankle))
                 if j and j.get("edge_peak")]
        obs.append({
            "type": "warn",
            "text": f"Pic en bord de fenêtre ({', '.join(edges)}) — la vidéo coupe "
                    "trop près du gate drop ou de la phase d'extension. Refilme avec "
                    "0.5s avant le bip 1 et 1.5s après le gate pour une mesure complète.",
        })

    # ── Séquence (CI) ────────────────────────────────────────────────
    ci = burst.get("ci_verdict")
    if ci == "proximal_distal":
        obs.append({"type": "good",
            "text": f"Séquence proximale-distale respectée — {burst.get('ci_reason', '')}"})
    elif ci == "simultaneous":
        obs.append({"type": "info",
            "text": f"Pics quasi-simultanés — coordination correcte mais peu "
                    f"discriminante à {burst.get('fps_est', 30)} fps."})
    elif ci == "inverted":
        obs.append({"type": "bad",
            "text": f"Séquence inversée — {burst.get('ci_reason', '')} "
                    "Pattern non-optimal : tu déclenches le genou avant la hanche."})

    # ── Comparaison perso ───────────────────────────────────────────
    has_perso_hip   = perso and perso.get("omega", {}).get("hip")  \
                      and perso["omega"]["hip"].get("n", 0) >= 3
    has_perso_knee  = perso and perso.get("omega", {}).get("knee") \
                      and perso["omega"]["knee"].get("n", 0) >= 3

    # Comparaison vs perso = chute de PERFORMANCE, pas défaut TECHNIQUE.
    # On utilise le type "dip" / "boost" / "record" pour ne pas inflater
    # le verdict global. Le frontend les style comme info, pas comme bad.
    def _cmp_obs(joint_name, current, ref, unit="°/s"):
        if not ref or ref.get("n", 0) < 3: return None
        mean = ref["mean"]; best = ref["best"]
        delta_pct = (current - mean) / mean * 100 if mean > 0 else 0
        if current > best * 1.02:
            return {"type": "record",
                "text": f"ω {joint_name} {current:.0f}{unit} = nouveau record perso "
                        f"(best précédent : {best:.0f}{unit}, n={ref['n']})."}
        if current >= best * 0.97:
            return {"type": "boost",
                "text": f"ω {joint_name} {current:.0f}{unit} = au niveau de ton meilleur "
                        f"({best:.0f}{unit}, n={ref['n']})."}
        if delta_pct >= 10:
            return {"type": "boost",
                "text": f"ω {joint_name} {current:.0f}{unit} → +{delta_pct:.0f}% "
                        f"vs ta moyenne ({mean:.0f}{unit}, n={ref['n']})."}
        if delta_pct <= -20:
            return {"type": "dip",
                "text": f"ω {joint_name} {current:.0f}{unit} → {delta_pct:.0f}% "
                        f"vs ta moyenne ({mean:.0f}{unit}, n={ref['n']}). "
                        "Essai sous ton niveau habituel — fatigue ? échauffement ?"}
        if delta_pct <= -10:
            return {"type": "dip",
                "text": f"ω {joint_name} {current:.0f}{unit} → {delta_pct:.0f}% "
                        f"vs ta moyenne ({mean:.0f}{unit}, n={ref['n']})."}
        return None  # dans la zone normale, on ne dit rien

    if has_perso_hip:
        o = _cmp_obs("hanche", hip_omega, perso["omega"]["hip"])
        if o: obs.append(o)
    if has_perso_knee:
        o = _cmp_obs("genou",  knee_omega, perso["omega"]["knee"])
        if o: obs.append(o)

    # ── Pattern : délai hanche→genou ────────────────────────────────
    delay_hip_knee_ms = None
    if hip_tpeak is not None and knee_tpeak is not None:
        delay_hip_knee_ms = (knee_tpeak - hip_tpeak) * 1000
        if delay_hip_knee_ms > 450:
            obs.append({"type": "warn",
                "text": f"Délai hanche→genou très long ({delay_hip_knee_ms:.0f}ms) — "
                        "ta poussée de quad arrive tard, tu perds le bénéfice du "
                        "transfert proximal-distal."})

    # ── Patterns d'amplitude (seulement si pas de perso pour ne pas
    #    répéter le même message) ────────────────────────────────────
    if not has_perso_hip and hip_omega < 100:
        obs.append({"type": "warn",
            "text": f"ω hanche faible ({hip_omega:.0f}°/s) — engagement hanche "
                    "limité. Les extensions hanche/genou au départ BMX devraient "
                    "être proches d'un saut balistique."})

    # ── Choix du verdict global ─────────────────────────────────────
    # Le verdict reflète la TECHNIQUE (CI, séquence, amplitude absolue).
    # Les chutes/boost vs perso (types dip/boost/record) sont des indicateurs
    # de FORME, pas de technique → ils colorient la headline mais ne font
    # pas basculer le verdict en bad.
    n_bad     = sum(1 for o in obs if o["type"] == "bad")
    n_warn    = sum(1 for o in obs if o["type"] == "warn")
    n_good    = sum(1 for o in obs if o["type"] == "good")
    n_record  = sum(1 for o in obs if o["type"] == "record")
    n_dip     = sum(1 for o in obs if o["type"] == "dip")

    if n_bad >= 1:
        verdict, label = "bad", "Départ à retravailler"
        headline = "Au moins un défaut technique majeur détecté."
    elif n_warn >= 2:
        verdict, label = "warn", "Départ correct, points à corriger"
        headline = "Bonne base mais plusieurs points à travailler."
    elif n_good >= 1 and n_warn == 0:
        if n_record >= 1:
            verdict, label = "good", "Excellent départ"
            headline = "Mécanique propre ET niveau supérieur à ton habitude."
        elif n_dip >= 1:
            verdict, label = "good", "Bon départ (forme en baisse)"
            headline = "Technique propre mais tu es sous ton niveau habituel."
        else:
            verdict, label = "good", "Bon départ"
            headline = "Exécution propre — focus sur la stabilité et l'explosivité."
    elif n_warn == 1 and n_good >= 1:
        verdict, label = "ok", "Départ correct"
        headline = "Mécanique acceptable, un point à surveiller."
    else:
        verdict, label = "ok", "Départ correct"
        headline = "Rien d'alarmant, mais peu de marqueurs forts non plus."

    # ── Choix du cue d'entraînement ─────────────────────────────────
    # On choisit UN cue basé sur le défaut dominant. Ordre de priorité :
    # 1) séquence inversée → travail postural & timing
    # 2) délai hanche↔genou trop long → explosivité quad
    # 3) ω hanche faible → engagement hanche / pull-back
    # 4) tout va bien → consigne de consolidation
    cue = None
    if ci == "inverted":
        cue = {
            "text": "Pull-back plus marqué : reste BAS dans la phase de set, "
                    "garde le tronc penché jusqu'au bip 3, puis hanche AVANT genou.",
            "why": "Une séquence inversée vient typiquement d'un buste trop "
                   "redressé en set — le genou prend le relai parce que la hanche "
                   "n'a plus de course à extension. (Grigg 2020 — back position)"
        }
    elif delay_hip_knee_ms is not None and delay_hip_knee_ms > 450:
        cue = {
            "text": "Drill explosivité quad : squat jumps avec charge légère, "
                    "5×5 à intensité max, en visant l'extension la plus rapide.",
            "why": "Ton délai hanche→genou est long : ta hanche tire mais ton "
                   "quad ne suit pas avec la même vitesse d'extension. (Gross 2017 — "
                   "ω genou élites 400-700°/s, comparable saut balistique chargé)"
        }
    elif hip_omega < 100 and not has_perso_hip:
        cue = {
            "text": "Travail engagement hanche : box jumps et hip thrusts explosifs. "
                    "Sur le vélo : exagère le pull-back en set (épaules devant l'axe "
                    "guidon, bassin reculé).",
            "why": "ω hanche en-dessous des standards de saut balistique. "
                   "La hanche est le moteur du transfert proximal-distal."
        }
    elif n_dip >= 2:
        cue = {
            "text": "Forme du jour : enchaîne 2-3 essais d'échauffement complet "
                    "(activation hanche/quad) avant de retester. Si l'écart persiste, "
                    "regarde la récup' (sommeil, charge récente).",
            "why": f"Tu es {abs(int(perso['omega']['knee']['mean'] - knee_omega) / perso['omega']['knee']['mean'] * 100) if has_perso_knee else 0:.0f}% "
                   "sous ta moyenne sur ≥2 articulations — ta technique est bonne, "
                   "mais l'explosivité du jour n'y est pas. Pas un problème de technique."
        }
    elif verdict == "good":
        cue = {
            "text": "Consolide : enchaîne 5-8 essais aujourd'hui et vise la "
                    "consistance des chiffres plutôt que le pic isolé.",
            "why": "Tu as une bonne mécanique. À ce stade, c'est la reproductibilité "
                   "qui fait gagner en course."
        }
    else:
        # Fallback générique : pas de défaut spécifique détecté
        cue = {
            "text": "Continue ton volume habituel. Pour progresser sur l'explosivité, "
                    "alterne séances power (squat jumps, sprints départ arrêté) et "
                    "séances vélo spécifique départ rampe (5-8 sets, récup complète).",
            "why": "Pas de défaut technique dominant à corriger. La progression vient "
                   "maintenant du volume de travail explosif et de la consistance."
        }

    return {
        "verdict": verdict,
        "verdict_label": label,
        "headline": headline,
        "observations": obs,
        "cue": cue,
    }


def _get_or_compute_burst(job_id: str, job: dict, force: bool = False) -> dict | None:
    """Renvoie le burst d'un job, en utilisant un cache persisté dans
    `job['results']['kinematic_burst']` si la version correspond. Calcule et
    persiste sinon. Retourne None si calcul impossible."""
    results = job.get("results") or {}
    cached  = results.get("kinematic_burst")
    if not force and cached and cached.get("_version") == BURST_CACHE_VERSION:
        return cached
    csv_path = OUTPUT_DIR / results.get("files", {}).get("landmarks_csv", "")
    side     = results.get("front_foot") or "L"
    gate_t   = float(results.get("gate_drop_t", 0.0))
    burst    = _compute_kinematic_burst(csv_path, gate_t, side)
    if burst is None:
        return None
    burst["_version"] = BURST_CACHE_VERSION
    results["kinematic_burst"] = burst
    job["results"] = results
    # Seuls les jobs avec athlète sont persistés (cf. save_jobs).
    if job.get("athlete_id"):
        save_jobs(jobs)
    return burst


@app.get("/explosivity/{job_id}")
async def explosivity(job_id: str):
    """Renvoie ω_max (°/s) + séquence proximale-distale + verdict coaching
    pour un job analysé. Le burst est mis en cache dans le job ; le verdict
    est calculé à la volée (rapide) car il dépend du référentiel perso en
    temps réel."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    burst = _get_or_compute_burst(job_id, job)
    if burst is None:
        return JSONResponse({"error": "Données insuffisantes sur la fenêtre post-gate."},
                            status_code=422)

    # Diagnostic coaching avec référentiel perso si athlète assigné
    perso = None
    aid = job.get("athlete_id")
    if aid and aid in athletes:
        # Recalcule les stats en excluant l'essai courant (cohérent avec l'UI)
        aj = [(jid, j) for jid, j in jobs.items()
              if j.get("status") == "done"
              and j.get("athlete_id") == aid
              and not j.get("excluded")
              and jid != job_id]
        sig_bursts = [b for b in (_get_or_compute_burst(jid, j) for jid, j in aj)
                      if _is_burst_significant(b)]
        def _agg(joint):
            vals = [b[joint]["omega_max"] for b in sig_bursts
                    if b.get(joint) and b[joint].get("omega_max") is not None
                    and not b[joint].get("edge_peak")]
            if not vals: return None
            return {"mean": round(float(np.mean(vals)), 1),
                    "best": round(float(np.max(vals)), 1),
                    "n":    len(vals)}
        perso = {"omega": {j: _agg(j) for j in ("hip", "knee", "ankle")}}

    diag = _burst_diagnose(burst, perso)
    return {**burst, "diagnosis": diag}


def _is_burst_significant(burst: dict, min_omega: float = 50.0) -> bool:
    """Filtre les essais non-significatifs (warmups, drills statiques) pour
    le calcul des stats perso. Un essai est gardé si les DEUX articulations
    principales (hanche + genou) montrent une extension franche."""
    if burst is None: return False
    hip = burst.get("hip");  knee = burst.get("knee")
    if not hip or not knee: return False
    return (hip.get("omega_max") or 0) > min_omega \
       and (knee.get("omega_max") or 0) > min_omega


@app.get("/athletes/{athlete_id}/burst_stats")
async def athlete_burst_stats(athlete_id: str, exclude_job: str = ""):
    """Agrège les ω_max et t_peak de tous les départs significatifs d'un
    athlète. Sert de référentiel personnel pour la carte Explosivité.

    `exclude_job` permet d'exclure l'essai en cours d'affichage si on veut
    que la moyenne représente l'historique HORS essai courant (sinon le
    chiffre courant biaise la moyenne quand l'athlète a peu d'essais).
    """
    if athlete_id not in athletes:
        return JSONResponse({"error": "Athlète introuvable."}, status_code=404)

    # Collecte tous les jobs done de cet athlète
    aj = [(jid, j) for jid, j in jobs.items()
          if j.get("status") == "done"
          and j.get("athlete_id") == athlete_id
          and not j.get("excluded")
          and jid != exclude_job]

    bursts: list[dict] = []
    for jid, j in aj:
        b = _get_or_compute_burst(jid, j)
        if _is_burst_significant(b):
            bursts.append(b)

    def _agg(joint: str, field: str) -> dict | None:
        vals = [b[joint][field] for b in bursts
                if b.get(joint) and b[joint].get(field) is not None
                and not b[joint].get("edge_peak")]
        if not vals: return None
        return {
            "mean":   round(float(np.mean(vals)),   1),
            "median": round(float(np.median(vals)), 1),
            "best":   round(float(np.max(vals)),    1),
            "n":      len(vals),
        }

    return {
        "athlete_id":     athlete_id,
        "athlete_name":   athletes[athlete_id].get("name", ""),
        "n_significant":  len(bursts),
        "n_total":        len(aj),
        "omega": {
            "hip":   _agg("hip",   "omega_max"),
            "knee":  _agg("knee",  "omega_max"),
            "ankle": _agg("ankle", "omega_max"),
        },
        "t_peak_ms": {
            # t_peak est en secondes, on remonte en ms pour l'UI
            joint: ({
                "mean":   round(v["mean"]   * 1000, 0),
                "median": round(v["median"] * 1000, 0),
                "best":   round(v["best"]   * 1000, 0),  # "best" t_peak = le plus précoce ? non, juste le max
                "n":      v["n"],
            } if (v := _agg(joint, "t_peak")) else None)
            for joint in ("hip", "knee", "ankle")
        },
    }


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

