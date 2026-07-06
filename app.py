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
from fastapi.responses import JSONResponse, Response, RedirectResponse
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
RACES_DB      = OUTPUT_DIR / "races_db.json"
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

def load_races() -> dict:
    if RACES_DB.exists():
        try:
            return json.loads(RACES_DB.read_text())
        except Exception:
            pass
    return {}

def save_races(races: dict):
    _save_db_with_backup(RACES_DB, json.dumps(races, indent=2, ensure_ascii=False))

pros:     dict = load_pros()
jobs:     dict = load_jobs()
athletes: dict = load_athletes()
tracks:   dict = load_tracks()
races:    dict = load_races()


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
            "files":        j.get("results", {}).get("files", {}),
            "front_foot":   j.get("results", {}).get("front_foot"),
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


def _reaction_ms(results: dict | None) -> int | None:
    """Temps de réaction recalé (1er bip → 1er mouvement) = from_gate_ms + 360.
    On dérive le 1er bip de la grille via la cadence UCI fixe (la grille tombe
    360 ms après le 1er bip) plutôt que d'utiliser le bip audio brut, souvent
    faussé par un écho. Voir _analyze_reaction / SCIENCE.md §2.
    None si N/A, faux départ, OU si le mouvement n'a pas été capté (pose trop
    faible) — pour ne pas polluer les stats/graphes avec un chiffre inventé.
    `results` doit contenir `reaction` (+ `files`/`gate_drop_t`/`front_foot`
    pour le contrôle de fiabilité)."""
    if not results:
        return None
    reaction = results.get("reaction") or {}
    # SEULES les réactions confirmées par le coach (grille + 1er mouvement
    # vérifiés à l'œil) comptent dans les stats et la progression. Les
    # estimations logicielles ne sont jamais comptées.
    v = reaction.get("verified")
    if not v or v.get("from_gate_ms") is None:
        return None
    return round(float(v["from_gate_ms"]) + GATE_FROM_BIP1_MS)


def _athlete_stats(athlete_jobs: list[dict]) -> dict:
    """Mini dashboard : nb de vidéos, meilleur / moyen temps de réaction (excluant
    les faux départs ET les sessions cochées comme exclues par l'utilisateur).
    Métrique = réaction recalée (1er bip → 1er mouvement, ancrée sur la grille)."""
    reactions_ms = []
    n_excluded   = 0
    for j in athlete_jobs:
        if j.get("excluded"):
            n_excluded += 1
            continue
        rms = _reaction_ms(j)   # j contient reaction + files + gate_drop_t + front_foot
        if rms is not None:
            reactions_ms.append(rms)
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
    info = {"fps": round(fps, 3), "n_frames": n_frames,
            "duration_s": round(n_frames / fps, 2)}
    info["precision"] = precision_tier(fps)
    return info


def precision_tier(fps: float) -> dict:
    """Classe la précision temporelle d'une vidéo selon sa cadence.

    Le pas de temps = 1000/fps ms est le plancher de précision : on ne peut pas
    situer un événement (gate, 1er mouvement, pic d'extension) plus finement
    qu'une image. Honnêteté scientifique : on AFFICHE cette limite, on ne
    prétend jamais une précision qu'on n'a pas. Voir SCIENCE.md.

    Retour : {fps, frame_ms, tier (excellent/good/fair/limited), label, verdict
    (ok/warn/bad), note}.
    """
    fps = float(fps or 30.0)
    frame_ms = round(1000.0 / fps) if fps > 0 else 33
    if fps >= 119:
        tier, label, verdict = "excellent", "Précision excellente", "ok"
        note = f"~{frame_ms} ms par image : timing et vitesses très fiables."
    elif fps >= 59:
        tier, label, verdict = "good", "Bonne précision", "ok"
        note = f"~{frame_ms} ms par image : bon compromis pour le départ."
    elif fps >= 48:
        tier, label, verdict = "fair", "Précision correcte", "warn"
        note = f"~{frame_ms} ms par image. Filme en 120 fps pour doubler la finesse."
    else:
        tier, label, verdict = "limited", "Précision limitée", "warn"
        note = (f"~{frame_ms} ms par image : c'est le plancher de précision du timing "
                f"et des vitesses. Filme en 60 ou 120 fps (mode ralenti) pour des "
                f"mesures nettement plus fines.")
    return {"fps": round(fps, 1), "frame_ms": frame_ms, "tier": tier,
            "label": label, "verdict": verdict, "note": note}


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


@app.get("/landing")
async def landing(request: Request):
    """Page vitrine — démo animée, méthode, science. L'app de travail reste sur /."""
    return templates.TemplateResponse(request, "landing.html", {})


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
        "precision": info.get("precision"),
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
            "gate_time_from_bip1": res.get("gate_t_from_bip1"),
            "beeps_t":          res["beeps_t"],
            "mean_interval_ms": res["mean_interval_ms"],
            "cadence_dev_ms":   res.get("cadence_dev_ms"),
            "cadence_ok":       res.get("cadence_ok"),
            "confidence":       res["confidence"],
            "quality":          res.get("quality", "medium"),
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


def _recompute_from_gate(job: dict, gate_frame: int) -> dict:
    """Recalcule phases + réaction depuis le CSV avec un nouveau gate drop
    (< 1 s, sans re-pose-detection). Mute job['results'] en place.
    Retourne {ok: True, ...} ou {ok: False, error: str}."""
    results  = job.get("results", {})
    csv_name = results.get("files", {}).get("landmarks_csv")
    if not csv_name:
        return {"ok": False, "error": "CSV landmarks introuvable."}
    csv_path = OUTPUT_DIR / csv_name
    if not csv_path.exists():
        return {"ok": False, "error": f"Fichier CSV manquant : {csv_name}"}

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
        return {"ok": False, "error": f"Erreur recalcul phases : {e}"}

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
    return {"ok": True, "gate_drop_t": new_gate, "first_move_t": t_move}


def _persist_results(job_id: str, job: dict):
    """Réécrit le fichier {job_id}_results.json pour que la récupération
    (get_job_or_recover) conserve les modifications faites après l'analyse."""
    try:
        results_path = OUTPUT_DIR / f"{job_id}_results.json"
        results_path.write_text(json.dumps(job.get("results", {}),
                                           indent=2, ensure_ascii=False))
    except Exception:
        pass   # non bloquant : jobs_db.json reste la source primaire


@app.post("/result/{job_id}/adjust_gate")
async def adjust_gate(job_id: str, gate_frame: int = Form(...)):
    """Recalcule les phases / temps de réaction avec un nouveau gate drop,
    sans repasser par la pose detection."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."}, status_code=404)
    res = _recompute_from_gate(job, gate_frame)
    if not res.get("ok"):
        return JSONResponse({"error": res.get("error", "Erreur.")}, status_code=500)
    save_jobs(jobs)
    _persist_results(job_id, job)
    return {"ok": True, "gate_drop_t": res["gate_drop_t"], "first_move_t": res["first_move_t"]}


# ── Confirmation coach de la réaction ─────────────────────────────────────────
# Philosophie : un temps de réaction n'est ÉVALUÉ (noté, compté dans les stats)
# que si le coach a confirmé À L'ŒIL les deux instants qui le composent : la
# chute de la grille ET le premier mouvement. Le logiciel pré-place les
# marqueurs (détection raffinée) mais ne fait qu'estimer tant que le coach n'a
# pas validé. « Continuer sans » = l'estimation reste affichée, jamais comptée.
@app.post("/jobs/{job_id}/reaction_verify")
async def reaction_verify(job_id: str,
                          gate_frame: int = Form(...),
                          move_frame: int = Form(...)):
    """Le coach confirme la grille + le 1er mouvement. Si la grille diffère du
    marquage actuel, les phases sont recalculées (cohérence globale)."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."}, status_code=404)
    results = job.get("results", {})
    fps = float(results.get("fps") or 30)
    if move_frame < gate_frame - int(1.0 * fps):
        return JSONResponse({"error": "Le 1er mouvement est plus d'une seconde avant "
                                      "la grille — vérifie tes deux marqueurs."},
                            status_code=400)

    # Grille corrigée → recalcul complet des phases (machinerie adjust_gate).
    cur_gate_frame = int(round(float(results.get("gate_drop_t", 0)) * fps))
    if gate_frame != cur_gate_frame:
        res = _recompute_from_gate(job, gate_frame)
        if not res.get("ok"):
            return JSONResponse({"error": res.get("error", "Erreur.")}, status_code=500)
        results = job.get("results", {})

    gate_t = float(results.get("gate_drop_t", gate_frame / fps))
    move_t = move_frame / fps
    rx = results.setdefault("reaction", {})
    rx["verified"] = {
        "gate_frame":   int(gate_frame),
        "move_frame":   int(move_frame),
        "gate_t":       round(gate_t, 3),
        "move_t":       round(move_t, 3),
        "from_gate_ms": round((move_t - gate_t) * 1000),
        "verified_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    rx.pop("verify_skipped", None)
    save_jobs(jobs)
    _persist_results(job_id, job)
    return {"ok": True, "verified": rx["verified"]}


@app.delete("/jobs/{job_id}/reaction_verify")
async def reaction_unverify(job_id: str):
    """Retire la confirmation coach (retour à l'estimation logicielle)."""
    job = get_job_or_recover(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    rx = (job.get("results") or {}).get("reaction") or {}
    rx.pop("verified", None)
    save_jobs(jobs)
    _persist_results(job_id, job)
    return {"ok": True}


@app.post("/jobs/{job_id}/reaction_skip")
async def reaction_skip(job_id: str):
    """« Continuer sans confirmer » : l'estimation reste affichée (discrète),
    jamais notée ni comptée dans les stats."""
    job = get_job_or_recover(job_id)
    if not job:
        return JSONResponse({"error": "Job introuvable."}, status_code=404)
    rx = (job.get("results") or {}).setdefault("reaction", {})
    rx["verify_skipped"] = True
    save_jobs(jobs)
    _persist_results(job_id, job)
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
        "precision":           precision_tier(job["results"].get("fps", 30)),
        "reaction":            _analyze_reaction(job["results"]),
        "raw_video_url":       job.get("video_url"),
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


# ── Courses & préparation (module « aide au coureur ») ───────────────────────
# Beta : un calendrier de courses (rattachées à un athlète) + une carte
# « prochaine course » avec compte à rebours. La prépa par course (6 piliers)
# arrive en Phase 2. Conçu pour qu'une vue coach multi-athlètes s'ajoute ensuite.
RACE_LEVELS = ["Entraînement", "Régionale", "Provinciale", "Nationale",
               "Coupe", "Championnat", "International"]


def _race_countdown(race: dict) -> dict:
    """Calcule les infos de compte à rebours d'une course par rapport à
    aujourd'hui : jours restants (négatif = passée), libellé J-x, statut."""
    out = {"days": None, "label": "", "status": "unknown", "is_next": False}
    d = race.get("date")
    if not d:
        return out
    try:
        race_d = datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        return out
    today = datetime.now().date()
    days = (race_d - today).days
    out["days"] = days
    if days > 1:
        out["label"], out["status"] = f"J-{days}", "upcoming"
    elif days == 1:
        out["label"], out["status"] = "Demain", "imminent"
    elif days == 0:
        out["label"], out["status"] = "Aujourd'hui", "today"
    else:
        out["label"], out["status"] = f"Il y a {abs(days)} j", "past"
    return out


def _races_sorted(athlete_id: str | None = None) -> list[dict]:
    """Liste des courses (enrichies du compte à rebours + nom d'athlète),
    triées par date croissante. Filtre optionnel par athlète."""
    items = []
    for rid, r in races.items():
        if athlete_id and r.get("athlete_id") != athlete_id:
            continue
        athlete = athletes.get(r.get("athlete_id") or "")
        items.append({
            **r,
            "countdown":     _race_countdown(r),
            "athlete_name":  athlete.get("name") if athlete else None,
        })
    items.sort(key=lambda x: x.get("date") or "9999")
    return items


def _next_race(race_list: list[dict]) -> dict | None:
    """La prochaine course (aujourd'hui ou future) la plus proche, sinon None."""
    upcoming = [r for r in race_list
                if r["countdown"]["days"] is not None and r["countdown"]["days"] >= 0]
    return upcoming[0] if upcoming else None


@app.get("/prepa")
async def prepa_page(request: Request, athlete: str = ""):
    """Page « Préparation » : carte prochaine course + calendrier + liste."""
    aid = athlete if athlete in athletes else None
    race_list = _races_sorted(aid)
    upcoming = [r for r in race_list if r["countdown"]["status"] != "past"]
    past     = [r for r in race_list if r["countdown"]["status"] == "past"]
    past.reverse()  # plus récentes d'abord
    athletes_list = sorted(athletes.values(), key=lambda a: a.get("name", "").lower())
    cal_json = json.dumps([{"id": r["id"], "name": r["name"], "date": r.get("date")}
                           for r in race_list], ensure_ascii=False)
    return templates.TemplateResponse(request, "prepa.html", {
        "active":         "prepa",
        "next_race":      _next_race(race_list),
        "upcoming":       upcoming,
        "past":           past,
        "all_races":      race_list,
        "all_races_json": cal_json,
        "athletes_list":  athletes_list,
        "current_athlete": aid,
        "race_levels":    RACE_LEVELS,
        "today":          datetime.now().strftime("%Y-%m-%d"),
    })


@app.post("/races")
async def races_create(name: str = Form(...),
                       date: str = Form(...),
                       athlete_id: str = Form(""),
                       location: str = Form(""),
                       level: str = Form(""),
                       notes: str = Form("")):
    """Crée une course rattachée (optionnellement) à un athlète."""
    name = (name or "").strip()
    date = (date or "").strip()
    if not name:
        return JSONResponse({"error": "Nom de course requis."}, status_code=400)
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return JSONResponse({"error": "Date invalide (AAAA-MM-JJ)."}, status_code=400)
    aid = (athlete_id or "").strip() or None
    if aid and aid not in athletes:
        aid = None
    rid = str(uuid.uuid4())[:8]
    races[rid] = {
        "id":         rid,
        "name":       name[:80],
        "date":       date,
        "athlete_id": aid,
        "location":   (location or "").strip()[:80],
        "level":      level if level in RACE_LEVELS else "",
        "notes":      (notes or "").strip()[:1000],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_races(races)
    return {"ok": True, "id": rid}


@app.delete("/races/{race_id}")
async def races_delete(race_id: str):
    if race_id not in races:
        return JSONResponse({"error": "Course introuvable."}, status_code=404)
    del races[race_id]
    save_races(races)
    return {"ok": True}


# ── Plan de préparation par course (Phase 2) ─────────────────────────────────
# Contenu basé sur des règles (pas d'IA) : repères sport-science généraux,
# étiquetés « à ajuster ». Honnêteté : ce n'est pas un avis médical ni un plan
# nutritionnel personnalisé. Tout est calibrable ici, dans le code.
PREP_DISCLAIMER = ("Repères généraux de préparation, à adapter à chaque athlète. "
                   "Ce n'est pas un avis médical ni un plan nutritionnel personnalisé.")

# Phases relatives à la course (par jours restants).
PREP_PHASES = [
    {"key": "far",     "label": "Plus d'une semaine"},
    {"key": "week",    "label": "J-7 → J-4 · Semaine"},
    {"key": "taper",   "label": "J-3 → J-2 · Affûtage"},
    {"key": "eve",     "label": "J-1 · Veille"},
    {"key": "raceday", "label": "Jour J"},
    {"key": "after",   "label": "J+1 · Récupération"},
]
PREP_FOCUS = {
    "far":     {"head": "La course est encore loin.",
                "actions": ["Continue ton plan normal.",
                            "Place tes grosses séances de départs maintenant, pas la dernière semaine."]},
    "week":    {"head": "Dernières séances de qualité.",
                "actions": ["Travaille le défaut vu à l'analyse, à haute intensité.",
                            "Aucune nouveauté (braquet, technique) à partir d'ici.",
                            "Soigne ton sommeil dès cette semaine."]},
    "taper":   {"head": "Affûtage — on cherche la fraîcheur.",
                "actions": ["Réduis le volume (~40–50 %), garde l'intensité : départs courts et explosifs.",
                            "Emphase glucides, repas que tu digères bien.",
                            "La nuit la plus importante, c'est l'avant-veille (J-2)."]},
    "eve":     {"head": "Veille — tout est prêt, on se calme.",
                "actions": ["Activation légère seulement, quelques départs à vide.",
                            "Repas du soir riche en glucides, tôt. Sac prêt.",
                            "Couche-toi tôt, écran coupé."]},
    "raceday": {"head": "Jour J — suis ta routine.",
                "actions": ["Échauffement progressif + 2–3 départs d'essai.",
                            "Entre les manches : boire + petite collation, rester au chaud.",
                            "Une moto à la fois."]},
    "after":   {"head": "Récupération.",
                "actions": ["Recharge dans les 1–2 h (glucides + protéines), réhydrate.",
                            "Upload tes départs du jour pour les analyser à froid."]},
}

# 5 piliers phasés (le 6e, « Jour J heure par heure », est la timeline ci-dessous).
PREP_PILLARS = [
    {"key": "charge", "label": "Charge & entraînement", "phases": {
        "far":  ["Entraînement normal. Tes plus grosses séances de départs, c'est maintenant."],
        "week": ["Dernières séances de qualité : départs à haute intensité, volume normal.",
                 "C'est le moment de corriger le défaut détecté à l'analyse.",
                 "Plus aucune nouveauté de matériel ou de technique."],
        "taper":["Affûtage : volume −40 à −50 %, intensité conservée.",
                 "Des départs courts, explosifs, peu nombreux. Pas de séance qui fatigue."],
        "eve":  ["Activation légère : quelques départs à vide pour rester affûté, sans forcer.",
                 "Jambes légères, repos relatif."],
        "raceday":["Tes départs d'essai font partie de l'échauffement, pas de l'entraînement."],
        "after":["Récup active légère (vélo souple, mobilité). Pas de grosse séance."],
    }},
    {"key": "nutrition", "label": "Nutrition & hydratation", "phases": {
        "far":  ["Alimentation équilibrée habituelle. Rien de radical avant une course."],
        "week": ["Assez de glucides pour soutenir l'entraînement, hydratation régulière.",
                 "Teste ton repas d'avant-course à l'entraînement — jamais de nouveauté le jour J."],
        "taper":["Légère emphase glucides (pâtes, riz, patate) pour faire le plein.",
                 "Garde des repas que tu digères bien."],
        "eve":  ["Repas du soir riche en glucides, pauvre en gras/fibres lourdes, pas trop tard.",
                 "Bois bien dans la journée. Évite l'alcool et les plats inhabituels."],
        "raceday":["Petit-déj 3 h avant la 1re moto : glucides + un peu de protéines, faible en gras.",
                   "Entre les manches : collation facile (banane, barre, compote) + boire avec électrolytes.",
                   "Ne pars jamais à jeun sur une moto."],
        "after":["Recharge dans les 1–2 h : glucides + protéines. Réhydrate."],
    }},
    {"key": "mental", "label": "Mental & échauffement", "phases": {
        "far":  ["Ajoute 5 min de visualisation de départ à tes séances : le bip, la grille, ta 1re pédale."],
        "week": ["Travaille ta routine de départ (respiration, point de fixation, déclencheur) jusqu'à l'automatisme."],
        "taper":["Visualisation quotidienne : revois un départ parfait avec le défaut corrigé.",
                 "Prépare ton plan anti-stress (respiration, mots-clés)."],
        "eve":  ["Répétition mentale calme le soir. Visualise la piste si tu la connais.",
                 "Sac prêt pour ne penser à rien le matin."],
        "raceday":["Échauffement progressif + 2–3 départs d'essai. Active ta routine à chaque moto.",
                   "Reste dans l'instant : une moto à la fois."],
        "after":["Note à chaud ce qui a marché sur tes départs (pour la prochaine analyse)."],
    }},
    {"key": "recup", "label": "Récupération & sommeil", "phases": {
        "far":  ["Sommeil régulier. Ça se construit sur des semaines, pas la veille."],
        "week": ["Priorité au sommeil (7–9 h), heures régulières."],
        "taper":["La nuit la plus importante est l'avant-veille (J-2) — celle d'avant peut être agitée, c'est normal.",
                 "Sieste courte possible, pas trop tard."],
        "eve":  ["Couche-toi tôt, écran coupé. Mal dormir de stress n'enlève pas ta forme."],
        "raceday":["Entre les manches : au chaud, jambes surélevées si possible, récupère entre les efforts.",
                   "Garde de l'énergie pour la finale."],
        "after":["Sommeil de récupération, hydratation, mobilité douce."],
    }},
    {"key": "strategie", "label": "Stratégie de course", "phases": {
        "far":  ["Renseigne-toi sur la piste (profil, 1er virage, sauts clés) si tu ne la connais pas."],
        "week": ["Définis ton objectif réaliste et ton plan : c'est quoi un bon run pour toi ?"],
        "taper":["Étudie le tracé : lignes, endroits pour doubler, pièges. Mentalise ton 1er droit."],
        "eve":  ["Repère les horaires (motos, quarts, demies, finale), ta plaque, le check-in."],
        "raceday":["Reconnaissance à pied : teste tes lignes, le 1er virage, les sauts.",
                   "Plan par phase : motos = se placer · quarts/demies = qualifier · finale = tout donner."],
        "after":["Débrief : départs, virages, ce qui a changé le résultat."],
    }},
]

# Timeline du Jour J (le 6e pilier). Générique, à ajuster selon l'horaire réel.
RACE_DAY_TIMELINE = [
    {"when": "3–4 h avant", "title": "Réveil", "text": "Assez tôt pour être réveillé et avoir digéré."},
    {"when": "3 h avant",   "title": "Petit-déjeuner", "text": "Glucides + un peu de protéines, faible en gras. Ce que tu as déjà testé."},
    {"when": "trajet",      "title": "Départ vers la piste", "text": "Pars large pour arriver sans stress."},
    {"when": "90 min+ avant","title": "Arrivée & check-in", "text": "Inscription, plaque, repère le paddock et la grille."},
    {"when": "60–75 min avant","title": "Reconnaissance", "text": "Marche la piste : lignes, 1er virage, sauts."},
    {"when": "45 min avant","title": "Échauffement", "text": "Activation progressive + 2–3 départs d'essai. Active ta routine."},
    {"when": "10 min avant","title": "Dernière prépa", "text": "Hydrate, respiration, point de fixation. Dans ta bulle."},
    {"when": "moto",        "title": "1re manche", "text": "Routine de départ. Une moto à la fois."},
    {"when": "entre les manches","title": "Récup active", "text": "Collation facile + boire. Reste au chaud, reset mental."},
    {"when": "finale",      "title": "Finale", "text": "Réchauffe-toi à nouveau, départ d'essai, focus maximal. Tout donner."},
]


def _prep_phase_key(days: int | None) -> str:
    """Bucket de phase selon les jours restants avant la course."""
    if days is None:
        return "week"
    if days > 7:   return "far"
    if days >= 4:  return "week"
    if days >= 2:  return "taper"
    if days == 1:  return "eve"
    if days == 0:  return "raceday"
    return "after"


def _race_prep_plan(race: dict) -> dict:
    """Construit le plan de prépa d'une course : phase courante, focus du jour,
    piliers phasés (avec la phase courante mise en avant), timeline Jour J."""
    cd = race.get("countdown") or _race_countdown(race)
    days = cd.get("days")
    phase = _prep_phase_key(days)
    pillars = []
    for p in PREP_PILLARS:
        pillars.append({
            "key":     p["key"],
            "label":   p["label"],
            "current": p["phases"].get(phase, []),
            "phases":  [{"key": ph["key"], "label": ph["label"],
                         "items": p["phases"].get(ph["key"], []),
                         "is_current": ph["key"] == phase}
                        for ph in PREP_PHASES if p["phases"].get(ph["key"])],
        })
    return {
        "days":        days,
        "phase":       phase,
        "phase_label": next((ph["label"] for ph in PREP_PHASES if ph["key"] == phase), ""),
        "focus":       PREP_FOCUS.get(phase, PREP_FOCUS["week"]),
        "pillars":     pillars,
        "timeline":    RACE_DAY_TIMELINE,
        "disclaimer":  PREP_DISCLAIMER,
    }


@app.get("/prepa/race/{race_id}")
async def race_detail(request: Request, race_id: str):
    """Page détail d'une course : compte à rebours + plan de préparation."""
    r = races.get(race_id)
    if not r:
        return RedirectResponse("/prepa", status_code=302)
    race = {**r, "countdown": _race_countdown(r),
            "athlete_name": athletes.get(r.get("athlete_id") or "", {}).get("name")}
    plan = _race_prep_plan(race)
    return templates.TemplateResponse(request, "race_detail.html", {
        "active": "prepa",
        "race":   race,
        "plan":   plan,
    })


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


# ── Calibration de la réaction (bip 1 + 1er mouvement, par essai) ─────────────
# Outil d'inspection : pour chaque essai, TOUS les instants de la chaîne de
# mesure (bip 1 audio, bip 1 théorique = grille − 360 ms, grille marquée,
# 1er mouvement pipeline, 1er mouvement raffiné) + drapeaux d'incohérence.
# Objectif : vérifier à l'œil, sur la vidéo, d'où viennent les temps impossibles
# (< ~170 ms = plancher de réaction humaine → deviné, gate mal marqué, ou
# détection trop sensible).
REACT_IMPOSSIBLE_MS = 170.0   # sous ce délai bip→mvt, un humain n'a pas pu réagir


def _reaction_cal_row(jid: str, j: dict) -> dict:
    """Ligne de calibration pour un job (audio = cache seulement, pas de run)."""
    r  = j.get("results") or {}
    rx = r.get("reaction") or {}
    gate_t = r.get("gate_drop_t")
    fps    = r.get("fps") or 30.0
    audio  = j.get("audio_detection") or {}
    bip1_audio = (audio.get("beeps_t") or [None])[0] if audio.get("detected") else None
    bip1_theo  = round(gate_t - UCI_BIP1_TO_GATE_S, 3) if gate_t is not None else None

    fm = _detect_first_move(r)
    move_t   = fm.get("t")
    verified = rx.get("verified")
    if verified:
        # La confirmation coach prime sur la détection logicielle.
        move_t   = verified.get("move_t")
        reaction = round(float(verified["from_gate_ms"]) + GATE_FROM_BIP1_MS)
    else:
        reaction = None
        if fm.get("detected") and fm.get("from_gate_ms") is not None:
            reaction = round(fm["from_gate_ms"] + GATE_FROM_BIP1_MS)

    # Drapeaux
    flags = []
    if verified:
        flags.append(("ok", "confirmé coach"))
    elif rx.get("type") == "false_start":
        flags.append(("bad", "faux départ"))
    elif not fm.get("detected"):
        flags.append(("warn", "mouvement non capté"))
        if fm.get("late_move_ms"):
            flags.append(("warn", f"mvt net à +{fm['late_move_ms']:.0f} ms → grille mal placée ?"))
    elif reaction is not None and reaction < REACT_IMPOSSIBLE_MS:
        flags.append(("bad", f"< {REACT_IMPOSSIBLE_MS:.0f} ms : impossible de réagir si vite"))
    if not verified and bip1_audio is not None and gate_t is not None:
        dev = (gate_t - bip1_audio) * 1000 - GATE_FROM_BIP1_MS
        if abs(dev) > 160:
            flags.append(("warn", f"bip audio vs grille : écart {dev:+.0f} ms vs cadence UCI"))

    return {
        "job_id":        jid,
        "display_name":  _display_name(jid, j),
        "athlete_name":  athletes.get(j.get("athlete_id") or "", {}).get("name"),
        "added_at":      j.get("added_at", ""),
        "fps":           fps,
        "gate_t":        gate_t,
        "bip1_theo":     bip1_theo,
        "bip1_audio":    round(bip1_audio, 3) if bip1_audio is not None else None,
        "audio_cached":  bool(audio),
        "move_pipeline": rx.get("first_move_t"),
        "move_refined":  move_t,
        "move_conf":     fm.get("confidence"),
        "move_signals":  fm.get("signals") or [],
        "reaction_ms":   reaction,
        "false_start":   rx.get("type") == "false_start",
        "detected":      bool(fm.get("detected")),
        "flags":         flags,
    }


@app.get("/reaction_calibration")
async def reaction_calibration_list(request: Request):
    """Table de calibration : tous les essais avec bips, grille, 1er mouvement
    et drapeaux. Audio depuis le cache (le détail lance la détection)."""
    rows = [_reaction_cal_row(jid, j) for jid, j in jobs.items()
            if j.get("status") == "done"]
    rows.sort(key=lambda r: r["added_at"], reverse=True)
    n_impossible = sum(1 for r in rows
                       if r["reaction_ms"] is not None and r["reaction_ms"] < REACT_IMPOSSIBLE_MS)
    n_undetected = sum(1 for r in rows if not r["detected"] and not r["false_start"])
    n_flagged    = sum(1 for r in rows if r["flags"])
    return templates.TemplateResponse(request, "reaction_calibration.html", {
        "active": "settings",
        "rows": rows,
        "n_total": len(rows),
        "n_impossible": n_impossible,
        "n_undetected": n_undetected,
        "n_flagged": n_flagged,
        "floor_ms": REACT_IMPOSSIBLE_MS,
    })


@app.get("/reaction_calibration/{job_id}")
async def reaction_calibration_detail(request: Request, job_id: str):
    """Détail : vidéo + marqueurs sautables (bips, grille, 1ers mouvements)."""
    j = jobs.get(job_id)
    if not j or j.get("status") != "done":
        return RedirectResponse("/reaction_calibration", status_code=302)
    audio = _get_or_run_audio_detection(job_id, j)   # run + cache si nécessaire
    if audio:
        save_jobs(jobs)
    row = _reaction_cal_row(job_id, j)
    r = j.get("results") or {}
    return templates.TemplateResponse(request, "reaction_calibration_detail.html", {
        "active":    "settings",
        "row":       row,
        "audio":     audio,
        "video_url": j.get("video_url"),
        "fps":       r.get("fps") or 30.0,
        "job_id":    job_id,
    })


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
        for p in (PROS_DB, JOBS_DB, ATHLETES_DB, TRACKS_DB, RACES_DB):
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
        for p in (PROS_DB, JOBS_DB, ATHLETES_DB, TRACKS_DB, RACES_DB):
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


# ── Centre du rider pour le mode de cadrage "Recadré sur le rider" ──────────
# On calcule la position médiane du bassin (moyenne L_hip + R_hip) sur une
# fenêtre courte autour du gate drop, exprimée en % des dimensions de la
# vidéo. Sert d'object-position côté CSS pour zoomer/croper intelligemment
# sur le rider dans un cadre uniforme, indépendamment du ratio source.
def _get_video_dims(video_path: Path) -> tuple[int, int] | None:
    """Lit (width, height) d'une vidéo via OpenCV — rapide, juste les
    métadonnées du conteneur."""
    if not video_path.exists():
        return None
    try:
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return None


def _compute_rider_center_pct(csv_path: Path, gate_t: float,
                              video_path: Path | None = None,
                              window_s: float = 0.4) -> dict:
    """Position médiane du bassin sur [gate-0.2s, gate+0.2s], exprimée en %
    des dimensions vidéo. Utilisée comme object-position CSS pour le mode
    de cadrage "Recadré sur le rider".

    Robuste aux trous : prend la médiane (pas la moyenne) → ignore les
    outliers. Si la fenêtre autour du gate est vide, fallback sur toute la
    vidéo. Si les dimensions vidéo sont introuvables, fallback sur le max
    observé des keypoints comme proxy. Clamp final à [5%, 95%] pour éviter
    qu'un keypoint aberrant en bord d'image fasse sortir le rider du cadre.
    """
    default = {"x_pct": 50.0, "y_pct": 50.0}
    if not csv_path.exists():
        return default
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return default
    if "time" not in df.columns or len(df) == 0:
        return default
    sub = df[(df["time"] >= gate_t - window_s/2) &
             (df["time"] <= gate_t + window_s/2)]
    if len(sub) < 3:
        sub = df  # fallback toute la vidéo si pas assez près du gate
    for c in ("L_hip_x", "R_hip_x", "L_hip_y", "R_hip_y"):
        if c not in sub.columns:
            return default
    center_x = ((sub["L_hip_x"] + sub["R_hip_x"]) / 2.0).dropna()
    center_y = ((sub["L_hip_y"] + sub["R_hip_y"]) / 2.0).dropna()
    if len(center_x) == 0 or len(center_y) == 0:
        return default
    px_x = float(center_x.median())
    px_y = float(center_y.median())
    # Dimensions de la vidéo
    dims = _get_video_dims(video_path) if video_path else None
    if not dims:
        # Fallback : proxy via les max observés des keypoints (sous-estime
        # un peu mais OK pour un centrage approximatif)
        all_x = pd.concat([sub[c] for c in sub.columns if c.endswith('_x')]).dropna()
        all_y = pd.concat([sub[c] for c in sub.columns if c.endswith('_y')]).dropna()
        if len(all_x) == 0 or len(all_y) == 0:
            return default
        dims = (max(int(all_x.max() * 1.05), 1), max(int(all_y.max() * 1.05), 1))
    w, h = dims
    if w <= 0 or h <= 0:
        return default
    return {
        "x_pct": round(min(95.0, max(5.0, px_x / w * 100.0)), 1),
        "y_pct": round(min(95.0, max(5.0, px_y / h * 100.0)), 1),
    }


def _get_or_compute_rider_center(entity_kind: str, entity_id: str,
                                  entity: dict) -> dict:
    """Lit le centre du rider mis en cache dans le job/pro ; calcule et
    persiste si absent. `entity_kind` ∈ {"job", "pro"}."""
    if entity_kind == "job":
        results = entity.get("results") or {}
        cached  = results.get("rider_center_pct")
        if cached and "x_pct" in cached and "y_pct" in cached:
            return cached
        csv_path  = OUTPUT_DIR / (results.get("files", {}).get("landmarks_csv") or "")
        annotated = results.get("files", {}).get("annotated_video")
        video_path = (OUTPUT_DIR / annotated) if annotated else None
        gate_t    = float(results.get("gate_drop_t", 0.0))
        center = _compute_rider_center_pct(csv_path, gate_t, video_path)
        results["rider_center_pct"] = center
        entity["results"] = results
        if entity.get("athlete_id"):
            save_jobs(jobs)
        return center
    elif entity_kind == "pro":
        cached = entity.get("rider_center_pct")
        if cached and "x_pct" in cached and "y_pct" in cached:
            return cached
        csv_path  = OUTPUT_DIR / (entity.get("landmarks_csv") or "")
        video_path = OUTPUT_DIR / (entity.get("video_file") or "")
        gate_t    = float(entity.get("gate_drop_t", 0.0))
        center = _compute_rider_center_pct(csv_path, gate_t, video_path)
        entity["rider_center_pct"] = center
        save_pros(pros)
        return center
    return {"x_pct": 50.0, "y_pct": 50.0}


def _burst_diagnose(burst: dict, perso: dict | None = None) -> dict:
    """Moteur de règles → verdict + observations + cue coaching.

    PRINCIPE : tout le texte destiné au coach/athlète est en langage SIMPLE
    (pas de "ω", "proximale-distale", "°/s" dans les phrases principales).
    Les chiffres techniques sont regroupés à part dans la clé `tech` pour
    être affichés sous un repli "Détails techniques".

    Vocabulaire coach utilisé :
      - "ouverture des hanches"   = vitesse d'extension hanche (ω hanche)
      - "coup de pédale"          = vitesse d'extension genou (ω genou)
      - "finition cheville"       = vitesse d'extension cheville
      - "ordre de poussée"        = séquence proximale-distale (hanches→jambe)
      - "la grille"               = gate drop

    Logique (inchangée, pas de ML) :
      1. Cas non-significatifs / mesure incomplète.
      2. Ordre de poussée (séquence).
      3. Comparaison à l'historique perso (forme du jour).
      4. Patterns techniques (délai hanches→jambe, poussée faible).
      5. UN conseil d'entraînement basé sur le défaut dominant.

    Sources (citées dans `tech`, pas dans le texte coach) :
      Gross 2017, Cowell 2020, Grigg 2020, Mahieu.

    Retour : {verdict, verdict_label, headline, observations:[{type,text}],
              cue:{text, why}, tech:{lines:[...], sources:[...]}}.
    """
    obs: list[dict] = []
    tech_lines: list[str] = []   # chiffres bruts pour le repli "Détails"
    hip   = burst.get("hip")
    knee  = burst.get("knee")
    ankle = burst.get("ankle")

    # ── Cas dégénérés ────────────────────────────────────────────────
    if not hip or not knee:
        return {
            "verdict": "incomplete",
            "verdict_label": "Analyse impossible",
            "headline": "La vidéo n'est pas assez nette pour mesurer le départ.",
            "observations": [{"type": "warn",
                "text": "On n'arrive pas à suivre la hanche ou le genou sur cet essai. "
                        "Vérifie que le rider est bien visible de profil et refilme."}],
            "cue": None,
            "tech": {"lines": ["Keypoints hanche ou genou manquants dans le CSV."],
                     "sources": []},
        }

    hip_omega   = hip.get("omega_max") or 0
    knee_omega  = knee.get("omega_max") or 0
    hip_tpeak   = hip.get("t_peak")
    knee_tpeak  = knee.get("t_peak")

    # Essai non-significatif (warmup, drill statique, gate raté…)
    if hip_omega < 50 and knee_omega < 50:
        return {
            "verdict": "non_significant",
            "verdict_label": "Pas un vrai départ",
            "headline": "Pas de poussée franche sur cet essai — sûrement un échauffement, "
                        "un exercice à l'arrêt ou un départ raté.",
            "observations": [
                {"type": "info", "text": "Aucune extension explosive détectée. "
                                          "Cet essai n'est pas compté dans tes statistiques."},
            ],
            "cue": None,
            "tech": {"lines": [
                f"Ouverture hanches : {hip_omega:.0f}°/s · coup de pédale : {knee_omega:.0f}°/s "
                "(les deux sous le seuil de 50°/s).",
            ], "sources": []},
        }

    # Chiffres techniques de base (toujours dans le repli)
    tech_lines.append(f"Ouverture des hanches : {hip_omega:.0f}°/s"
                      + (f" (pic à {hip_tpeak*1000:.0f} ms après la grille)" if hip_tpeak is not None else ""))
    tech_lines.append(f"Coup de pédale (genou) : {knee_omega:.0f}°/s"
                      + (f" (pic à {knee_tpeak*1000:.0f} ms)" if knee_tpeak is not None else ""))
    if ankle and ankle.get("omega_max") is not None:
        a_t = ankle.get("t_peak")
        tech_lines.append(f"Finition cheville : {ankle['omega_max']:.0f}°/s"
                          + (f" (pic à {a_t*1000:.0f} ms)" if a_t is not None else ""))
    tech_lines.append(f"Cadence vidéo : {burst.get('fps_est', 30):.0f} fps.")

    # Mesure en bord de fenêtre → fiabilité douteuse
    if burst.get("has_edge_warning"):
        edges = [name for name, j in (("hanches", hip), ("jambe", knee), ("cheville", ankle))
                 if j and j.get("edge_peak")]
        obs.append({
            "type": "warn",
            "text": "La vidéo coupe trop tôt ou trop tard : on n'est pas sûr d'avoir "
                    "capté toute la poussée. Filme en gardant ~1 seconde avant le 1er bip "
                    "et ~1,5 seconde après que la grille tombe.",
        })
        tech_lines.append("⚠ Pic en bord de fenêtre : " + ", ".join(edges)
                          + " — mesure possiblement incomplète.")

    # ── Ordre de poussée (séquence) ──────────────────────────────────
    ci = burst.get("ci_verdict")
    if ci == "proximal_distal":
        obs.append({"type": "good",
            "text": "Bon ordre de poussée : tu ouvres tes hanches en premier, "
                    "puis tu donnes le coup de pédale. C'est exactement ce qu'on veut."})
    elif ci == "simultaneous":
        obs.append({"type": "info",
            "text": "Hanches et jambe poussent quasiment en même temps. "
                    "Ce n'est pas un défaut, mais à cette vitesse de caméra c'est "
                    "difficile d'être plus précis."})
    elif ci == "inverted":
        obs.append({"type": "bad",
            "text": "Mauvais ordre de poussée : tu donnes le coup de pédale AVANT "
                    "d'ouvrir tes hanches. C'est moins efficace — la puissance des "
                    "hanches est perdue."})
    if burst.get("ci_reason"):
        tech_lines.append("Ordre des pics : " + burst["ci_reason"])

    # ── Comparaison perso (forme du jour, pas technique) ─────────────
    has_perso_hip   = perso and perso.get("omega", {}).get("hip")  \
                      and perso["omega"]["hip"].get("n", 0) >= 3
    has_perso_knee  = perso and perso.get("omega", {}).get("knee") \
                      and perso["omega"]["knee"].get("n", 0) >= 3

    # Comparaison vs perso = chute/hausse de FORME, pas défaut technique.
    # Types dip/boost/record → n'inflatent pas le verdict technique.
    # `label_simple` = "ouverture des hanches" / "coup de pédale".
    def _cmp_obs(label_simple, current, ref):
        if not ref or ref.get("n", 0) < 3: return None
        mean = ref["mean"]; best = ref["best"]; n = ref["n"]
        delta_pct = (current - mean) / mean * 100 if mean > 0 else 0
        tech_lines.append(
            f"{label_simple.capitalize()} : {current:.0f}°/s "
            f"(ta moyenne {mean:.0f}, ton record {best:.0f}, sur {n} départs).")
        if current > best * 1.02:
            return {"type": "record",
                "text": f"Nouveau record perso sur ton {label_simple} ! "
                        "Le plus puissant qu'on t'ait mesuré."}
        if current >= best * 0.97:
            return {"type": "boost",
                "text": f"Ton {label_simple} est au niveau de ton meilleur départ. "
                        "Très bon."}
        if delta_pct >= 10:
            return {"type": "boost",
                "text": f"Tu as poussé plus fort que d'habitude sur ton {label_simple} "
                        f"(environ {delta_pct:.0f}% au-dessus de ta moyenne)."}
        if delta_pct <= -20:
            return {"type": "dip",
                "text": f"Tu as nettement moins poussé que d'habitude sur ton "
                        f"{label_simple} (~{abs(delta_pct):.0f}% sous ton niveau). "
                        "Fatigue ou échauffement pas complet ?"}
        if delta_pct <= -10:
            return {"type": "dip",
                "text": f"Ton {label_simple} était un peu en-dessous de ton "
                        f"habitude (~{abs(delta_pct):.0f}% sous ta moyenne)."}
        return None  # zone normale → on ne dit rien

    if has_perso_hip:
        o = _cmp_obs("ouverture des hanches", hip_omega, perso["omega"]["hip"])
        if o: obs.append(o)
    if has_perso_knee:
        o = _cmp_obs("coup de pédale", knee_omega, perso["omega"]["knee"])
        if o: obs.append(o)

    # ── Pattern : délai hanches→jambe ───────────────────────────────
    delay_hip_knee_ms = None
    if hip_tpeak is not None and knee_tpeak is not None:
        delay_hip_knee_ms = (knee_tpeak - hip_tpeak) * 1000
        tech_lines.append(f"Délai hanches → coup de pédale : {delay_hip_knee_ms:.0f} ms.")
        if delay_hip_knee_ms > 450:
            obs.append({"type": "warn",
                "text": "Trop de temps entre l'ouverture de tes hanches et ton coup "
                        "de pédale. Ta jambe pousse trop tard, tu perds de la vitesse."})

    # ── Pattern d'amplitude (si pas de perso, pour ne pas répéter) ──
    if not has_perso_hip and hip_omega < 100:
        obs.append({"type": "warn",
            "text": "Tes hanches s'ouvrent mollement au départ. C'est le moteur "
                    "principal de la poussée — il faut plus d'explosivité là."})

    # ── Choix du verdict global ─────────────────────────────────────
    # Le verdict reflète la TECHNIQUE (ordre de poussée, amplitude).
    # Les baisses/hausses vs perso (dip/boost/record) sont de la FORME du
    # jour → elles colorient la phrase mais ne font pas basculer le verdict.
    n_bad     = sum(1 for o in obs if o["type"] == "bad")
    n_warn    = sum(1 for o in obs if o["type"] == "warn")
    n_good    = sum(1 for o in obs if o["type"] == "good")
    n_record  = sum(1 for o in obs if o["type"] == "record")
    n_dip     = sum(1 for o in obs if o["type"] == "dip")

    if n_bad >= 1:
        verdict, label = "bad", "Départ à retravailler"
        headline = "Il y a un défaut technique important à corriger."
    elif n_warn >= 2:
        verdict, label = "warn", "Départ correct, à peaufiner"
        headline = "Bonne base, mais quelques points à travailler."
    elif n_good >= 1 and n_warn == 0:
        if n_record >= 1:
            verdict, label = "good", "Excellent départ"
            headline = "Technique propre ET meilleur que ton habitude. Bravo."
        elif n_dip >= 1:
            verdict, label = "good", "Bon départ"
            headline = "Ta technique est propre, mais tu étais moins explosif que d'habitude aujourd'hui."
        else:
            verdict, label = "good", "Bon départ"
            headline = "Exécution propre. Continue sur la régularité et l'explosivité."
    elif n_warn == 1 and n_good >= 1:
        verdict, label = "ok", "Départ correct"
        headline = "Ça tient la route, juste un point à surveiller."
    else:
        verdict, label = "ok", "Départ correct"
        headline = "Rien d'alarmant, mais pas de point fort marquant non plus."

    # ── Choix du conseil d'entraînement ─────────────────────────────
    # UN conseil basé sur le défaut dominant. Priorité :
    # 1) mauvais ordre de poussée  2) jambe trop tardive
    # 3) hanches molles  4) forme en baisse  5) bon → consolider
    cue = None
    if ci == "inverted":
        cue = {
            "text": "Reste BAS dans ta position de set, garde le buste penché vers "
                    "l'avant jusqu'au bip 3. Pense « hanches d'abord, jambe ensuite ».",
            "why": "Quand on se redresse trop tôt en set, les hanches n'ont plus de "
                   "course pour pousser, donc c'est la jambe qui part en premier. "
                   "Garder le buste bas garde la puissance des hanches disponible."
        }
    elif delay_hip_knee_ms is not None and delay_hip_knee_ms > 450:
        cue = {
            "text": "Travaille l'explosivité de jambe : squat jumps légers, 5×5, "
                    "en cherchant à te détendre le plus VITE possible.",
            "why": "Tes hanches tirent bien mais ta jambe ne suit pas avec la même "
                   "vitesse. Le but est de raccourcir le délai entre les deux."
        }
    elif hip_omega < 100 and not has_perso_hip:
        cue = {
            "text": "Renforce et réveille tes hanches : hip thrusts et box jumps "
                    "explosifs. Sur le vélo, exagère la position basse en set "
                    "(épaules devant le guidon, fesses reculées).",
            "why": "L'ouverture des hanches est ce qui lance tout le départ. Plus "
                   "elles sont explosives, plus le reste suit."
        }
    elif n_dip >= 2:
        cue = {
            "text": "C'est une question de forme du jour, pas de technique. Refais "
                    "un échauffement complet (active hanches et cuisses) puis "
                    "réessaie. Si c'est encore mou, regarde ta récup (sommeil, "
                    "charge des derniers jours).",
            "why": "Ta technique est bonne, mais tu pousses moins fort que d'habitude "
                   "sur deux mouvements à la fois — c'est typique d'un manque "
                   "d'énergie ou d'échauffement, pas d'un problème de geste."
        }
    elif verdict == "good":
        cue = {
            "text": "Enchaîne 5 à 8 départs aujourd'hui et cherche à reproduire le "
                    "même geste à chaque fois. La régularité, c'est ce qui gagne des "
                    "courses.",
            "why": "Tu as une bonne mécanique. À ton niveau, c'est la constance d'un "
                   "départ à l'autre qui fait la différence."
        }
    else:
        cue = {
            "text": "Continue ton travail habituel. Pour gagner en explosivité, "
                    "alterne séances de force (squat jumps, sprints départ arrêté) "
                    "et départs sur rampe (5-8 essais, bien récupérés entre chaque).",
            "why": "Pas de défaut technique précis à corriger ici. La marge de "
                   "progression est sur la puissance et la régularité."
        }

    return {
        "verdict": verdict,
        "verdict_label": label,
        "headline": headline,
        "observations": obs,
        "cue": cue,
        "tech": {
            "lines": tech_lines,
            "sources": [
                "Ordre de poussée hanches→jambe→cheville : Gross 2017, Cowell 2020.",
                "Repères de vitesse d'extension : Gross 2017 (élites 400-700°/s au genou).",
                "Position de set basse / pull-back : Grigg 2020.",
            ],
        },
    }


# ── Module "Consignes posturales par phase" ─────────────────────────────────
# Génère des consignes PRÉCISES de position (« recule tes hanches », « déplie
# la jambe », « avance ton bassin ») en mesurant la géométrie du corps à des
# moments clés (position de set, push 1) et en la comparant à des repères
# biomécaniques.
#
# CHOIX DE FIABILITÉ : on privilégie les mesures CAMÉRA-INVARIANTES, c.-à-d.
# qui comparent des positions du corps ENTRE ELLES dans la même image (angle
# d'un membre, alignement de segments, position du bassin vs pied avant), et
# PAS des valeurs absolues qui dépendraient de l'angle de caméra. Hypothèse :
# filmé À PEU PRÈS DE PROFIL (ce que fait l'utilisateur).
#
# SEUILS : valeurs de PREMIÈRE PASSE, documentées et à calibrer ensemble sur
# de vraies vidéos. Sources : Grigg 2020 (back position, set), Mahieu
# (alignement épaules-bassin-chevilles pour la transmission de puissance).
POSTURE_VERSION = "1.0"

# Position du bassin vs pied avant en set, normalisée par la longueur du
# tronc (épaule→hanche). >0 = bassin en AVANT du pied ; <0 = en arrière.
# Back position (Grigg) = bassin au-dessus / légèrement en arrière du pied.
SET_HIP_FOOT_TOO_FORWARD =  0.25   # au-delà → « recule tes hanches »
SET_HIP_FOOT_VERY_BACK   = -0.95   # en-deçà → bassin très reculé (info)

# Angle du genou de la jambe avant en set (180° = jambe tendue, 90° = pliée).
# Trop tendue = pas de course pour pousser ; trop pliée = accroupi.
SET_KNEE_TOO_STRAIGHT = 152.0      # au-delà → « plie ta jambe pour la charger »
SET_KNEE_TOO_BENT     =  88.0      # en-deçà → « déplie un peu ta jambe avant »

# Inclinaison du buste vs verticale en set (0° = droit, + = penché en avant).
SET_TRUNK_TOO_UPRIGHT = 22.0       # en-deçà → « penche ton buste vers l'avant »
SET_TRUNK_TOO_LOW     = 72.0       # au-delà → « tu es trop penché »

# Alignement épaules-bassin-chevilles (angle au bassin, 180° = aligné).
ALIGN_MIN = 150.0                  # en-deçà → « aligne épaules-bassin-chevilles »

# Au push 1 : extension de la jambe au pic de poussée (180° = tendue).
PUSH_KNEE_INCOMPLETE = 150.0       # en-deçà → « pousse jusqu'au bout »


def _posture_at_frame(row, side: str, direction: int) -> dict | None:
    """Mesure la géométrie posturale sur UNE frame (une ligne du CSV).
    `side` = pied avant (L/R), `direction` = +1 si le rider regarde vers la
    droite de l'image, -1 sinon. Retourne None si keypoints manquants."""
    def _pt(part):
        x = row.get(f"{side}_{part}_x", np.nan)
        y = row.get(f"{side}_{part}_y", np.nan)
        return (float(x), float(y))
    sh = _pt("shoulder"); hi = _pt("hip"); kn = _pt("knee"); an = _pt("ankle")
    pts = [sh, hi, kn, an]
    if any(np.isnan(p[0]) or np.isnan(p[1]) for p in pts):
        return None

    # Longueur de référence (tronc) pour normaliser les distances
    L = float(np.hypot(sh[0] - hi[0], sh[1] - hi[1]))
    if L < 1e-3:
        return None

    # Angle du genou (hanche-genou-cheville) : 180 = tendu
    knee_angle = float(calculate_angle(hi, kn, an))
    # Alignement épaules-bassin-chevilles (angle au bassin) : 180 = aligné
    align_angle = float(calculate_angle(sh, hi, an))
    # Inclinaison du buste vs verticale, signée selon la direction du rider
    dx = sh[0] - hi[0]; dy = sh[1] - hi[1]
    trunk_lean = float(np.degrees(np.arctan2(direction * dx, -dy)))
    # Position horizontale du bassin vs pied avant, normalisée (+ = en avant)
    hip_foot_offset = (hi[0] - an[0]) * direction / L

    return {
        "knee_angle":       round(knee_angle, 1),
        "align_angle":      round(align_angle, 1),
        "trunk_lean":       round(trunk_lean, 1),
        "hip_foot_offset":  round(hip_foot_offset, 3),
    }


def _frame_direction(row) -> int:
    """+1 si le rider regarde vers la droite de l'image, -1 sinon."""
    nose_x = row.get("nose_x", np.nan)
    L_hi_x = row.get("L_hip_x", np.nan)
    R_hi_x = row.get("R_hip_x", np.nan)
    if not (np.isnan(nose_x) or np.isnan(L_hi_x) or np.isnan(R_hi_x)):
        return 1 if nose_x > (L_hi_x + R_hi_x) / 2 else -1
    return 1


def _compute_posture(csv_path: Path, gate_t: float, side: str,
                     phases: list | None = None) -> dict | None:
    """Mesure la posture en SET (gate−0.05s) et au PUSH 1 (pic d'extension),
    génère les consignes. Retourne None si CSV illisible."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "time" not in df.columns or len(df) == 0:
        return None

    # ── SET : frame la plus proche de gate − 0.05s ──────────────────
    t_set = gate_t - 0.05
    set_idx = int((df["time"] - t_set).abs().idxmin())
    set_row = df.loc[set_idx]
    direction = _frame_direction(set_row)
    set_geo = _posture_at_frame(set_row, side, direction)

    # ── PUSH 1 : frame la plus tendue dans la phase Push 1 ──────────
    push_geo = None
    if phases:
        push_phase = next((p for p in phases if p.get("name") == "Push 1"), None)
        if push_phase:
            t0 = push_phase.get("start_t", gate_t)
            t1 = push_phase.get("end_t", gate_t + 0.5)
            sub = df[(df["time"] >= t0) & (df["time"] <= t1)]
            best_knee = -1.0
            for _, r in sub.iterrows():
                g = _posture_at_frame(r, side, _frame_direction(r))
                if g and g["knee_angle"] > best_knee:
                    best_knee = g["knee_angle"]
                    push_geo = g

    cues = _posture_cues(set_geo, push_geo)
    return {
        "_version":   POSTURE_VERSION,
        "set":        set_geo,
        "push1":      push_geo,
        "cues":       cues,
    }


def _posture_cues(set_geo: dict | None, push_geo: dict | None) -> dict:
    """Transforme les mesures géométriques en consignes posturales par phase.
    Chaque consigne : {type ✓/⚠, text (langage coach), why}."""
    set_cues: list[dict] = []
    push_cues: list[dict] = []

    if set_geo:
        off = set_geo["hip_foot_offset"]
        if off > SET_HIP_FOOT_TOO_FORWARD:
            set_cues.append({"type": "fix",
                "text": "Recule tes hanches. Ton bassin est trop en avant : ramène-le "
                        "vers l'arrière du vélo pour charger ton appui.",
                "why": "Bassin en avant = poids déjà engagé, tu perds le ressort. "
                       "Les meilleurs partent avec le bassin reculé (back position)."})
        elif off < SET_HIP_FOOT_VERY_BACK:
            set_cues.append({"type": "info",
                "text": "Tes hanches sont très reculées — OK si tu es stable, mais "
                        "vérifie que tu ne tires pas trop sur les bras.",
                "why": "Un bassin très en arrière peut surcharger les bras et "
                       "retarder le transfert vers l'avant."})
        else:
            set_cues.append({"type": "ok",
                "text": "Bonne position de bassin : reculé comme il faut pour charger "
                        "l'appui.",
                "why": "Bassin au-dessus / légèrement en arrière du pied avant = "
                       "back position, la plus efficace."})

        kn = set_geo["knee_angle"]
        if kn > SET_KNEE_TOO_STRAIGHT:
            set_cues.append({"type": "fix",
                "text": "Plie plus ta jambe avant. Elle est trop tendue en set : tu "
                        "n'as pas de course pour pousser.",
                "why": "Une jambe déjà tendue ne peut plus se détendre — c'est la "
                       "détente qui crée la puissance au départ."})
        elif kn < SET_KNEE_TOO_BENT:
            set_cues.append({"type": "fix",
                "text": "Déplie un peu ta jambe avant. Tu es trop accroupi, ça bloque "
                        "ta poussée.",
                "why": "Trop plié = position basse difficile à relancer vite. Cherche "
                       "un angle de charge intermédiaire."})
        else:
            set_cues.append({"type": "ok",
                "text": "Bon angle de jambe avant : chargée juste comme il faut.",
                "why": "Jambe ni trop tendue ni trop pliée → prête à se détendre vite."})

        tr = set_geo["trunk_lean"]
        if tr < SET_TRUNK_TOO_UPRIGHT:
            set_cues.append({"type": "fix",
                "text": "Penche ton buste plus vers l'avant. Tu es trop droit en set.",
                "why": "Un buste trop droit remonte ton centre de gravité et réduit "
                       "le transfert de poids vers l'avant."})
        elif tr > SET_TRUNK_TOO_LOW:
            set_cues.append({"type": "info",
                "text": "Tu es très penché vers l'avant — assure-toi de garder le "
                        "contrôle du guidon.",
                "why": "Très penché peut aider l'explosivité mais demande un bon "
                       "gainage pour ne pas s'écrouler."})
        else:
            set_cues.append({"type": "ok",
                "text": "Bonne inclinaison du buste : penché vers l'avant comme il faut.",
                "why": "Buste penché = centre de gravité bas, prêt à transférer vers "
                       "l'avant."})
        # NB : on NE juge PAS l'alignement épaules-bassin-chevilles en set —
        # en set le corps est volontairement plié/ramassé pour charger. Cet
        # alignement (principe Mahieu) se juge au PIC DE POUSSÉE, pas ici.

    if push_geo:
        kn = push_geo["knee_angle"]
        if kn < PUSH_KNEE_INCOMPLETE:
            push_cues.append({"type": "fix",
                "text": "Pousse jusqu'au bout sur ton premier coup de pédale. Ta jambe "
                        "ne se déplie pas complètement.",
                "why": "Une extension incomplète laisse de la puissance dans la jambe. "
                       "Va chercher l'extension complète à chaque coup."})
        else:
            push_cues.append({"type": "ok",
                "text": "Tu pousses bien jusqu'au bout sur ton premier coup de pédale.",
                "why": "Extension complète = tu utilises toute la course de ta jambe."})

        al = push_geo["align_angle"]
        if al < ALIGN_MIN:
            push_cues.append({"type": "fix",
                "text": "Garde ton corps aligné quand tu pousses : avance ton bassin "
                        "en même temps que tu tires sur le guidon.",
                "why": "Si le bassin reste en arrière pendant la poussée, tu casses la "
                       "chaîne de transmission et tu cabres au lieu d'avancer."})

    return {"set": set_cues, "push1": push_cues}


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


@app.get("/posture/{job_id}")
async def posture(job_id: str):
    """Consignes posturales précises par phase (set, push 1) basées sur des
    repères biomécaniques. Voir _compute_posture pour la méthode."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    results  = job["results"]
    csv_path = OUTPUT_DIR / results["files"]["landmarks_csv"]
    side     = results.get("front_foot") or "L"
    gate_t   = float(results.get("gate_drop_t", 0.0))
    phases   = results.get("phases")
    post = _compute_posture(csv_path, gate_t, side, phases)
    if post is None:
        return JSONResponse({"error": "Lecture des positions impossible."},
                            status_code=422)
    return post


# ── Classification de la position de set (Grigg 2020, variable #1) ────────────
# Grigg 2020 identifie la position de SET comme la variable la plus prédictive
# du départ, avec trois archétypes : « back » (poids/bassin reculé, la plus
# performante en moyenne), « upright » (buste vertical), « angled » (buste
# penché vers l'avant). On classe à partir de la géométrie CORPS pure (pas de
# pièce de vélo) mesurée à l'instant du set : inclinaison du buste (trunk_lean)
# et position du bassin vs pied avant (hip_foot_offset). Honnêteté : Grigg
# travaille sur n=10 élites masculins → « back » est une tendance, pas une loi ;
# les seuils sont calibrables.
SETPOS_BACK_OFFSET   = -0.30   # hip_foot_offset ≤ ce seuil = bassin reculé (back)
SETPOS_UPRIGHT_TRUNK =  25.0   # trunk_lean < ce seuil (et pas back) = upright


def _compute_set_position(csv_path: Path, gate_t: float, side: str) -> dict | None:
    """Classe la position de set en back / upright / angled depuis la géométrie
    du corps à l'instant du set (gate − 0.05s)."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "time" not in df.columns or len(df) == 0:
        return None

    t_set = gate_t - 0.05
    idx = int((df["time"] - t_set).abs().idxmin())
    row = df.loc[idx]
    geo = _posture_at_frame(row, side, _frame_direction(row))
    if geo is None:
        return None

    offset = geo["hip_foot_offset"]
    trunk  = geo["trunk_lean"]

    if offset <= SETPOS_BACK_OFFSET:
        set_type, vclass = "back", "ok"
        label = "Position « back » (bassin reculé)"
        headline = ("Tu te places avec le bassin reculé et le poids en arrière — "
                    "c'est la position de set la plus performante en moyenne chez "
                    "les élites.")
        cue = "Garde cette base : c'est un bon point d'appui pour charger et exploser."
    elif trunk < SETPOS_UPRIGHT_TRUNK:
        set_type, vclass = "upright", "warn"
        label = "Position « upright » (buste vertical)"
        headline = ("Ton buste est plutôt vertical en set. C'est une position "
                    "valable, mais elle remonte ton centre de gravité.")
        cue = ("À tester : recule un peu le bassin et baisse le buste pour aller "
               "vers une position « back », associée aux meilleurs départs.")
    else:
        set_type, vclass = "angled", "warn"
        label = "Position « angled » (buste penché)"
        headline = ("Tu pars buste penché vers l'avant. Position dynamique, mais "
                    "le poids est déjà engagé devant.")
        cue = ("À tester : ramène le bassin vers l'arrière pour charger l'appui "
               "avant d'exploser — la position « back » de Grigg.")

    return {
        "type":      set_type,
        "label":     label,
        "vclass":    vclass,
        "headline":  headline,
        "cue":       cue,
        "trunk_lean":      trunk,
        "hip_foot_offset": offset,
        "source":    "Grigg 2020",
        "caveat":    "Tendance issue d'un petit échantillon d'élites (n=10) ; seuils calibrables.",
    }


@app.get("/set_position/{job_id}")
async def set_position(job_id: str):
    """Classe la position de set (back / upright / angled) — variable #1 du
    départ selon Grigg 2020. Voir _compute_set_position."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    results  = job["results"]
    csv_path = OUTPUT_DIR / results["files"]["landmarks_csv"]
    side     = results.get("front_foot") or "L"
    gate_t   = float(results.get("gate_drop_t", 0.0))
    sp = _compute_set_position(csv_path, gate_t, side)
    if sp is None:
        return JSONResponse({"error": "Lecture de la position de set impossible."},
                            status_code=422)
    return sp


# ── Scorecard du départ ──────────────────────────────────────────────────────
# Synthétise un départ en 4 notes (0–100) + une note globale. Toute la logique
# est un moteur de règles transparent (pas de ML), avec des seuils CALIBRABLES
# regroupés ici et étiquetés par source. Honnêteté scientifique : les bandes de
# réaction n'ont pas de benchmark universel publié pour l'amateur → marquées
# « indicatif ». Les bandes d'explosivité s'appuient sur Gross 2017.
SCORECARD_VERSION = "2.0"

# Pondérations des 5 dimensions (renormalisées sur celles disponibles).
# Réaction et contre-mouvement en tête : timing au gate (déterminant en course)
# et countermovement (déterminant biomécanique n°1, Gross 2017). Voir SCIENCE.md.
SCORE_WEIGHTS = {
    "reaction":        0.24,
    "countermovement": 0.22,
    "explosivity":     0.20,
    "sequence":        0.17,
    "posture":         0.17,
}

# Notes par régime de contre-mouvement (déterminant n°1).
CMV_SCORES = {"early": 100, "late": 58, "absent": 30}

# Réaction = délai entre le 1er bip et le 1er mouvement (définition coach :
# « dès qu'il y a un peu de mouvement »). On l'ANCRE sur la grille : la cadence
# UCI étant fixe (la grille tombe 360 ms après le 1er bip, cf. SCIENCE.md §1),
# le 1er bip = grille − 360 ms. On ne se sert PAS du bip détecté à l'audio comme
# ancre du chiffre : il attrape souvent un écho ou la voix « watch the gate »,
# ce qui gonflait la réaction à 800–1700 ms (incohérent). En recalant sur la
# cadence, réaction = from_gate_ms + 360, toujours cohérente et plausible.
#
# IMPORTANT (rappel terrain) : le 1er bip arrive après un délai ALÉATOIRE
# (0,1–2,7 s) → on ne peut PAS l'anticiper, on y RÉAGIT. Seule la grille est
# anticipable, car fixe à +360 ms une fois le bip entendu. Donc un mouvement
# sous ~170 ms après le bip = sous le plancher de réaction humaine = le rider a
# deviné le départ (coup de poker), pas réagi.
#
# Repères (réaction = bip1 → 1er mouvement, en ms ; la grille tombe à 360 ms) :
#   < 230  → parti bien avant la grille : anticipation risquée / coup de poker
#   230–300 → anticipation franche (parti 60–130 ms avant la grille) : bon
#   300–385 → IDÉAL : parti avec la grille
#   385–470 → un peu réactif (parti après la grille)
#   > 470   → réagi, pas anticipé (lent)
# Plancher de réaction humaine ~170 ms (auditif) : sous ce seuil ce n'est plus
# une réaction mais une anticipation pure du rythme. Bandes indicatives/calibrables.
GATE_FROM_BIP1_MS  = 360.0    # cadence UCI : grille = bip1 + 360 ms
REACT_HUMAN_FLOOR  = 170.0    # plancher réaction auditive humaine (ms)
REACT_GAMBLE_MAX   = 230.0    # en-deçà = coup de poker
REACT_EARLY_MAX    = 300.0    # 230–300 = anticipation franche
REACT_OPTIMAL_MAX  = 385.0    # 300–385 = idéal (autour de la grille à 360)
REACT_REACTIVE_MAX = 470.0    # 385–470 = un peu réactif ; au-delà = lent

# ── Détecteur de 1er mouvement multi-articulaire ─────────────────────────────
# Remplace le check binaire genou-seul (8°) par une détection fine sur 3 signaux
# (genou, hanche, inclinaison du tronc) calculés depuis le CSV DÉJÀ stocké
# (aucun re-passage YOLO). Baseline robuste (médiane/MAD sur la fenêtre AVANT le
# 1er bip), seuil adaptatif par signal, onset = 1re excursion soutenue 2 frames.
# « Dès qu'il y a un peu de mouvement » = le plus précoce des 3 signaux.
FIRSTMOVE_VERSION = "1.0"
FM_Z_THRESH  = 6.0                     # z-score robuste (× MAD) minimal
FM_MIN_DEG   = {"knee": 5.0, "hip": 5.0, "trunk": 4.0}   # plancher absolu (°)
FM_NOISE_FLOOR_DEG = 0.8               # bruit minimal supposé (°) si MAD ~ 0
FM_SUSTAIN   = 2                       # frames consécutives au-dessus du seuil
FM_BASE_LO, FM_BASE_HI = -1.5, -0.6    # baseline (s, rel. grille) — avant bip 1
FM_SCAN_LO,  FM_SCAN_HI = -0.55, 0.8   # scan (s, rel. grille) — bip 1 → post
FM_LATE_HI   = 2.5                     # scan étendu pour l'indice « gate mal marqué »

_FIRSTMOVE_CACHE: dict = {}            # {csv_path: (mtime, result)}


def _detect_first_move(results: dict) -> dict:
    """Détection raffinée du 1er mouvement autour de la grille.

    Retour : {detected: bool, from_gate_ms, t, confidence (0–1),
              signals: [noms], late_move_ms (si non détecté mais gros mouvement
              plus tard — indice de gate mal marqué), reason}.
    Mise en cache par (chemin CSV, mtime)."""
    out = {"detected": False, "from_gate_ms": None, "t": None,
           "confidence": 0.0, "signals": [], "late_move_ms": None, "reason": ""}
    try:
        files = results.get("files") or {}
        csv_path = OUTPUT_DIR / files["landmarks_csv"]
        if not csv_path.exists():
            out["reason"] = "csv_missing"
            return out
        mtime = csv_path.stat().st_mtime
        cached = _FIRSTMOVE_CACHE.get(str(csv_path))
        if cached and cached[0] == mtime:
            return cached[1]

        df = pd.read_csv(csv_path)
        if "time" not in df.columns or len(df) < 10:
            out["reason"] = "csv_unreadable"
            return out
        gate_t = float(results.get("gate_drop_t", 0.0))
        side = results.get("front_foot") or "R"

        def col(part, axis):
            c = f"{side}_{part}_{axis}"
            return df[c].to_numpy(dtype=float) if c in df.columns else np.full(len(df), np.nan)

        t = df["time"].to_numpy(dtype=float)
        sh_x, sh_y = col("shoulder", "x"), col("shoulder", "y")
        hi_x, hi_y = col("hip", "x"),      col("hip", "y")
        kn_x, kn_y = col("knee", "x"),     col("knee", "y")
        an_x, an_y = col("ankle", "x"),    col("ankle", "y")

        def angle(p1x, p1y, p2x, p2y, p3x, p3y):
            v1x, v1y = p1x - p2x, p1y - p2y
            v2x, v2y = p3x - p2x, p3y - p2y
            dot = v1x * v2x + v1y * v2y
            n1 = np.sqrt(v1x**2 + v1y**2); n2 = np.sqrt(v2x**2 + v2y**2)
            with np.errstate(invalid="ignore", divide="ignore"):
                cos = np.clip(dot / (n1 * n2), -1.0, 1.0)
            return np.degrees(np.arccos(cos))

        signals = {
            "knee":  angle(hi_x, hi_y, kn_x, kn_y, an_x, an_y),
            "hip":   angle(sh_x, sh_y, hi_x, hi_y, kn_x, kn_y),
            # Inclinaison du tronc vs verticale — capte le rock-back/lean même
            # quand les jambes ne bougent pas encore.
            "trunk": np.degrees(np.arctan2(np.abs(sh_x - hi_x), np.abs(hi_y - sh_y) + 1e-9)),
        }

        base_m = (t >= gate_t + FM_BASE_LO) & (t <= gate_t + FM_BASE_HI)
        scan_m = (t >= gate_t + FM_SCAN_LO) & (t <= gate_t + FM_SCAN_HI)
        if base_m.sum() < 5 or scan_m.sum() < 3:
            out["reason"] = "window_too_short"
            _FIRSTMOVE_CACHE[str(csv_path)] = (mtime, out)
            return out

        def find_onset(x, name, mask):
            """1er instant (dans mask) où |x−baseline| dépasse le seuil adaptatif
            pendant FM_SUSTAIN frames valides consécutives. None sinon."""
            base = x[base_m]
            base = base[~np.isnan(base)]
            if len(base) < 5:
                return None, None
            med = float(np.median(base))
            mad = float(np.median(np.abs(base - med)))
            sigma = max(1.4826 * mad, FM_NOISE_FLOOR_DEG)
            thresh = max(FM_Z_THRESH * sigma, FM_MIN_DEG[name])
            idxs = np.where(mask)[0]
            run = 0; first = None
            for i in idxs:
                if np.isnan(x[i]):
                    run = 0; first = None
                    continue
                if abs(x[i] - med) > thresh:
                    if first is None:
                        first = i
                    run += 1
                    if run >= FM_SUSTAIN:
                        return float(t[first]), float(abs(x[first] - med) / thresh)
                else:
                    run = 0; first = None
            return None, None

        onsets = {}
        for name, x in signals.items():
            ot, strength = find_onset(x, name, scan_m)
            if ot is not None:
                onsets[name] = (ot, strength)

        if onsets:
            first_t = min(ot for ot, _ in onsets.values())
            # Confiance : nb de signaux qui confirment dans les 150 ms + force.
            confirming = [n for n, (ot, _) in onsets.items() if ot - first_t <= 0.15]
            conf = min(1.0, 0.4 + 0.25 * len(confirming))
            out.update({
                "detected": True,
                "t": round(first_t, 3),
                "from_gate_ms": round((first_t - gate_t) * 1000),
                "confidence": round(conf, 2),
                "signals": confirming,
                "reason": "ok",
            })
        else:
            # Rien dans la fenêtre : cherche un gros mouvement plus tard → indice
            # que la grille est peut-être mal marquée sur cette vidéo.
            late_m = (t > gate_t + FM_SCAN_HI) & (t <= gate_t + FM_LATE_HI)
            for name, x in signals.items():
                ot, _ = find_onset(x, name, late_m)
                if ot is not None:
                    prev = out["late_move_ms"]
                    ms = round((ot - gate_t) * 1000)
                    out["late_move_ms"] = ms if prev is None else min(prev, ms)
            out["reason"] = "no_movement_in_window"

        _FIRSTMOVE_CACHE[str(csv_path)] = (mtime, out)
        return out
    except Exception:
        out["reason"] = "error"
        return out


def _reaction_reliable(results: dict) -> bool:
    """Compat : la réaction est mesurable si le détecteur raffiné a trouvé un
    vrai mouvement autour de la grille."""
    return bool(_detect_first_move(results).get("detected"))


def _analyze_reaction(results: dict) -> dict | None:
    """Réaction = 1er bip → 1er mouvement (ms), recalée sur la cadence UCI.

    Retourne {score, verdict, label, value_text, reaction_ms, from_gate_ms,
    regime, detail, audio_note, source} ou None si non mesurable.
    `regime` ∈ false_start / undetected / gamble / early / optimal / reactive / late.
    """
    rx = results.get("reaction") or {}
    g = rx.get("from_gate_ms")
    b_audio = rx.get("from_bip1_ms")   # bip détecté à l'audio (souvent bruité)
    verified_info = rx.get("verified")   # confirmation coach (grille + 1er mvt)
    is_verified = bool(verified_info)
    skipped = bool(rx.get("verify_skipped"))

    if rx.get("type") == "false_start" and not is_verified:
        return {
            "score": 0, "verdict": "bad", "label": "Réaction", "regime": "false_start",
            "value_text": "Faux départ", "reaction_ms": None, "from_gate_ms": g,
            "detail": "Premier mouvement AVANT le 1er bip — départ annulé en course.",
            "verified": False, "skipped": skipped,
            "audio_note": None, "source": "spec UCI"}

    refine_note = None
    if is_verified:
        # Le coach a confirmé les deux instants à l'œil : c'est LA référence.
        g = float(verified_info["from_gate_ms"])
    else:
        if g is None:
            return None
        # Détection raffinée du 1er mouvement (multi-articulaire, CSV stocké) —
        # sert d'ESTIMATION et pré-place les marqueurs de vérification.
        fm = _detect_first_move(results)
        if fm.get("detected"):
            g_refined = float(fm["from_gate_ms"])
            if abs(g_refined - float(g)) > 60:
                direction = "plus tôt" if g_refined < float(g) else "plus tard"
                refine_note = (
                    f"Détection affinée (genou+hanche+tronc) : 1er mouvement repéré "
                    f"{abs(g_refined - float(g)):.0f} ms {direction} que l'analyse initiale.")
            g = g_refined
        elif fm.get("reason") in ("no_movement_in_window", "window_too_short"):
            # Honnêteté : rien de franc autour de la grille → on n'invente pas.
            detail = ("La pose n'a pas capté de mouvement franc autour de la grille — "
                      "souvent parce que le rider est trop petit dans l'image ou que la "
                      "caméra le suit. Confirme la grille et le 1er mouvement à l'œil "
                      "pour obtenir un temps fiable.")
            if fm.get("late_move_ms"):
                detail += (f" Indice : un mouvement net existe {fm['late_move_ms']:.0f} ms "
                           f"après la grille marquée — la grille est peut-être mal placée "
                           f"sur cette vidéo.")
            return {
                "score": None, "verdict": "none", "label": "Réaction", "regime": "undetected",
                "value_text": "Mouvement non capté", "reaction_ms": None,
                "from_gate_ms": round(float(g)),
                "detail": detail,
                "late_move_ms": fm.get("late_move_ms"),
                "verified": False, "skipped": skipped,
                "audio_note": None, "source": "non mesurable"}
        # (csv manquant / erreur de lecture → on garde la valeur du pipeline)

    g = float(g)

    # Réaction recalée : 1er bip = grille − 360 ms (cadence UCI fixe).
    reaction = g + GATE_FROM_BIP1_MS

    # Note de transparence si l'audio donnait un bip très différent (≥ 160 ms
    # d'écart avec la cadence) → on explique qu'on s'appuie sur la grille.
    audio_note = refine_note
    if b_audio is not None and abs(float(b_audio) - reaction) > 160:
        note2 = (
            f"La détection audio plaçait le 1er bip à {b_audio:.0f} ms du mouvement, "
            f"mais sa cadence était bruitée (écho / voix). Réaction recalée sur la "
            f"grille + cadence UCI (360 ms).")
        audio_note = f"{refine_note} {note2}" if refine_note else note2

    if reaction < REACT_GAMBLE_MAX:
        regime = "gamble"
        score = _lin_score(reaction, REACT_HUMAN_FLOOR - 60, REACT_GAMBLE_MAX, 52.0, 84.0)
        detail = ("Parti sous le plancher de réaction humaine (~170 ms). Le 1er bip "
                  "tombe au hasard : tu ne peux pas le devancer, donc partir si tôt "
                  "c'est un coup de poker — risque de faux départ ou de déséquilibre.")
    elif reaction < REACT_EARLY_MAX:
        regime = "early"
        score = _lin_score(reaction, REACT_GAMBLE_MAX, REACT_EARLY_MAX, 86.0, 99.0)
        detail = ("Réaction vive au 1er bip : tu déclenches vite et tu pars un peu "
                  "avant la grille. Solide.")
    elif reaction <= REACT_OPTIMAL_MAX:
        regime = "optimal"
        score = 100.0
        detail = ("Timing idéal : tu réagis au 1er bip puis tu pars AVEC la grille "
                  "(elle tombe 360 ms après le bip). Pile au bon moment.")
    elif reaction <= REACT_REACTIVE_MAX:
        regime = "reactive"
        score = _lin_score(reaction, REACT_OPTIMAL_MAX, REACT_REACTIVE_MAX, 95.0, 70.0)
        detail = ("Tu pars juste après la grille : un peu tard. Une fois le 1er bip "
                  "entendu, la grille est à 360 ms — déclenche un cheveu plus tôt.")
    else:
        regime = "late"
        score = _lin_score(reaction, REACT_REACTIVE_MAX, REACT_REACTIVE_MAX + 350, 70.0, 20.0)
        detail = ("Réaction lente : tu pars nettement après la grille. Le gate tombe "
                  "360 ms après le 1er bip — gros gain possible en déclenchant plus vite.")

    reaction_i = round(reaction)
    # Contexte : où ça tombe par rapport à la grille (360 ms).
    if g < -8:    gate_ctx = f"{abs(g):.0f} ms avant la grille"
    elif g > 8:   gate_ctx = f"{g:.0f} ms après la grille"
    else:         gate_ctx = "pile avec la grille"

    if is_verified:
        # Confirmé coach → évalué : note, stats, progression.
        return {
            "score": round(score), "verdict": _score_to_verdict(score),
            "label": "Réaction", "regime": regime,
            "reaction_ms": reaction_i, "from_gate_ms": round(g),
            "value_text": f"{reaction_i} ms — {gate_ctx}",
            "gate_context": gate_ctx, "detail": detail,
            "verified": True, "verified_at": verified_info.get("verified_at"),
            "skipped": False,
            "audio_note": None, "source": "confirmé coach · cadence UCI"}

    # Estimation logicielle : affichée pour information, JAMAIS notée ni comptée.
    # Le coach doit confirmer la grille + le 1er mouvement pour évaluer.
    return {
        "score": None, "verdict": "none",
        "label": "Réaction", "regime": regime,
        "reaction_ms": reaction_i, "from_gate_ms": round(g),
        "value_text": f"~{reaction_i} ms — {gate_ctx}",
        "gate_context": gate_ctx, "detail": detail,
        "verified": False, "skipped": skipped,
        "audio_note": audio_note, "source": "estimation · à confirmer"}

# Explosivité — pic ω genou (°/s). ATTENTION méthodologique : Gross 2017 mesure
# 400–700°/s en capture MARQUEURS haute cadence. Nos ω viennent de pose
# markerless à ~30 fps, systématiquement plus basses (lissage + différentiation).
# On ne peut donc PAS noter sur l'échelle absolue de Gross sans tout mettre à 0.
# Bande calibrée pour le régime markerless 30 fps (indicative, à affiner par
# club via les essais accumulés). Gross reste la référence conceptuelle.
EXPLO_KNEE_FLOOR  = 110.0   # plancher → 0
EXPLO_KNEE_TARGET = 360.0   # haut de gamme markerless observé → 100
# Mélange genou/hanche (le genou domine l'explosivité balistique).
EXPLO_KNEE_WEIGHT = 0.65
EXPLO_HIP_FLOOR   = 90.0
EXPLO_HIP_TARGET  = 300.0

# Notes par verdict de séquence proximale-distale.
SEQ_SCORES = {
    "proximal_distal": 100,
    "simultaneous":     65,
    "partial":          45,
    "inverted":         25,
}


def _lin_score(x: float, x0: float, x1: float,
               y0: float = 0.0, y1: float = 100.0) -> float:
    """Interpolation linéaire bornée de x∈[x0,x1] vers [y0,y1]."""
    if x1 == x0:
        return y1
    u = (x - x0) / (x1 - x0)
    u = max(0.0, min(1.0, u))
    return y0 + (y1 - y0) * u


def _score_to_verdict(score: float | None) -> str:
    """0–100 → ok / warn / bad (les 3 couleurs de jugement du DS)."""
    if score is None:
        return "none"
    if score >= 70:
        return "ok"
    if score >= 45:
        return "warn"
    return "bad"


def _score_reaction(results: dict) -> dict | None:
    """Dimension réaction pour la scorecard — délègue à _analyze_reaction et ne
    garde que les clés attendues par la scorecard. None si non notable
    (non mesurable / non détecté) → la dimension est traitée comme « non mesurée »."""
    a = _analyze_reaction(results)
    if a is None or a.get("score") is None:
        return None
    out = {k: a[k] for k in ("score", "verdict", "label", "value_text", "detail", "source")}
    if a.get("audio_note"):
        out["caveat"] = "Réaction recalée sur la grille (audio du bip bruité)."
    return out


def _score_explosivity(burst: dict | None) -> dict | None:
    """Note d'explosivité depuis le pic ω genou (+ hanche). Gross 2017."""
    if not burst:
        return None
    knee = burst.get("knee") or {}
    hip  = burst.get("hip") or {}
    knee_w = knee.get("omega_max")
    hip_w  = hip.get("omega_max")
    if knee_w is None and hip_w is None:
        return None
    parts, weight_sum = 0.0, 0.0
    if knee_w is not None:
        parts += EXPLO_KNEE_WEIGHT * _lin_score(knee_w, EXPLO_KNEE_FLOOR, EXPLO_KNEE_TARGET)
        weight_sum += EXPLO_KNEE_WEIGHT
    if hip_w is not None:
        parts += (1 - EXPLO_KNEE_WEIGHT) * _lin_score(hip_w, EXPLO_HIP_FLOOR, EXPLO_HIP_TARGET)
        weight_sum += (1 - EXPLO_KNEE_WEIGHT)
    score = parts / weight_sum if weight_sum else 0.0
    kv = f"{knee_w:.0f}°/s" if knee_w is not None else "—"
    out = {"score": round(score), "verdict": _score_to_verdict(score),
           "label": "Explosivité",
           "value_text": f"genou {kv}",
           "detail": "Vitesse d'extension du genou. Échelle markerless (les °/s vidéo "
                     "sont plus bas que les 400–700°/s mesurés au labo par Gross 2017).",
           "source": "markerless · indic."}
    if burst.get("has_edge_warning"):
        out["caveat"] = "Pic en bord de fenêtre — mesure possiblement incomplète."
    return out


def _score_sequence(burst: dict | None) -> dict | None:
    """Note de séquence proximale-distale (ordre hanche→genou→cheville) +
    délais réels entre les pics d'extension, en ms."""
    if not burst:
        return None
    ci = burst.get("ci_verdict")
    if ci not in SEQ_SCORES:
        return None
    score = SEQ_SCORES[ci]
    labels = {"proximal_distal": "hanche → genou → cheville",
              "simultaneous": "poussée simultanée",
              "partial": "séquence partielle",
              "inverted": "ordre inversé"}
    # Délais réels entre pics (t_peak en s → ms).
    def tp(j):
        d = burst.get(j)
        return d.get("t_peak") if d else None
    hip_t, knee_t, ankle_t = tp("hip"), tp("knee"), tp("ankle")
    delays = {}
    if hip_t is not None and knee_t is not None:
        delays["hip_knee_ms"] = round((knee_t - hip_t) * 1000)
    if knee_t is not None and ankle_t is not None:
        delays["knee_ankle_ms"] = round((ankle_t - knee_t) * 1000)
    detail = "L'ordre d'allumage hanche→genou→cheville signe les départs puissants."
    if delays:
        parts = []
        if "hip_knee_ms" in delays:
            parts.append(f"hanche→genou {delays['hip_knee_ms']:+d} ms")
        if "knee_ankle_ms" in delays:
            parts.append(f"genou→cheville {delays['knee_ankle_ms']:+d} ms")
        detail = "Délais entre pics d'extension : " + " · ".join(parts) + "."
    return {"score": score, "verdict": _score_to_verdict(score),
            "label": "Séquence",
            "value_text": labels.get(ci, ci),
            "delays": delays,
            "detail": detail,
            "source": "Gross 2017"}


def _score_countermovement(cmv: dict | None) -> dict | None:
    """Note de contre-mouvement (déterminant n°1, Gross 2017) pour la scorecard."""
    if not cmv:
        return None
    verdict_key = cmv.get("verdict")  # early / late / absent
    if verdict_key not in CMV_SCORES:
        return None
    score = CMV_SCORES[verdict_key]
    knee = cmv.get("knee") or {}
    depth = knee.get("depth_deg")
    if cmv.get("verdict") == "early":
        vt = "load précoce" + (f" · {depth:.0f}° genou" if depth is not None else "")
    elif cmv.get("verdict") == "late":
        vt = "load tardif" + (f" · {depth:.0f}° genou" if depth is not None else "")
    else:
        vt = "pas de load franc"
    return {"score": score, "verdict": _score_to_verdict(score),
            "label": "Contre-mouvement", "value_text": vt,
            "detail": "Charger les jambes juste avant la grille — déterminant n°1 du départ.",
            "source": "Gross 2017",
            "caveat": cmv.get("caveat")}


def _score_posture(posture: dict | None) -> dict | None:
    """Note de posture depuis le ratio de consignes OK / à corriger."""
    if not posture:
        return None
    cues = (posture.get("cues") or {})
    flat = (cues.get("set") or []) + (cues.get("push1") or [])
    ok  = sum(1 for c in flat if c.get("type") == "ok")
    fix = sum(1 for c in flat if c.get("type") == "fix")
    if ok + fix == 0:
        return None
    score = 100.0 * ok / (ok + fix)
    return {"score": round(score), "verdict": _score_to_verdict(score),
            "label": "Posture",
            "value_text": f"{ok}/{ok + fix} repères OK",
            "detail": "Position en set et extension au push 1 vs repères élites.",
            "source": "Grigg 2018"}


def _compute_scorecard(job_id: str, job: dict) -> dict:
    """Assemble la scorecard : 5 dimensions notées + note globale pondérée +
    note lettre. Les dimensions non mesurables sont marquées et exclues du
    calcul global (renormalisation des poids)."""
    results = job.get("results") or {}
    burst = _get_or_compute_burst(job_id, job)

    posture = None
    cmv = None
    try:
        csv_path = OUTPUT_DIR / results["files"]["landmarks_csv"]
        side   = results.get("front_foot") or "L"
        gate_t = float(results.get("gate_drop_t", 0.0))
        posture = _compute_posture(csv_path, gate_t, side, results.get("phases"))
        cmv     = _compute_countermovement(csv_path, gate_t, side)
    except Exception:
        posture = None

    dims = {
        "reaction":        _score_reaction(results),
        "countermovement": _score_countermovement(cmv),
        "explosivity":     _score_explosivity(burst),
        "sequence":        _score_sequence(burst),
        "posture":         _score_posture(posture),
    }

    # Note globale : moyenne pondérée sur les dimensions disponibles.
    num, den = 0.0, 0.0
    for key, dim in dims.items():
        if dim is not None and dim.get("score") is not None:
            w = SCORE_WEIGHTS.get(key, 0.0)
            num += w * dim["score"]
            den += w
    overall = round(num / den) if den > 0 else None

    if overall is None:
        grade, grade_label = "—", "Mesure insuffisante"
    elif overall >= 85:
        grade, grade_label = "A", "Départ d'élite"
    elif overall >= 70:
        grade, grade_label = "B", "Solide"
    elif overall >= 55:
        grade, grade_label = "C", "À structurer"
    elif overall >= 40:
        grade, grade_label = "D", "Beaucoup de marge"
    else:
        grade, grade_label = "E", "Départ à reconstruire"

    return {
        "_version":   SCORECARD_VERSION,
        "overall":    overall,
        "grade":      grade,
        "grade_label": grade_label,
        "verdict":    _score_to_verdict(overall),
        "dimensions": dims,
        "weights":    SCORE_WEIGHTS,
    }


@app.get("/scorecard/{job_id}")
async def scorecard(job_id: str):
    """Scorecard synthétique d'un départ : réaction, explosivité, séquence,
    posture → 4 notes + note globale. Voir _compute_scorecard."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    return _compute_scorecard(job_id, job)


@app.get("/reaction/{job_id}")
async def reaction(job_id: str):
    """Analyse détaillée du temps de réaction (modèle « anticipation » ancré sur
    la grille, cadence UCI). Voir _analyze_reaction et SCIENCE.md §2."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    a = _analyze_reaction(job["results"])
    if a is None:
        return JSONResponse({"error": "Réaction non mesurable (audio manquant)."},
                            status_code=422)
    return a


@app.get("/report/{job_id}")
async def report(request: Request, job_id: str):
    """Rapport imprimable d'un départ (print-to-PDF depuis Safari iPad).
    Tout est assemblé côté serveur → page statique, aucune dépendance JS,
    impression fiable. Réutilise scorecard + posture + diagnostic explosivité."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return templates.TemplateResponse(request, "index.html",
                                          {"error": "Rapport indisponible : job introuvable."})

    results = job["results"]
    sc = _compute_scorecard(job_id, job)

    # Diagnostic explosivité (langage coach) + posture + contre-mouvement.
    burst = _get_or_compute_burst(job_id, job)
    diag = _burst_diagnose(burst) if burst else None
    posture = None
    cmv = None
    setpos = None
    try:
        csv_path = OUTPUT_DIR / results["files"]["landmarks_csv"]
        side   = results.get("front_foot") or "L"
        gate_t = float(results.get("gate_drop_t", 0.0))
        posture = _compute_posture(csv_path, gate_t, side, results.get("phases"))
        cmv     = _compute_countermovement(csv_path, gate_t, side)
        setpos  = _compute_set_position(csv_path, gate_t, side)
    except Exception:
        posture = None

    # Les 2 consignes prioritaires (à corriger) toutes phases confondues.
    top_fixes: list[dict] = []
    if posture:
        cues = posture.get("cues") or {}
        for phase_key in ("set", "push1"):
            for c in (cues.get(phase_key) or []):
                if c.get("type") == "fix":
                    top_fixes.append({**c, "phase": phase_key})
    if diag and diag.get("cue"):
        top_fixes.append({"type": "fix", "text": diag["cue"].get("text", ""),
                          "why": diag["cue"].get("why", ""), "phase": "explosivité"})

    aid = job.get("athlete_id")
    athlete = athletes.get(aid) if aid else None

    return templates.TemplateResponse(request, "report.html", {
        "job_id":       job_id,
        "results":      results,
        "scorecard":    sc,
        "diagnosis":    diag,
        "posture":      posture,
        "countermovement": cmv,
        "set_position": setpos,
        "top_fixes":    top_fixes[:3],
        "athlete":      athlete,
        "display_name": _display_name(job_id, job),
        "track_name":   tracks.get(job.get("track_id"), {}).get("name", "") if job.get("track_id") else "",
        "generated_at": datetime.now().strftime("%d/%m/%Y à %H:%M"),
    })


def _trend(values: list[float]) -> dict | None:
    """Tendance d'une série chronologique (ancien→récent) par régression
    linéaire simple. Retourne {slope_per_trial, direction}. None si < 3 points."""
    n = len(values)
    if n < 3:
        return None
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((xs[i] - mx) * (values[i] - my) for i in range(n)) / denom
    direction = "flat"
    if slope > 0.5:
        direction = "up"
    elif slope < -0.5:
        direction = "down"
    return {"slope": round(slope, 2), "direction": direction}


@app.get("/athletes/{athlete_id}/progression")
async def athlete_progression(athlete_id: str):
    """Séries temporelles de progression : par essai (ancien→récent) la
    réaction, la note globale, le pic ω genou et la séquence. Plus des
    agrégats : records, tendance, régularité (écart-type des 5 derniers).
    Sert le dashboard de progression de la fiche athlète."""
    if athlete_id not in athletes:
        return JSONResponse({"error": "Athlète introuvable."}, status_code=404)

    # Jobs de l'athlète, du plus ancien au plus récent, hors exclus.
    aj = [(jid, j) for jid, j in jobs.items()
          if j.get("status") == "done"
          and j.get("athlete_id") == athlete_id
          and not j.get("excluded_from_stats")]
    aj.sort(key=lambda x: x[1].get("added_at", ""))

    series: list[dict] = []
    for jid, j in aj:
        r = (j.get("results") or {}).get("reaction") or {}
        reaction_ms = _reaction_ms(j.get("results") or {})  # recalé + fiabilité (SCIENCE.md §2)
        sc = _compute_scorecard(jid, j)
        burst = _get_or_compute_burst(jid, j)
        knee_omega = None
        if burst and burst.get("knee"):
            knee_omega = burst["knee"].get("omega_max")
        series.append({
            "job_id":      jid,
            "label":       _display_name(jid, j),
            "date":        j.get("added_at", ""),
            "reaction_ms": reaction_ms,
            "overall":     sc.get("overall"),
            "grade":       sc.get("grade"),
            "knee_omega":  round(knee_omega, 0) if knee_omega is not None else None,
            "false_start": r.get("type") == "false_start",
        })

    # Agrégats
    reactions = [s["reaction_ms"] for s in series if s["reaction_ms"] is not None]
    overalls  = [s["overall"] for s in series if s["overall"] is not None]
    last5     = reactions[-5:]
    consistency = round(float(np.std(last5)), 1) if len(last5) >= 2 else None

    return {
        "athlete_id":   athlete_id,
        "athlete_name": athletes[athlete_id].get("name", ""),
        "n":            len(series),
        "series":       series,
        "best_reaction": min(reactions) if reactions else None,
        "best_overall":  max(overalls) if overalls else None,
        "consistency_ms": consistency,
        "trend_reaction": _trend(reactions),
        "trend_overall":  _trend([float(o) for o in overalls]),
    }


# ── Détection du countermovement (Gross 2017, déterminant #1) ─────────────────
# Le contre-mouvement précoce — charger en fléchissant juste avant/au moment de
# la chute de la grille, puis exploser — est le déterminant n°1 de la perf au
# gate selon Gross 2017. On le détecte sur la série d'angle du genou (et de la
# hanche) autour du gate : une flexion supplémentaire (dip sous la position de
# set) suivie de l'extension. Honnêteté : à 30 fps markerless, la profondeur en
# degrés est indicative (bruit de pose + lissage), le TIMING est plus robuste.
CMV_PRE  = 0.55     # fenêtre avant le gate (s)
CMV_POST = 0.50     # fenêtre après le gate (s)
CMV_MIN_DEPTH = 10.0    # flexion mini (°) sous le set pour parler de contre-mouv.
CMV_EARLY_T   = 0.06    # début du load ≤ ce seuil (s, rel. gate) = précoce
CMV_RECOVER   = 0.30    # part de la profondeur qui doit être ré-étendue après le load


def _angle_series_for(df, side: str, joint: str):
    """Série d'angle (°) frame par frame pour 'knee' (hanche-genou-cheville) ou
    'hip' (épaule-hanche-genou), côté `side`. Retourne (t_array, angle_array)."""
    def col(part, axis):
        c = f"{side}_{part}_{axis}"
        return df[c].to_numpy(dtype=float) if c in df.columns else np.full(len(df), np.nan)
    t = df["time"].to_numpy(dtype=float)
    if joint == "knee":
        p1 = (col("hip", "x"), col("hip", "y"))
        p2 = (col("knee", "x"), col("knee", "y"))
        p3 = (col("ankle", "x"), col("ankle", "y"))
    else:  # hip
        p1 = (col("shoulder", "x"), col("shoulder", "y"))
        p2 = (col("hip", "x"), col("hip", "y"))
        p3 = (col("knee", "x"), col("knee", "y"))
    v1x, v1y = p1[0] - p2[0], p1[1] - p2[1]
    v2x, v2y = p3[0] - p2[0], p3[1] - p2[1]
    dot = v1x * v2x + v1y * v2y
    n1 = np.sqrt(v1x * v1x + v1y * v1y)
    n2 = np.sqrt(v2x * v2x + v2y * v2y)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return t, np.degrees(np.arccos(cos))


def _countermovement_one(t_arr, ang, gate_t: float) -> dict | None:
    """Détecte le contre-mouvement (load → extension) sur une articulation.

    Méthode : le « load » est le minimum d'angle (flexion maximale) dans la
    fenêtre autour du gate. On mesure sa profondeur depuis le sommet qui le
    précède (position chargée vs set), l'instant où la descente commence
    (timing du load) et on confirme qu'une ré-extension suit (sinon ce n'est
    pas un contre-mouvement mais un affaissement).

    Retour {present, depth_deg, t_min, t_load_start, set_angle, min_angle,
    recover_deg} ou None."""
    mask = (t_arr >= gate_t - CMV_PRE) & (t_arr <= gate_t + CMV_POST)
    t = t_arr[mask]
    a = ang[mask]
    valid = ~np.isnan(a)
    if valid.sum() < 8:
        return None
    t = t[valid]; a = a[valid]
    if len(a) >= 5:
        win = min(7, len(a) if len(a) % 2 == 1 else len(a) - 1)
        if win >= 5:
            try:
                a = savgol_filter(a, window_length=win, polyorder=2, mode='interp')
            except Exception:
                pass
    min_idx   = int(np.argmin(a))                       # bas du load
    min_angle = float(a[min_idx])
    # Sommet AVANT le load (position de set / chargée la plus haute).
    set_angle = float(np.max(a[:min_idx + 1]))
    depth     = set_angle - min_angle
    # Ré-extension après le load (pour distinguer load réel d'un affaissement).
    recover   = float(np.max(a[min_idx:]) - min_angle) if min_idx < len(a) - 1 else 0.0
    # Début de la descente : 1re frame où l'angle passe sous (set − 30% depth).
    thresh = set_angle - 0.30 * depth
    t_load_start = float(t[min_idx] - gate_t)
    for i in range(min_idx + 1):
        if a[i] <= thresh:
            t_load_start = float(t[i] - gate_t)
            break
    present = depth >= CMV_MIN_DEPTH and recover >= CMV_RECOVER * depth
    return {
        "present":      bool(present),
        "depth_deg":    round(depth, 1),
        "t_min":        round(float(t[min_idx] - gate_t), 3),
        "t_load_start": round(t_load_start, 3),
        "set_angle":    round(set_angle, 1),
        "min_angle":    round(min_angle, 1),
        "recover_deg":  round(recover, 1),
    }


def _compute_countermovement(csv_path: Path, gate_t: float, side: str) -> dict | None:
    """Analyse le contre-mouvement (genou + hanche) et produit un verdict coach."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "time" not in df.columns or len(df) == 0:
        return None

    t_arr, knee_ang = _angle_series_for(df, side, "knee")
    _,     hip_ang  = _angle_series_for(df, side, "hip")
    knee = _countermovement_one(t_arr, knee_ang, gate_t)
    hip  = _countermovement_one(t_arr, hip_ang,  gate_t)
    if knee is None and hip is None:
        return None

    # Verdict : on se fie d'abord au genou (signal le plus net), hanche en appui.
    primary = knee or hip
    present = (knee and knee["present"]) or (hip and hip["present"])
    t_load_start = primary["t_load_start"]
    early = present and t_load_start <= CMV_EARLY_T

    if not present:
        verdict, vclass = "absent", "warn"
        headline = ("Pas de vrai contre-mouvement détecté : tu sembles partir "
                    "depuis ta position de set sans recharger.")
        cue = ("Travaille le « load » : juste avant la grille, fléchis légèrement "
               "pour armer tes jambes, puis explose. C'est le geste n°1 des "
               "meilleurs départs.")
    elif early:
        verdict, vclass = "early", "ok"
        headline = ("Bon contre-mouvement précoce : tu charges tes jambes au bon "
                    "moment avant d'exploser.")
        cue = ("Continue — un load juste avant/au moment de la grille te donne le "
               "ressort. Garde ce timing.")
    else:
        verdict, vclass = "late", "warn"
        headline = ("Tu charges tes jambes, mais un peu tard (après la grille) : "
                    "tu perds une partie du ressort.")
        cue = ("Anticipe ton load : amorce la flexion juste AVANT que la grille "
               "tombe pour être déjà en train d'exploser quand elle part.")

    return {
        "knee":     knee,
        "hip":      hip,
        "verdict":  verdict,
        "vclass":   vclass,
        "headline": headline,
        "cue":      cue,
        "source":   "Gross 2017",
        "caveat":   "Profondeur indicative (markerless 30 fps) ; le timing est plus fiable.",
    }


@app.get("/countermovement/{job_id}")
async def countermovement(job_id: str):
    """Détection du contre-mouvement (charge avant extension) — déterminant #1
    du départ selon Gross 2017. Voir _compute_countermovement."""
    job = get_job_or_recover(job_id)
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "Job introuvable ou non terminé."},
                            status_code=404)
    results = job["results"]
    csv_path = OUTPUT_DIR / results["files"]["landmarks_csv"]
    side   = results.get("front_foot") or "L"
    gate_t = float(results.get("gate_drop_t", 0.0))
    cm = _compute_countermovement(csv_path, gate_t, side)
    if cm is None:
        return JSONResponse({"error": "Données insuffisantes autour du gate."},
                            status_code=422)
    return cm


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

    # Centre du rider courant pour le mode de cadrage "Recadré sur le rider"
    rider_center = _get_or_compute_rider_center("job", job_id, job)

    # Pros : on enrichit chaque pro avec son rider_center_pct calculé (et caché)
    pros_done = []
    for p in pros.values():
        if p.get("status") != "done":
            continue
        pc = _get_or_compute_rider_center("pro", p.get("id", ""), p)
        # Copie superficielle + injection du centre pour le template
        pp = dict(p)
        pp["rider_center_pct"] = pc
        pros_done.append(pp)

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
        # Centre rider pour les jobs aussi (utilisé si on les choisit comme réf)
        entry["rider_center_pct"] = _get_or_compute_rider_center("job", jid, j)
        if cur_athlete_id and aid == cur_athlete_id:
            same_athlete_jobs.append(entry)
        else:
            other_jobs.append(entry)
    same_athlete_jobs.sort(key=lambda x: x.get("added_at", ""), reverse=True)
    other_jobs.sort(key=lambda x: x.get("added_at", ""), reverse=True)

    return templates.TemplateResponse(request, "compare.html", {
        "job_id":             job_id,
        "rider":              job["results"],
        "rider_center":       rider_center,
        "rider_display_name": _display_name(job_id, job),
        "pros_list":          pros_done,
        "same_athlete_jobs":  same_athlete_jobs,
        "other_jobs":         other_jobs,
        "current_athlete":    athletes.get(cur_athlete_id, {}).get("name") if cur_athlete_id else None,
    })

