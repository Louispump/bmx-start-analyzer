"""
Pré-construit le cache /test/mahieu pour toutes les vidéos d'uploads/.

Usage : python mahieu_warm.py
  → écrit output/mahieu_db.json + images dans output/mahieu_debug/
  → /test/mahieu s'ouvre instantanément ensuite
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "agg"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mahieu import analyze_video, render_debug_frame  # noqa: E402

REPO_ROOT = Path("/Users/louis-edouarddube/Projets/bmx-start-analyzer")
UP        = REPO_ROOT / "uploads"
OUT       = REPO_ROOT / "output"
CACHE     = OUT / "mahieu_db.json"
DEBUG     = OUT / "mahieu_debug"


def main():
    DEBUG.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text())
        except Exception:
            cache = {}

    videos = sorted([p for p in UP.iterdir()
                     if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")])
    print(f"Pre-warming Mahieu cache for {len(videos)} videos...")

    for i, p in enumerate(videos, 1):
        key = f"{p.name}::{int(p.stat().st_mtime)}"
        if key in cache:
            print(f"  [{i}/{len(videos)}] {p.name}: cached", flush=True)
            continue
        t0 = time.time()
        res = analyze_video(p)
        debug_url = None
        if res.get("detected") and res.get("best_frame"):
            bf  = res["best_frame"]
            png = DEBUG / f"{p.stem}.jpg"
            if render_debug_frame(p, bf["frame"], bf["metric"], bf["side"], png):
                debug_url = f"/output/mahieu_debug/{png.name}"
        cache[key] = {"res": res, "debug_url": debug_url}
        CACHE.write_text(json.dumps(cache, indent=2, default=str))
        status = (f"OK ({res['deviation_pct']:+.1f}%, {res['label']})"
                  if res.get("detected") else f"FAIL ({res.get('reason')})")
        print(f"  [{i}/{len(videos)}] {p.name}: {status} in {time.time()-t0:.1f}s",
              flush=True)

    print(f"\nDone. {len(cache)} entries in {CACHE}")
    print("→ Ouvre maintenant http://127.0.0.1:8000/test/mahieu")


if __name__ == "__main__":
    main()
