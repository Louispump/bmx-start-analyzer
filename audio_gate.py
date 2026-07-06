"""
Détection audio automatique du gate drop BMX.

Cadence UCI ProGate : 4 tones rapprochés. Le 4e coïncide avec la chute de la
grille. On extrait l'audio via ffmpeg (mono, 16 kHz), on calcule un onset
strength à partir de l'enveloppe d'énergie, on cherche les pics, puis on
identifie un groupe de 4 pics quasi-équidistants.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

SR = 16000
HOP_S = 0.005
WIN_S = 0.020
# Plage d'intervalles plausible entre 2 beeps consécutifs UCI (sec)
MIN_INTERVAL_S = 0.080
MAX_INTERVAL_S = 0.400
# Tolérance de régularité (écart-type / moyenne des intervalles)
MAX_REGULARITY = 0.35

# Décalage audio → chute visuelle du gate.
# L'onset du 4e bip = clic mécanique du déclencheur capté par le micro.
# La chute visuelle du gate suit avec ~95 ms de délai (temps mécanique entre
# déclenchement et grille effectivement au sol).
# Mesuré sur 4 vidéos via /gate_calibration (mai 2026) :
# diffs -90.4 / -90.4 / -97.3 / -130.7 ms (médiane -94 ms).
# À recalibrer si on accumule des calibrations divergentes.
GATE_VISUAL_OFFSET_S = 0.095

# Cadence UCI réelle : 120 ms entre les 4 bips, la grille tombe sur le 4e —
# soit 360 ms (3 × 120) après le 1er bip (voir SCIENCE.md §1). On s'en sert
# comme contrôle de qualité : un quadruplet détecté dont l'intervalle moyen
# s'éloigne de 120 ms est suspect (échos, faux pics), même s'il est régulier.
UCI_INTERVAL_S       = 0.120
UCI_BIP1_TO_GATE_S   = 0.360
CADENCE_TOL_MS       = 55.0   # écart toléré à 120 ms avant de déclasser


def _extract_audio(video_path: Path) -> np.ndarray | None:
    """Extrait l'audio en mono float32 normalisé via ffmpeg. None si échec."""
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-ac", "1", "-ar", str(SR),
        "-f", "s16le", "-loglevel", "error", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _onset_strength(samples: np.ndarray) -> np.ndarray:
    hop = int(SR * HOP_S)
    win = int(SR * WIN_S)
    n_hops = (len(samples) - win) // hop
    if n_hops < 10:
        return np.array([], dtype=np.float32)
    energy = np.empty(n_hops, dtype=np.float32)
    for i in range(n_hops):
        seg = samples[i * hop : i * hop + win]
        energy[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
    log_e = np.log(energy + 1e-6)
    onset = np.maximum(np.diff(log_e), 0.0)
    kernel = np.ones(3, dtype=np.float32) / 3.0
    return np.convolve(onset, kernel, mode="same")


def detect_gate_drop(video_path: Path) -> dict:
    """Retourne un dict avec au minimum 'detected' (bool).

    En cas de succès :
        gate_t (s), beeps_t [4 floats], mean_interval_ms, regularity, confidence (0-1)
    Sinon :
        reason (str)
    """
    samples = _extract_audio(video_path)
    if samples is None or len(samples) == 0:
        return {"detected": False, "reason": "audio_extract_failed"}

    onset = _onset_strength(samples)
    if onset.size == 0:
        return {"detected": False, "reason": "audio_too_short"}

    min_dist_hops = max(1, int(0.070 / HOP_S))
    threshold = max(onset.mean() + 2 * onset.std(),
                    onset.max() * 0.30)
    peak_idx, props = find_peaks(onset, distance=min_dist_hops, height=threshold)
    if len(peak_idx) < 4:
        return {"detected": False, "reason": f"only_{len(peak_idx)}_peaks"}

    peak_t = peak_idx * HOP_S
    peak_h = props["peak_heights"]

    best = None
    for i in range(len(peak_idx) - 3):
        ts = peak_t[i:i + 4]
        intervals = np.diff(ts)
        if intervals.min() < MIN_INTERVAL_S or intervals.max() > MAX_INTERVAL_S:
            continue
        regularity = float(intervals.std() / intervals.mean())
        if regularity > MAX_REGULARITY:
            continue
        strength = float(peak_h[i:i + 4].mean())
        score = strength / (1.0 + regularity)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "beeps_t": [float(t) for t in ts],
                "gate_t": float(ts[3]),
                "regularity": regularity,
                "mean_interval_ms": float(intervals.mean() * 1000),
                "method": "4-beep",
            }

    # Fallback : si le 4e beep est masqué (souvent par le bruit de la grille
    # qui tombe), on accepte un triplet de 3 pics dominants quasi-équidistants
    # et on extrapole la position du 4e (gate = pic3 + intervalle).
    if best is None and len(peak_idx) >= 3:
        height_thresh = float(np.percentile(peak_h, 60))
        for i in range(len(peak_idx) - 2):
            ts = peak_t[i:i + 3]
            hs = peak_h[i:i + 3]
            if hs.min() < height_thresh:
                continue
            intervals = np.diff(ts)
            if intervals.min() < MIN_INTERVAL_S or intervals.max() > MAX_INTERVAL_S:
                continue
            regularity = float(intervals.std() / intervals.mean())
            if regularity > 0.20:
                continue
            interval = float(intervals.mean())
            gate_t = float(ts[2] + interval)
            strength = float(hs.mean())
            score = strength / (1.0 + regularity) * 0.7  # pénalise le fallback
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "beeps_t": [float(t) for t in ts] + [gate_t],
                    "gate_t": gate_t,
                    "regularity": regularity,
                    "mean_interval_ms": interval * 1000,
                    "method": "3-beep+extrapolate",
                }

    if best is None:
        return {"detected": False, "reason": "no_4_beep_pattern"}

    # ── Contrôle qualité par la cadence UCI réelle (120 ms) ──────────────
    # Un quadruplet régulier mais à ~180 ms d'intervalle trahit des échos ou
    # de faux pics : on déclasse la confiance et on le signale.
    bip1          = best["beeps_t"][0]
    cadence_dev   = abs(best["mean_interval_ms"] - UCI_INTERVAL_S * 1000.0)
    cadence_ok    = cadence_dev <= CADENCE_TOL_MS
    base_conf     = min(1.0, best["score"] / 5.0)
    # Pénalité douce et bornée si la cadence dévie du 120 ms attendu.
    cadence_factor = max(0.4, 1.0 - cadence_dev / 240.0)
    confidence     = round(base_conf * cadence_factor, 3)

    # Gate alternatif ancré sur le 1er bip (souvent plus propre que le 4e, qui
    # est masqué par le bruit de la grille). Sert de croisement / repli.
    gate_from_bip1 = bip1 + UCI_BIP1_TO_GATE_S + GATE_VISUAL_OFFSET_S
    gate_beep4     = best["gate_t"] + GATE_VISUAL_OFFSET_S

    if confidence >= 0.5 and cadence_ok:
        quality = "high"
    elif confidence >= 0.3:
        quality = "medium"
    else:
        quality = "low"

    # Applique la correction visuelle : on retourne le moment où le gate est
    # visuellement tombé, pas le moment où le micro entend le clic.
    # beeps_t reste l'audio brut (bip1 sert d'ancrage du temps de réaction).
    return {
        "detected": True,
        "gate_t": gate_beep4,
        "gate_t_from_bip1": round(gate_from_bip1, 3),
        "beeps_t": best["beeps_t"],
        "mean_interval_ms": best["mean_interval_ms"],
        "regularity": best["regularity"],
        "cadence_dev_ms": round(cadence_dev, 1),
        "cadence_ok": cadence_ok,
        "confidence": confidence,
        "quality": quality,
        "method": best["method"],
        "gate_t_audio_raw": best["gate_t"],  # avant correction (debug)
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python audio_gate.py <video>")
        sys.exit(1)
    print(detect_gate_drop(Path(sys.argv[1])))
