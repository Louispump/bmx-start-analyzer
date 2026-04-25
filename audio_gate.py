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

    return {
        "detected": True,
        "gate_t": best["gate_t"],
        "beeps_t": best["beeps_t"],
        "mean_interval_ms": best["mean_interval_ms"],
        "regularity": best["regularity"],
        "confidence": min(1.0, best["score"] / 5.0),
        "method": best["method"],
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python audio_gate.py <video>")
        sys.exit(1)
    print(detect_gate_drop(Path(sys.argv[1])))
