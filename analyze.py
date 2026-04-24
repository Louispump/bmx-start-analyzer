"""
BMX Start Analyzer - MVP #2.0
Pose estimation + tracking + segmentation en phases (Kalichová)

Usage:
  python analyze.py videos/nom_video.mp4                      # gate drop auto-détecté
  python analyze.py videos/nom_video.mp4 --gate-drop 1.8     # secondes
  python analyze.py videos/nom_video.mp4 --gate-frame 55     # numéro de frame
  python analyze.py videos/nom_video.mp4 --gate-frame 55 --front-foot L
"""

import sys
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import savgol_filter, find_peaks
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

DEVICE = "mps"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

KEYPOINTS = {
    0: "nose", 5: "L_shoulder", 6: "R_shoulder",
    7: "L_elbow", 8: "R_elbow", 9: "L_wrist", 10: "R_wrist",
    11: "L_hip", 12: "R_hip", 13: "L_knee", 14: "R_knee",
    15: "L_ankle", 16: "R_ankle"
}

# Couleurs par phase (pour le graphique)
PHASE_COLORS = {
    "Set":         "#e8e8e8",
    "Reaction":    "#ffd6a5",
    "Push 1":      "#fdffb6",
    "Pull 1":      "#caffbf",
    "Push 2":      "#9bf6ff",
    "Post":        "#ffffff",
}


def calculate_angle(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
        return np.nan
    v1 = p1 - p2
    v2 = p3 - p2
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def pick_main_track(track_data, n_first_frames=15):
    scores = defaultdict(list)
    for frame_idx, tracks in track_data.items():
        if frame_idx >= n_first_frames:
            break
        for tid, (bbox_area, _) in tracks.items():
            scores[tid].append(bbox_area)
    if not scores:
        return None
    avg_scores = {tid: np.mean(areas) for tid, areas in scores.items()}
    return max(avg_scores, key=avg_scores.get)


def detect_visible_side(all_tracks, main_tid, n_frames=20):
    left_indices = [5, 7, 9, 11, 13, 15]
    right_indices = [6, 8, 10, 12, 14, 16]
    left_confs, right_confs = [], []
    for frame_idx in sorted(all_tracks.keys())[:n_frames]:
        tracks = all_tracks[frame_idx]
        if main_tid not in tracks:
            continue
        _, kpts = tracks[main_tid]
        if kpts is None:
            continue
        left_confs.extend([kpts[i, 2] for i in left_indices])
        right_confs.extend([kpts[i, 2] for i in right_indices])
    left_mean = np.mean(left_confs) if left_confs else 0
    right_mean = np.mean(right_confs) if right_confs else 0
    print(f"  Confiance moyenne côté gauche: {left_mean:.3f}")
    print(f"  Confiance moyenne côté droit:  {right_mean:.3f}")
    return "L" if left_mean > right_mean else "R"


def smooth_series(series, window=7, polyorder=3, max_gap=5):
    s = pd.Series(series).copy()
    s = s.interpolate(method='linear', limit=max_gap, limit_area='inside')
    mask = s.notna()
    if mask.sum() < window:
        return s.values
    arr = s.values.copy()
    valid_idx = np.where(mask)[0]
    if len(valid_idx) >= window:
        groups = np.split(valid_idx, np.where(np.diff(valid_idx) > 1)[0] + 1)
        for g in groups:
            if len(g) >= window:
                arr[g] = savgol_filter(arr[g], window_length=window, polyorder=polyorder)
    return arr


def detect_first_movement(df, gate_drop_time, threshold_deg=8.0,
                          pre_gate_s=2.0, post_gate_s=0.6,
                          bip1_time=None):
    """
    Détecte le premier mouvement significatif du rider.

    Référence = angle médian du genou sur les frames centrales du Set.
    Cherche depuis bip1 (ou 2s avant gate si pas de bip1) jusqu'à 0.6s après gate.

    Retourne (first_move_idx, reaction_type) où reaction_type est:
      "false_start"  — mouvement avant bip 1 (illégal)
      "bip"          — mouvement entre bip 1 et gate drop (stratégie normale)
      "gate"         — mouvement après gate drop
    """
    gate_idx = (df["time"] - gate_drop_time).abs().idxmin()
    fps_approx = 1.0 / df["time"].diff().median()

    # Référence stable: frames centrales du Set (25%→50% du Set)
    stable_start = max(0, gate_idx // 4)
    stable_end   = max(stable_start + 1, gate_idx // 2)
    knee_stable = df.loc[stable_start:stable_end, "knee_angle"].dropna().median()
    if np.isnan(knee_stable):
        knee_stable = df.loc[gate_idx, "knee_angle"]
    if np.isnan(knee_stable):
        return gate_idx, "gate"

    # Début du scan: depuis bip 1 si disponible, sinon 2s avant gate
    if bip1_time is not None:
        search_start = max(0, (df["time"] - bip1_time).abs().idxmin())
    else:
        search_start = max(0, gate_idx - int(pre_gate_s * fps_approx))
    search_end = min(len(df) - 1, gate_idx + int(post_gate_s * fps_approx))

    # Exiger 2 frames consécutives au-dessus du seuil (robustesse bruit YOLO)
    consecutive = 0
    first_candidate = None
    for i in range(search_start, search_end + 1):
        if not np.isnan(df.loc[i, "knee_angle"]):
            if abs(df.loc[i, "knee_angle"] - knee_stable) > threshold_deg:
                if first_candidate is None:
                    first_candidate = i
                consecutive += 1
                if consecutive >= 2:
                    t_move = df.loc[first_candidate, "time"]
                    if bip1_time is not None and t_move < bip1_time:
                        return first_candidate, "false_start"
                    elif t_move < gate_drop_time:
                        return first_candidate, "bip"
                    else:
                        return first_candidate, "gate"
            else:
                consecutive = 0
                first_candidate = None
    return gate_idx, "gate"


def detect_crank_events(df, first_move_idx, ankle_col, n_cranks=3):
    """
    Détecte les points morts haut/bas de la pédale (séquence push-pull-push).

    Utilise la position Y de la cheville RELATIVE à la hanche pour supprimer
    la dérive de posture (corps qui descend lors de l'accélération).

    Signal relatif (ankle_y - hip_y):
    - relatif MIN = cheville haute par rapport hanche = pédale à 12h (fin Pull)
    - relatif MAX = cheville basse par rapport hanche = pédale à 6h  (fin Push)

    Séquence attendue: MAX → MIN → MAX (fin Push1, fin Pull1, fin Push2)
    """
    sub = df.iloc[first_move_idx:].copy().reset_index(drop=False)

    # Dériver le nom de la hanche depuis le nom de la cheville (ex: L_ankle_y → L_hip_y)
    y_ankle = sub[ankle_col].values

    valid_times = sub["time"].dropna().values
    if len(valid_times) < 2:
        return []
    fps_approx = 1 / np.median(np.diff(valid_times))

    # Ignorer les 150ms après le premier mouvement (zone transitoire bruitée)
    skip_frames = max(2, int(0.15 * fps_approx))
    # Distance minimale entre deux événements: 180ms
    min_dist = max(3, int(0.18 * fps_approx))

    # Interpoler pour que find_peaks fonctionne sur les NaN
    y_interp = pd.Series(y_ankle).interpolate(method='linear', limit_area='inside').values

    # Détrendre le signal: soustraire un polynôme quadratique pour supprimer la
    # dérive de posture (corps qui monte/avance lors de l'accélération), ne garder
    # que l'oscillation du crank.
    x = np.arange(len(y_interp))
    valid_mask = ~np.isnan(y_ankle)
    if valid_mask.sum() > 6:
        coeffs = np.polyfit(x[valid_mask], y_interp[valid_mask], deg=2)
        trend = np.polyval(coeffs, x)
        y_detrended = y_interp - trend
    else:
        y_detrended = y_interp

    # Prominence adaptative sur le signal détrendé: 15% de l'amplitude (plancher 5px)
    amplitude = float(np.nanmax(y_detrended) - np.nanmin(y_detrended))
    if amplitude < 5:
        print("  [cranks] signal trop plat après détrend, abandon")
        return []
    prominence = max(5.0, 0.15 * amplitude)

    maxima, _ = find_peaks(y_detrended, distance=min_dist, prominence=prominence)
    minima, _ = find_peaks(-y_detrended, distance=min_dist, prominence=prominence)

    # Supprimer les événements dans la zone transitoire
    maxima = maxima[maxima >= skip_frames]
    minima = minima[minima >= skip_frames]

    print(f"  [cranks] signal=ankle_y détrendé (polynôme deg2)")
    print(f"  [cranks] amplitude={amplitude:.0f}px  prominence={prominence:.0f}px  "
          f"skip={skip_frames}f  min_dist={min_dist}f")
    print(f"  [cranks] maxima bruts: {list(maxima[:6])} | minima bruts: {list(minima[:6])}")

    # Séquence attendue: max → min → max
    all_events = sorted(
        [(i, "max") for i in maxima] + [(i, "min") for i in minima],
        key=lambda x: x[0]
    )

    expected = ["max", "min", "max"]
    events = []
    for e_idx, e_type in all_events:
        if len(events) >= len(expected):
            break
        if e_type == expected[len(events)]:
            events.append(sub.loc[e_idx, "index"])

    print(f"  [cranks] événements retenus: {events} "
          f"({len(events)}/3 phases détectées)")
    return events


def segment_phases(df, gate_drop_time, ankle_col, bip1_time=None):
    """
    Segmente la vidéo en phases selon Kalichová.

    Set toujours ancré sur gate_drop (définition Kalichová).
    reaction_type:
      "gate"        → rider réagit après gate drop → Reaction = gate→first_move
      "bip"         → rider démarre entre bip1 et gate (stratégie normale) → Reaction = 0ms
      "false_start" → rider bouge avant bip1 (illégal) → Reaction = 0ms
    """
    phases = {}
    gate_idx = (df["time"] - gate_drop_time).abs().idxmin()

    first_move_idx, reaction_type = detect_first_movement(
        df, gate_drop_time, bip1_time=bip1_time
    )

    # Set = toujours 0 → gate_drop
    phases["Set"] = (0, gate_idx)

    if reaction_type == "gate":
        phases["Reaction"] = (gate_idx, first_move_idx)
        crank_start = first_move_idx
    else:
        # Bip ou faux départ: rider déjà en mouvement au gate drop
        phases["Reaction"] = (gate_idx, gate_idx)  # 0ms
        crank_start = gate_idx

    crank_events = detect_crank_events(df, crank_start, ankle_col)
    if len(crank_events) >= 1:
        phases["Push 1"] = (crank_start, crank_events[0])
    if len(crank_events) >= 2:
        phases["Pull 1"] = (crank_events[0], crank_events[1])
    if len(crank_events) >= 3:
        phases["Push 2"] = (crank_events[1], crank_events[2])
        phases["Post"] = (crank_events[2], len(df) - 1)
    else:
        last_idx = crank_events[-1] if crank_events else crank_start
        phases["Post"] = (last_idx, len(df) - 1)

    return phases, first_move_idx, reaction_type

def detect_front_foot(all_tracks, main_tid, n_frames=30):
    """
    Détecte le pied avant en utilisant la direction du vélo (estimée par
    la position du nez ou des poignets par rapport aux hanches).
    Le pied avant = celui dont la cheville est la plus avancée dans cette direction.
    """
    direction_signs = []  # +1 si rider orienté vers droite image, -1 vers gauche
    L_ankle_xs, R_ankle_xs = [], []
    hip_xs = []
    
    for frame_idx in sorted(all_tracks.keys())[:n_frames]:
        tracks = all_tracks[frame_idx]
        if main_tid not in tracks:
            continue
        _, kpts = tracks[main_tid]
        if kpts is None:
            continue
        
        nose_x, nose_conf = kpts[0, 0], kpts[0, 2]
        L_hip_x, L_hip_conf = kpts[11, 0], kpts[11, 2]
        R_hip_x, R_hip_conf = kpts[12, 0], kpts[12, 2]
        L_ankle_x, L_ankle_conf = kpts[15, 0], kpts[15, 2]
        R_ankle_x, R_ankle_conf = kpts[16, 0], kpts[16, 2]
        
        # Centre des hanches
        if L_hip_conf > 0.3 and R_hip_conf > 0.3:
            hip_center_x = (L_hip_x + R_hip_x) / 2
            hip_xs.append(hip_center_x)
            
            # Direction: le nez est-il à gauche ou à droite du centre des hanches?
            if nose_conf > 0.3:
                # Si nose_x > hip_center_x : rider regarde vers la droite image (+1)
                # Si nose_x < hip_center_x : rider regarde vers la gauche image (-1)
                direction_signs.append(1 if nose_x > hip_center_x else -1)
        
        if L_ankle_conf > 0.3:
            L_ankle_xs.append(L_ankle_x)
        if R_ankle_conf > 0.3:
            R_ankle_xs.append(R_ankle_x)
    
    if not L_ankle_xs or not R_ankle_xs or not direction_signs:
        return None, 0

    # Direction moyenne du vélo (+1 = roule vers droite image, -1 = vers gauche)
    direction = 1 if np.mean(direction_signs) > 0 else -1
    
    L_mean_x = np.mean(L_ankle_xs)
    R_mean_x = np.mean(R_ankle_xs)
    
    print(f"  Direction du vélo: {'→ droite image' if direction > 0 else '← gauche image'}")
    print(f"  Position X moyenne cheville gauche: {L_mean_x:.0f} px")
    print(f"  Position X moyenne cheville droite: {R_mean_x:.0f} px")
    
    # Le pied avant est celui qui est dans la direction du mouvement
    # Si direction = +1 (vélo vers droite), le pied avant a X plus GRAND
    # Si direction = -1 (vélo vers gauche), le pied avant a X plus PETIT
    if direction > 0:
        front = "L" if L_mean_x > R_mean_x else "R"
    else:
        front = "L" if L_mean_x < R_mean_x else "R"
    
    diff = abs(L_mean_x - R_mean_x)
    print(f"  → Pied avant détecté: {'GAUCHE' if front == 'L' else 'DROIT'} "
          f"(écart: {diff:.0f}px)")
    
    if diff < 20:
        print(f"  ⚠️  Écart faible ({diff:.0f}px), détection peu fiable")

    return front, direction


def classify_set_position(df, gate_idx, side, direction):
    """
    Classifie la position de set selon Grigg: Upright / Angled / Back.

    Angle du tronc = vecteur hanche→épaule par rapport à la verticale,
    signé selon la direction du vélo (+ = penché vers l'avant).

    Thresholds Grigg:
      Back    : angle < 0°    (penché vers l'arrière)
      Upright : 0° à 20°     (quasi vertical)
      Angled  : > 20°         (penché vers l'avant)
    """
    # Frames stables du Set: 25%→75% de la phase (évite début bruité et fin avec bips)
    stable_start = max(0, gate_idx // 4)
    stable_end   = max(stable_start + 1, 3 * gate_idx // 4)
    sub = df.loc[stable_start:stable_end]

    sh_x = sub[f"{side}_shoulder_x"].dropna()
    sh_y = sub[f"{side}_shoulder_y"].dropna()
    hi_x = sub[f"{side}_hip_x"].dropna()
    hi_y = sub[f"{side}_hip_y"].dropna()

    common = sh_x.index & sh_y.index & hi_x.index & hi_y.index
    if len(common) < 3:
        return None, None

    dx = (sh_x[common] - hi_x[common]).values  # déplacement horizontal épaule vs hanche
    dy = (sh_y[common] - hi_y[common]).values  # négatif car épaule au-dessus hanche (coords image)

    # Composante "avant" selon direction du vélo
    forward_dx = dx * direction
    # Angle depuis la verticale: + = penché avant, - = penché arrière
    angles = np.degrees(np.arctan2(forward_dx, -dy))
    trunk_angle = float(np.median(angles))

    if trunk_angle < 0:
        label = "Back"
    elif trunk_angle <= 20:
        label = "Upright"
    else:
        label = "Angled"

    return trunk_angle, label


def track_hub_trajectory(video_path, all_tracks, main_tid, ankle_k, side, direction,
                          gate_drop_time, fps):
    """
    Suit la trajectoire du moyeu avant sur la vidéo brute.

    Stratégie hybride:
      1. Hough circles dans une ROI autour de la roue avant (guidée par poignet avant)
      2. Fallback: position du poignet avant comme proxy si Hough échoue

    Retourne un dict {frame_idx: (hub_x, hub_y)} et la classification.
    """
    # Indice du poignet avant (même côté que le pied avant)
    wrist_key = ankle_k.replace("ankle", "wrist")   # ex: L_ankle → L_wrist
    wrist_x_col = f"{wrist_key}_x"
    wrist_y_col = f"{wrist_key}_y"
    # Indice keypoint du poignet avant
    wrist_kp_idx = 9 if ankle_k.startswith("L") else 10   # L_wrist=9, R_wrist=10
    ankle_kp_idx = 15 if ankle_k.startswith("L") else 16
    hip_kp_idx   = 11 if ankle_k.startswith("L") else 12

    cap = cv2.VideoCapture(str(video_path))
    hub_positions = {}   # frame_idx → (x, y)

    frame_idx = 0
    prev_hub = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in all_tracks and main_tid in all_tracks[frame_idx]:
            _, kpts = all_tracks[frame_idx][main_tid]
            if kpts is not None:
                wx = float(kpts[wrist_kp_idx, 0])
                wy = float(kpts[wrist_kp_idx, 1])
                ay = float(kpts[ankle_kp_idx, 1])
                hy = float(kpts[hip_kp_idx,   1])

                if wx > 0 and wy > 0 and ay > 0:
                    h_img, w_img = frame.shape[:2]

                    # Rayon estimé de la roue: ~40% de la distance hanche-cheville
                    wheel_r = max(20, min(int(abs(ay - hy) * 0.40), int(h_img * 0.25)))

                    # ROI: centré sur le poignet avant, étendu vers le bas
                    cx_roi = int(wx)
                    cy_roi = int((wy + ay) / 2)
                    margin = int(wheel_r * 1.6)
                    x1 = max(0, cx_roi - margin)
                    x2 = min(w_img, cx_roi + margin)
                    y1 = max(0, cy_roi - margin)
                    y2 = min(h_img, cy_roi + margin)

                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        frame_idx += 1
                        continue

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (7, 7), 2)

                    circles = cv2.HoughCircles(
                        gray, cv2.HOUGH_GRADIENT, dp=1.2,
                        minDist=wheel_r,
                        param1=60, param2=25,
                        minRadius=int(wheel_r * 0.75),
                        maxRadius=int(wheel_r * 1.30)
                    )

                    if circles is not None:
                        circles = np.squeeze(circles, axis=0)
                        if circles.ndim == 1:
                            circles = circles[np.newaxis, :]
                        # Choisir le cercle le plus près du hub précédent (ou du centre ROI)
                        ref = prev_hub if prev_hub else (cx_roi, cy_roi)
                        best = min(circles,
                                   key=lambda c: (x1+c[0]-ref[0])**2 + (y1+c[1]-ref[1])**2)
                        hx = int(x1 + best[0])
                        hy_ = int(y1 + best[1])
                        hub_positions[frame_idx] = (hx, hy_)
                        prev_hub = (hx, hy_)
                    else:
                        # Fallback: poignet avant
                        hub_positions[frame_idx] = (int(wx), int(wy))

        frame_idx += 1
    cap.release()
    return hub_positions


def smooth_hub_positions(hub_positions, fps, max_jump_px=80):
    """
    Filtre les outliers et lisse la trajectoire du moyeu.
    - Supprime les points où le saut inter-frame dépasse max_jump_px
    - Lisse avec Savitzky-Golay
    """
    if len(hub_positions) < 4:
        return hub_positions

    frames = sorted(hub_positions.keys())
    xs = np.array([hub_positions[f][0] for f in frames], dtype=float)
    ys = np.array([hub_positions[f][1] for f in frames], dtype=float)

    # Filtrer les outliers par saut brusque
    keep = np.ones(len(frames), dtype=bool)
    for i in range(1, len(frames)):
        dx = abs(xs[i] - xs[i-1])
        dy = abs(ys[i] - ys[i-1])
        if dx > max_jump_px or dy > max_jump_px:
            keep[i] = False

    frames_f = [f for f, k in zip(frames, keep) if k]
    xs_f = xs[keep]
    ys_f = ys[keep]

    if len(xs_f) < 5:
        return {f: (x, y) for f, x, y in zip(frames_f, xs_f, ys_f)}

    # Savitzky-Golay
    win = min(7, len(xs_f) if len(xs_f) % 2 == 1 else len(xs_f) - 1)
    win = max(win, 3)
    if win % 2 == 0:
        win -= 1
    xs_s = savgol_filter(xs_f, window_length=win, polyorder=2)
    ys_s = savgol_filter(ys_f, window_length=win, polyorder=2)

    return {f: (float(x), float(y)) for f, x, y in zip(frames_f, xs_s, ys_s)}


def classify_hub_trajectory(hub_positions, gate_drop_time, fps, direction, phases):
    """
    Classifie la trajectoire du moyeu: 'hairpin' ou 'high'.

    Hairpin (épingle): le moyeu recule en X (vers la gate) juste après le gate drop
    avant d'avancer. Caractéristique des meilleurs départs BMX (Grigg).

    direction: +1 = rider va vers la droite, -1 = vers la gauche

    Retourne (type, backward_px, traj_xy) où traj_xy = [(x,y,frame)] pendant Push 1.
    """
    if not hub_positions:
        return "unknown", 0.0, []

    gate_frame = int(gate_drop_time * fps)

    # Extraire Push 1 (de gate_drop à fin Push 1 selon les phases)
    push1_end_frame = None
    if "Push 1" in phases:
        _, end_idx = phases["Push 1"]
        push1_end_frame = end_idx

    # Fenêtre d'analyse: gate_drop → gate_drop + 0.6s (ou fin Push 1)
    max_frame = gate_frame + int(0.6 * fps)
    if push1_end_frame is not None:
        max_frame = max(max_frame, push1_end_frame)

    traj = [(x, y, f) for f, (x, y) in sorted(hub_positions.items())
            if gate_frame <= f <= max_frame]

    if len(traj) < 4:
        return "unknown", 0.0, traj

    xs = np.array([p[0] for p in traj])
    # x_forward: positif = vers l'avant du rider
    x_fwd = xs * direction

    # Chercher le recul max dans les 200ms post-gate
    n_early = max(3, int(0.20 * fps))
    x_early = x_fwd[:min(n_early, len(x_fwd))]
    x0     = x_early[0]
    x_min  = float(np.min(x_early))
    backward_px = float(x0 - x_min)   # positif = recul réel

    HAIRPIN_THRESHOLD = 8.0   # px minimum pour parler de hairpin

    if backward_px >= HAIRPIN_THRESHOLD:
        traj_type = "hairpin"
    else:
        traj_type = "high"

    return traj_type, backward_px, traj


def main(video_path, front_foot=None, gate_drop=None, bip1_time=None):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERREUR: fichier introuvable: {video_path}")
        return
    if gate_drop is None:
        print("ERREUR: --gate-drop est requis (ex: --gate-drop 1.8)")
        print("Indique le temps en secondes où la grille commence à tomber.")
        return

    video_name = video_path.stem
    print(f"Loading YOLOv11-Pose model...")
    model = YOLO("yolo11m-pose.pt")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_name} | {width}x{height} @ {fps:.1f}fps, {n_frames} frames")
    print(f"Gate drop spécifié: t = {gate_drop:.2f}s (frame ~{int(gate_drop * fps)})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(OUTPUT_DIR / f"{video_name}_annotated.mp4"),
        fourcc, fps, (width, height)
    )

    # === PASSE 1: Tracking ===
    print("Pass 1: tracking...")
    all_tracks = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, verbose=False, device=DEVICE,
                               persist=True, tracker="bytetrack.yaml")
        r = results[0]
        frame_tracks = {}
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            boxes = r.boxes.xywh.cpu().numpy()
            kpts_all = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for i, tid in enumerate(ids):
                bbox_area = boxes[i, 2] * boxes[i, 3]
                kpts = kpts_all[i] if kpts_all is not None else None
                frame_tracks[int(tid)] = (bbox_area, kpts)
        all_tracks[frame_idx] = frame_tracks
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{n_frames}")
    cap.release()

    main_tid = pick_main_track(all_tracks, n_first_frames=15)
    if main_tid is None:
        print("ERREUR: aucun rider détecté")
        return
    print(f"Rider principal: track_id = {main_tid}")

    print("Detecting camera side...")
    camera_side = detect_visible_side(all_tracks, main_tid)
    camera_side_name = "GAUCHE" if camera_side == "L" else "DROIT"
    print(f"  → Côté filmé: {camera_side_name}")

    print("Detecting front foot...")
    detected_front, direction = detect_front_foot(all_tracks, main_tid)

    # Déterminer le pied avant à analyser (priorité à --front-foot si fourni)
    if front_foot is not None:
        side = front_foot
        source = "spécifié par l'utilisateur"
    elif detected_front is not None:
        side = detected_front
        source = "détecté automatiquement"
    else:
        side = camera_side
        source = "fallback sur côté filmé (détection auto a échoué)"

    side_name = "GAUCHE" if side == "L" else "DROIT"
    print(f"\n  → Pied avant analysé: {side_name} ({source})")

    # Warning si le pied avant est caché
    if side != camera_side:
        print(f"  ⚠️  Pied avant ({side_name}) caché derrière le vélo.")
        print(f"     L'analyse sera moins précise (landmarks inférés).")
        print(f"     Pour une analyse optimale: filmer du côté {side_name}.")
    else:
        print(f"  ✓ Setup optimal: pied avant visible à la caméra.")

    shoulder_k = f"{side}_shoulder"
    elbow_k    = f"{side}_elbow"
    wrist_k    = f"{side}_wrist"
    hip_k      = f"{side}_hip"
    knee_k     = f"{side}_knee"
    ankle_k    = f"{side}_ankle"

    # === PASSE 2: Extraction ===
    print("Pass 2: extracting data...")
    cap = cv2.VideoCapture(str(video_path))
    records = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        row = {"frame": frame_idx, "time": frame_idx / fps}
        frame_tracks = all_tracks.get(frame_idx, {})
        if main_tid in frame_tracks:
            _, kpts = frame_tracks[main_tid]
            for idx, name in KEYPOINTS.items():
                x, y, conf = kpts[idx]
                row[f"{name}_x"] = float(x) if conf > 0.3 else np.nan
                row[f"{name}_y"] = float(y) if conf > 0.3 else np.nan
                row[f"{name}_conf"] = float(conf)
        else:
            for name in KEYPOINTS.values():
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_conf"] = 0.0
        records.append(row)
        frame_idx += 1
    cap.release()

    df = pd.DataFrame(records)

    # Lissage + calcul d'angles
    print("Smoothing and computing angles...")
    for name in KEYPOINTS.values():
        df[f"{name}_x"] = smooth_series(df[f"{name}_x"].values)
        df[f"{name}_y"] = smooth_series(df[f"{name}_y"].values)

    knee_angles, hip_angles, elbow_angles = [], [], []
    for _, r in df.iterrows():
        shoulder = (r[f"{shoulder_k}_x"], r[f"{shoulder_k}_y"])
        elbow    = (r[f"{elbow_k}_x"], r[f"{elbow_k}_y"])
        wrist    = (r[f"{wrist_k}_x"], r[f"{wrist_k}_y"])
        hip      = (r[f"{hip_k}_x"], r[f"{hip_k}_y"])
        knee     = (r[f"{knee_k}_x"], r[f"{knee_k}_y"])
        ankle    = (r[f"{ankle_k}_x"], r[f"{ankle_k}_y"])
        knee_angles.append(calculate_angle(hip, knee, ankle))
        hip_angles.append(calculate_angle(shoulder, hip, knee))
        elbow_angles.append(calculate_angle(shoulder, elbow, wrist))
    df["knee_angle"] = knee_angles
    df["hip_angle"] = hip_angles
    df["elbow_angle"] = elbow_angles

    # === Position de set (Grigg) ===
    gate_idx_for_set = (df["time"] - gate_drop).abs().idxmin()
    trunk_angle, set_label = classify_set_position(df, gate_idx_for_set, side, direction)

    # === Segmentation en phases ===
    print("Segmenting phases (Kalichová)...")
    ankle_col = f"{ankle_k}_y"  # ex: "R_ankle_y" ou "L_ankle_y"
    phases, first_move_idx, reaction_type = segment_phases(df, gate_drop, ankle_col,
                                                             bip1_time=bip1_time)

    # === Trajectoire du moyeu avant (Grigg: hairpin vs haute) ===
    print("Tracking hub trajectory...")
    hub_positions = track_hub_trajectory(
        video_path, all_tracks, main_tid, ankle_k, side, direction, gate_drop, fps
    )
    hub_positions = smooth_hub_positions(hub_positions, fps)
    hub_type, hub_backward_px, hub_traj = classify_hub_trajectory(
        hub_positions, gate_drop, fps, direction, phases
    )
    
    # Étiquette de phase pour chaque frame dans le CSV
    df["phase"] = "Unknown"
    for phase_name, (start, end) in phases.items():
        df.loc[start:end, "phase"] = phase_name

    df["set_trunk_angle"] = trunk_angle if trunk_angle is not None else np.nan
    df["set_position"]    = set_label   if set_label   is not None else "Unknown"
    df.to_csv(OUTPUT_DIR / f"{video_name}_landmarks.csv", index=False)

    # === Rendu vidéo annotée ===
    print("Rendering annotated video...")
    cap = cv2.VideoCapture(str(video_path))
    if side == "R":
        segments = [(6, 8), (8, 10), (6, 12), (12, 14), (14, 16),
                    (5, 6), (11, 12), (5, 11)]
        points_to_draw = [6, 8, 10, 12, 14, 16, 0, 5, 11]
    else:
        segments = [(5, 7), (7, 9), (5, 11), (11, 13), (13, 15),
                    (5, 6), (11, 12), (6, 12)]
        points_to_draw = [5, 7, 9, 11, 13, 15, 0, 6, 12]

    idx_to_name = {i: n for i, n in KEYPOINTS.items()}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        row = df.iloc[frame_idx]
        current_phase = row["phase"]
        
        for idx in points_to_draw:
            name = idx_to_name[idx]
            x, y = row[f"{name}_x"], row[f"{name}_y"]
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        for s, e in segments:
            s_name, e_name = idx_to_name[s], idx_to_name[e]
            x1, y1 = row[f"{s_name}_x"], row[f"{s_name}_y"]
            x2, y2 = row[f"{e_name}_x"], row[f"{e_name}_y"]
            if not any(np.isnan(v) for v in [x1, y1, x2, y2]):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 0), 2)

        # Infobox en haut à gauche
        cv2.rectangle(frame, (10, 10), (400, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Phase: {current_phase}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Cote: {side_name}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"t={row['time']:.2f}s", (280, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_video.write(frame)
        frame_idx += 1
    cap.release()
    out_video.release()

    # === Graphique avec zones colorées par phase ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    plot_defs = [
        (axes[0], "knee_angle",  "Angle du genou (°)"),
        (axes[1], "hip_angle",   "Angle de la hanche (°)"),
        (axes[2], "elbow_angle", "Angle du coude (°)"),
    ]
    for ax, col, title in plot_defs:
        # Bandes colorées pour les phases
        for phase_name, (start, end) in phases.items():
            t_start = df.loc[start, "time"]
            t_end = df.loc[end, "time"]
            ax.axvspan(t_start, t_end, alpha=0.3, color=PHASE_COLORS.get(phase_name, "#ffffff"),
                       label=phase_name if ax == axes[0] else None)
        # Ligne verticale à gate drop
        ax.axvline(gate_drop, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        # Courbe
        ax.plot(df["time"], df[col], color='#1f77b4', linewidth=1.8, zorder=10)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9, ncol=6)
    axes[-1].set_xlabel("Temps (s)")
    fig.suptitle(f"Angles et phases — {video_name} | côté {side_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{video_name}_angles.png", dpi=120)
    plt.close()

    # === Position de set ===
    print(f"\n=== POSITION DE SET (Grigg) ===")
    if trunk_angle is not None:
        print(f"  Angle du tronc: {trunk_angle:+.1f}° depuis la verticale "
              f"({'avant' if trunk_angle >= 0 else 'arrière'})")
        print(f"  Classification: {set_label}")
        print(f"  Référence Grigg: Upright (0-20°) | Angled (>20° avant) | Back (<0°)")
    else:
        print(f"  ⚠️  Données insuffisantes pour classifier la position de set")

    # === Stats par phase ===
    print(f"\n=== PHASES pour {video_name} ===")
    for phase_name, (start, end) in phases.items():
        t_start = df.loc[start, "time"]
        t_end = df.loc[end, "time"]
        duration = t_end - t_start
        print(f"  {phase_name:<10} | t={t_start:.2f}s → {t_end:.2f}s | durée={duration*1000:.0f}ms")

    print(f"\n=== MÉTRIQUES PAR PHASE ===")
    benchmark = {"knee_angle": "93°±12°", "hip_angle": "62°±11°", "elbow_angle": "47°±15°"}
    labels = {"knee_angle": "Genou", "hip_angle": "Hanche", "elbow_angle": "Coude"}
    
    for phase_name in ["Push 1", "Pull 1", "Push 2"]:
        if phase_name not in phases:
            continue
        start, end = phases[phase_name]
        sub = df.loc[start:end]
        print(f"\n  --- {phase_name} ---")
        for col in ["knee_angle", "hip_angle", "elbow_angle"]:
            vals = sub[col].dropna()
            if len(vals) > 0:
                amp = vals.max() - vals.min()
                print(f"    {labels[col]:<8}: amplitude={amp:.1f}° "
                      f"(benchmark élite Grigg: {benchmark[col]})")

    # Stat de réaction
    t_move = df.loc[first_move_idx, "time"]
    react_from_gate = t_move - gate_drop
    print(f"\n=== TEMPS DE RÉACTION ===")
    print(f"  Premier mouvement détecté à t={t_move:.3f}s")
    if reaction_type == "false_start":
        print(f"  ⚠️  FAUX DÉPART — mouvement {abs(react_from_gate)*1000:.0f}ms "
              f"AVANT le bip 1 (t={bip1_time:.3f}s)")
    elif reaction_type == "bip":
        react_from_bip1 = t_move - bip1_time if bip1_time is not None else None
        if react_from_bip1 is not None:
            print(f"  Réaction aux bips: {react_from_bip1*1000:.0f}ms après bip 1")
        print(f"  Avance sur le gate: {abs(react_from_gate)*1000:.0f}ms "
              f"(gate drop = bip 4, t={gate_drop:.3f}s)")
        print(f"  → Stratégie bips: normal en BMX compétitif")
    else:
        print(f"  Temps de réaction depuis gate drop: {react_from_gate*1000:.0f}ms")

    # === Trajectoire du moyeu ===
    print(f"\n=== TRAJECTOIRE DU MOYEU AVANT (Grigg) ===")
    if hub_type == "unknown":
        print(f"  ⚠️  Données insuffisantes pour classifier la trajectoire")
    else:
        icon = "✓" if hub_type == "hairpin" else "○"
        print(f"  {icon} Type: {hub_type.upper()}")
        print(f"  Recul initial: {hub_backward_px:.0f}px "
              f"({'hairpin confirmé' if hub_backward_px >= 8 else 'trajectoire haute'})")
        print(f"  Référence Grigg: hairpin = meilleur transfert d'énergie au départ")

    # Graphique trajectoire hub
    if len(hub_traj) >= 4:
        fig_h, ax_h = plt.subplots(figsize=(7, 5))
        hx = np.array([p[0] for p in hub_traj])
        hy = np.array([p[1] for p in hub_traj])
        hf = np.array([p[2] for p in hub_traj])
        ht = hf / fps

        # Couleur par phase
        phase_frames = {name: rng for name, rng in phases.items()}
        colors_traj = []
        for f in hf:
            c = "#aaaaaa"
            for pname, (ps, pe) in phase_frames.items():
                if ps <= f <= pe:
                    c = PHASE_COLORS.get(pname, "#aaaaaa")
                    break
            colors_traj.append(c)

        # Tracer la trajectoire (Y inversé: haut = bas en image)
        for i in range(len(hx) - 1):
            ax_h.plot(hx[i:i+2], [-hy[i], -hy[i+1]], color=colors_traj[i], linewidth=2.5)

        # Marquer le début (gate drop)
        ax_h.plot(hx[0], -hy[0], 'ro', markersize=10, label='Gate drop', zorder=5)
        ax_h.plot(hx[-1], -hy[-1], 'bs', markersize=8, label='Fin Push 1', zorder=5)

        # Flèche direction avant
        ax_h.annotate("", xy=(hx[0] + 30 * direction, -hy[0]),
                       xytext=(hx[0], -hy[0]),
                       arrowprops=dict(arrowstyle="->", color="gray"))

        ax_h.set_xlabel("Position X (px)")
        ax_h.set_ylabel("Position Y (px, haut = positif)")
        title_type = "HAIRPIN ✓" if hub_type == "hairpin" else "HIGH (pas de hairpin)"
        ax_h.set_title(f"Trajectoire moyeu avant — {video_name}\n{title_type} | recul={hub_backward_px:.0f}px",
                        fontsize=11)
        ax_h.legend(fontsize=9)
        ax_h.grid(True, alpha=0.3)
        ax_h.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{video_name}_hub_trajectory.png", dpi=120)
        plt.close()
        print(f"  Graphique: output/{video_name}_hub_trajectory.png")

    print(f"\nFichiers: output/{video_name}_annotated.mp4 + _landmarks.csv + _angles.png")


def detect_beeps_audio(video_path):
    """
    Détecte les 4 bips de départ BMX dans la piste audio.

    Séquence réelle:
      "Riders ready watch the gate" → silence (0.1-2.9s) → 4 bips rapides (<1s) → gate drop

    Les 4 bips sont régulièrement espacés (même intervalle ~60ms), tous en <1s.
    Le gate tombe au 4ème bip.

    Stratégie:
      1. Filtre 400-800Hz (meilleure plage pour les bips BMX ~630-680Hz dans le bruit)
      2. Détecter le silence entre l'annonce et les bips → calibrer le noise floor
      3. Chercher 3 ou 4 pics réguliers (40-120ms) juste après le silence
      4. Si seulement 3 détectés, extrapoler le 4ème par l'intervalle médian

    Retourne (beep_times, gate_time, confidence) ou (None, None, 0) si pas d'audio.
    """
    import subprocess, tempfile, os
    from scipy.signal import butter, filtfilt

    # 1. Extraire l'audio en WAV mono 44100Hz
    wav_path = tempfile.mktemp(suffix='.wav')
    result = subprocess.run(
        ['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le',
         '-ar', '44100', '-ac', '1', wav_path, '-y', '-loglevel', 'quiet'],
        capture_output=True
    )
    if result.returncode != 0 or not Path(wav_path).exists():
        return None, None, 0.0

    try:
        import scipy.io.wavfile as wavfile
        sr, audio = wavfile.read(wav_path)
    except Exception:
        return None, None, 0.0
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    if audio is None or len(audio) == 0:
        return None, None, 0.0

    audio = audio.astype(float) / (2**15)

    # 2. Filtre passe-bande 400-800Hz
    #    (ratio bip/bruit: 5.1× vs 1.9× pour la bande étroite 580-750Hz)
    nyq = sr / 2
    b, a = butter(4, [400 / nyq, 800 / nyq], btype='band')
    filtered = filtfilt(b, a, audio)

    # 3. Énergie RMS haute résolution: fenêtre 20ms, hop 5ms
    hop = int(0.005 * sr)   # 5ms
    win = int(0.020 * sr)   # 20ms — résolution pour bips courts
    energy = np.array([
        np.sqrt(np.mean(filtered[i:i + win] ** 2))
        for i in range(0, len(filtered) - win, hop)
    ])
    t_e = np.arange(len(energy)) * hop / sr

    # 4. Détecter le silence (annonce → silence → bips) sur signal large bande
    hop_lo = int(0.050 * sr)
    win_lo = int(0.200 * sr)
    energy_broad = np.array([
        np.sqrt(np.mean(audio[i:i + win_lo] ** 2))
        for i in range(0, len(audio) - win_lo, hop_lo)
    ])
    t_broad = np.arange(len(energy_broad)) * hop_lo / sr

    # Trouver le dernier bloc de silence ≥ 0.4s
    sil_threshold = max(np.percentile(energy_broad, 10) * 4,
                        np.percentile(energy_broad, 30))
    silence_mask  = energy_broad < sil_threshold

    last_silence_start_t = None
    last_silence_end_t   = None
    run_start = None
    for i, is_sil in enumerate(silence_mask):
        if is_sil:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len_s = (i - run_start) * hop_lo / sr
                if run_len_s >= 0.4:
                    last_silence_start_t = t_broad[run_start]
                    last_silence_end_t   = t_broad[min(i, len(t_broad) - 1)]
                run_start = None
    if run_start is not None and (len(silence_mask) - run_start) * hop_lo / sr >= 0.4:
        last_silence_start_t = t_broad[run_start]
        last_silence_end_t   = t_broad[min(len(t_broad) - 1, run_start)]

    # 5. Calibrer le seuil sur le noise floor pendant le silence
    #    Les bips BMX sont 20-80× le bruit de fond → seuil = 5× noise floor
    if last_silence_start_t is not None and last_silence_end_t is not None:
        mask_sil = (t_e >= last_silence_start_t) & (t_e <= last_silence_end_t)
        if mask_sil.sum() > 5:
            # Utiliser le 50e percentile (médiane) pour éviter que le début
            # bruité de la fenêtre ne gonfle le plancher de bruit
            noise_floor = float(np.percentile(energy[mask_sil], 50))
        else:
            noise_floor = float(np.percentile(energy, 10))
        search_start_t = max(0.0, last_silence_end_t - 0.3)
        search_end_t   = last_silence_end_t + 1.5
        print(f"  [Audio] Silence [{last_silence_start_t:.2f}s→{last_silence_end_t:.2f}s] "
              f"→ recherche bips [{search_start_t:.2f}s, {search_end_t:.2f}s]")
    else:
        noise_floor    = float(np.percentile(energy, 10))
        search_start_t = 0.0
        search_end_t   = t_e[-1]

    threshold = max(0.0005, 5.0 * noise_floor)

    # 6. Extraire la fenêtre de recherche
    si = max(0, int(search_start_t / (hop / sr)))
    ei = min(len(energy), int(search_end_t / (hop / sr)))
    energy_s = energy[si:ei]
    t_s      = t_e[si:ei]

    if len(energy_s) < 20:
        return None, None, 0.0

    # 7. Détecter les pics de bips (50ms min entre pics, ignorer les gros crashs >50× noise)
    min_dist_f   = max(1, int(0.05 / (hop / sr)))
    crash_thresh = 50.0 * noise_floor
    peaks_all, _ = find_peaks(energy_s, distance=min_dist_f, height=threshold)
    # Garder uniquement les pics < crash_thresh (exclure gate drop + foule)
    peaks = peaks_all[energy_s[peaks_all] < crash_thresh]

    if len(peaks) < 3:
        print(f"  [Audio] Pas assez de pics bips ({len(peaks)}<3) — pas de bips trouvés")
        return None, None, 0.0

    # 8. Chercher la séquence la plus régulière: 4 ou 3 bips, intervalles 40-120ms
    best_seq   = None
    best_score = np.inf
    best_n     = 0

    for n in [4, 3]:
        if len(peaks) < n:
            continue
        for start in range(len(peaks) - n + 1):
            seq       = peaks[start:start + n]
            intervals = np.diff(t_s[seq])
            total_dur = t_s[seq[-1]] - t_s[seq[0]]
            if (np.all((intervals >= 0.04) & (intervals <= 0.12))
                    and total_dur < 1.0):
                score = np.std(intervals)
                if score < best_score:
                    best_score = score
                    best_seq   = seq
                    best_n     = n
        if best_seq is not None:
            break

    if best_seq is None:
        print(f"  [Audio] Aucune séquence de bips réguliers (40-120ms) trouvée")
        return None, None, 0.0

    beep_times    = [float(t_s[p]) for p in best_seq]
    mean_interval = float(np.mean(np.diff(beep_times)))

    # 9. Extrapoler le 4ème bip si seulement 3 détectés
    n_missing = 4 - best_n
    for _ in range(n_missing):
        beep_times.append(beep_times[-1] + mean_interval)

    intervals = np.diff(beep_times)
    gate_time  = beep_times[3]   # gate drop = 4ème bip

    # 10. Confiance: régularité (std) × couverture (n détectés / 4)
    reg_conf  = float(np.clip(1.0 - best_score / 0.015, 0.0, 1.0))
    n_conf    = best_n / 4.0
    confidence = reg_conf * n_conf

    tag = "extrapolé" if n_missing > 0 else "détecté"
    print(f"  [Audio] {best_n}/4 bips détectés → gate {tag}")
    print(f"  [Audio] Bips: {[f'{t:.3f}s' for t in beep_times]}")
    print(f"  [Audio] Intervalles: {[f'{i*1000:.0f}ms' for i in intervals]} "
          f"(moy={mean_interval*1000:.0f}ms, std={best_score*1000:.1f}ms)")
    print(f"  [Audio] Gate drop (4ème bip): {gate_time:.3f}s | confiance: {confidence:.0%}")

    return beep_times, gate_time, confidence


def detect_gate_drop(video_path, gate_zone_frac=0.30):
    """
    Détecte automatiquement le gate drop par différence de frames dans la zone
    haute de l'image (top 30%), où le gate et l'activité pré-départ créent
    un spike de mouvement distinct.

    Stratégie:
    - Trouver la fenêtre la plus calme de la vidéo (phase Set)
    - Baseline = mouvement médian pendant le Set
    - Gate drop = premier franchissement de 2× baseline confirmé sur 3 frames

    Nécessite au moins 3s de vidéo avec une phase Set calme identifiable.
    Pour les vidéos courtes (<3s), utiliser --gate-frame.

    Retourne (gate_frame, gate_time_s, confidence_0_to_1, motion_signal)
    """
    cap     = cv2.VideoCapture(str(video_path))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gate_h  = int(height * gate_zone_frac)
    skip    = int(fps * 1.0)   # ignorer la 1ère seconde (bruit init caméra)

    upper_scores = []
    prev_upper   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        upper = gray[:gate_h, :]
        if prev_upper is not None:
            upper_scores.append(float(np.mean(cv2.absdiff(upper, prev_upper))))
        else:
            upper_scores.append(0.0)
        prev_upper = upper
    cap.release()

    motion = np.array(upper_scores)

    # Vidéo trop courte : impossible d'établir une baseline fiable
    min_frames_needed = skip + int(fps * 1.5) + 5
    if len(motion) < min_frames_needed:
        return None, None, 0.0, motion

    # Fenêtre la plus calme = phase Set (grille immobile)
    win_frames  = max(3, int(fps * 1.5))
    roll_med    = pd.Series(motion).rolling(win_frames, center=True).median().values
    roll_med    = np.where(np.isnan(roll_med), np.nanmedian(roll_med), roll_med)
    calm_center = int(np.argmin(roll_med[skip:])) + skip
    calm_start  = max(skip, calm_center - win_frames // 2)
    calm_end    = min(len(motion) - 5, calm_center + win_frames // 2)

    if calm_end <= calm_start:
        return None, None, 0.0, motion

    baseline  = float(np.median(motion[calm_start:calm_end]))
    noise     = float(np.std(motion[calm_start:calm_end])) + 1e-8
    threshold = 2.0 * baseline
    confirm_n = 3

    gate_frame = None
    for i in range(calm_end, len(motion) - confirm_n):
        if motion[i] >= threshold:
            if all(motion[i + k] >= threshold for k in range(1, confirm_n)):
                gate_frame = i
                break

    if gate_frame is None:
        remaining = motion[calm_end:]
        if len(remaining) > 0:
            gate_frame = int(np.argmax(remaining)) + calm_end
        else:
            return None, None, 0.0, motion

    gate_time  = gate_frame / fps
    peak_val   = float(np.max(motion[gate_frame:min(gate_frame + int(fps), len(motion))]))
    snr        = (peak_val - baseline) / noise
    confidence = float(np.clip(snr / 8.0, 0.0, 1.0))

    return gate_frame, gate_time, confidence, motion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMX Start Analyzer MVP 2.0")
    parser.add_argument("video", help="Chemin vers la vidéo")
    parser.add_argument("--front-foot", choices=["L", "R"], default=None,
                        help="Pied avant du rider (L ou R)")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--gate-drop", type=float,
                       help="Temps (en secondes) où la grille commence à tomber")
    group.add_argument("--gate-frame", type=int,
                       help="Numéro de frame où la grille commence à tomber (plus précis)")
    args = parser.parse_args()

    _cap = cv2.VideoCapture(args.video)
    _fps = _cap.get(cv2.CAP_PROP_FPS)
    _cap.release()

    bip1_time     = None   # temps du 1er bip (si détecté par audio)

    if args.gate_frame is not None:
        gate_drop_time = args.gate_frame / _fps
        print(f"Gate frame {args.gate_frame} → t={gate_drop_time:.3f}s (fps={_fps:.2f})")

    elif args.gate_drop is not None:
        gate_drop_time = args.gate_drop

    else:
        # Détection automatique — audio en priorité, visuel en fallback
        print("Gate drop non spécifié — détection automatique...")

        # --- Tentative audio ---
        print("  [1/2] Analyse audio (bips 580-750Hz)...")
        beep_times, gate_time_audio, audio_conf = detect_beeps_audio(args.video)

        if beep_times is not None and audio_conf >= 0.40:
            conf_bar = "★" * int(audio_conf * 5) + "☆" * (5 - int(audio_conf * 5))
            print(f"  ✓ Audio — {len(beep_times)} bips détectés:")
            for i, bt in enumerate(beep_times, 1):
                print(f"     Bip {i}: t={bt:.3f}s")
            print(f"  → Gate drop (bip 4): t={gate_time_audio:.3f}s | {conf_bar} ({audio_conf:.0%})")
            gate_drop_time = gate_time_audio
            bip1_time      = beep_times[0]

        else:
            # --- Fallback visuel ---
            if beep_times is None:
                print(f"  ✗ Pas de piste audio")
            else:
                print(f"  ✗ Audio: confiance faible ({audio_conf:.0%})")

            print("  [2/2] Détection visuelle (mouvement zone haute)...")
            gate_frame, gate_time_visual, visual_conf, signal = detect_gate_drop(args.video)

            if gate_frame is None:
                print(f"  ✗ Vidéo trop courte pour baseline visuelle.")
                print(f"\n  ⚠️  Impossible de détecter le gate automatiquement.")
                print(f"     --gate-frame <numéro>   ou   --gate-drop <secondes>")
                raise SystemExit(1)

            conf_bar = "★" * int(visual_conf * 5) + "☆" * (5 - int(visual_conf * 5))
            print(f"  → Visuel: frame {gate_frame} | t={gate_time_visual:.3f}s | {conf_bar} ({visual_conf:.0%})")

            if visual_conf >= 0.50:
                gate_drop_time = gate_time_visual
                print(f"  ✓ Gate drop visuel utilisé")
            else:
                print(f"  ⚠️  Confiance faible. Utilise --gate-frame ou --gate-drop")
                raise SystemExit(1)

    main(args.video, front_foot=args.front_foot, gate_drop=gate_drop_time,
         bip1_time=bip1_time)