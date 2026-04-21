"""
BMX Start Analyzer - MVP #2.0
Pose estimation + tracking + segmentation en phases (Kalichová)

Usage:
  python analyze.py videos/nom_video.mp4 --gate-drop 1.8
  python analyze.py videos/nom_video.mp4 --gate-drop 1.8 --front-foot L
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


def detect_first_movement(df, gate_drop_time, threshold_deg=5.0, search_window_s=0.6):
    """
    Détecte la première frame où l'angle du genou change significativement
    par rapport à la valeur au gate drop.
    """
    gate_idx = (df["time"] - gate_drop_time).abs().idxmin()
    knee_at_gate = df.loc[gate_idx, "knee_angle"]
    if np.isnan(knee_at_gate):
        return gate_idx
    search_end = df[df["time"] <= gate_drop_time + search_window_s].index.max()
    for i in range(gate_idx, min(search_end + 1, len(df))):
        if not np.isnan(df.loc[i, "knee_angle"]):
            if abs(df.loc[i, "knee_angle"] - knee_at_gate) > threshold_deg:
                return i
    return gate_idx


def detect_crank_events(df, first_move_idx, ankle_col, n_cranks=3):
    """
    Détecte les points morts haut/bas de la pédale du pied analysé en 
    cherchant les pics et creux de la POSITION Y de la cheville.
    
    Signal: y_ankle (cheville du pied analysé)
    - y MINIMUM = cheville au plus haut = pédale à 12h (point mort haut)
    - y MAXIMUM = cheville au plus bas = pédale à 6h (point mort bas)
    
    Séquence typique en BMX départ (pied avant sur pédale haute):
    - Début: pédale avant vers 12h (y_ankle petit) 
    - Push 1: pédale descend vers 6h (y_ankle augmente → MAX)
    - Pull 1: pédale remonte vers 12h (y_ankle diminue → MIN)
    - Push 2: redescend (y_ankle augmente → MAX)
    
    Retourne la liste d'indices: [fin_push1, fin_pull1, fin_push2]
    """
    sub = df.iloc[first_move_idx:].copy().reset_index(drop=False)
    y_ankle = sub[ankle_col].values

    # Estimation fps
    valid_times = sub["time"].dropna().values
    if len(valid_times) < 2:
        return []
    fps_approx = 1 / np.median(np.diff(valid_times))
    min_dist = max(3, int(0.12 * fps_approx))  # min 120ms entre événements

    # Maxima de y (point mort bas de la pédale) 
    maxima, _ = find_peaks(y_ankle, distance=min_dist, prominence=10)
    # Minima de y (point mort haut de la pédale)
    minima, _ = find_peaks(-y_ankle, distance=min_dist, prominence=10)

    # Séquence attendue: max → min → max (push-pull-push)
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

    return events


def segment_phases(df, gate_drop_time, ankle_col):
    """
    Segmente la vidéo en phases selon Kalichová.
    Retourne un dict {phase_name: (start_idx, end_idx)}
    """
    phases = {}
    gate_idx = (df["time"] - gate_drop_time).abs().idxmin()
    phases["Set"] = (0, gate_idx)

    first_move_idx = detect_first_movement(df, gate_drop_time)
    phases["Reaction"] = (gate_idx, first_move_idx)

    crank_events = detect_crank_events(df, first_move_idx, ankle_col)
    if len(crank_events) >= 1:
        phases["Push 1"] = (first_move_idx, crank_events[0])
    if len(crank_events) >= 2:
        phases["Pull 1"] = (crank_events[0], crank_events[1])
    if len(crank_events) >= 3:
        phases["Push 2"] = (crank_events[1], crank_events[2])
        phases["Post"] = (crank_events[2], len(df) - 1)
    else:
        # Si on n'a pas détecté tous les cranks, la dernière zone est "Post"
        last_idx = crank_events[-1] if crank_events else first_move_idx
        phases["Post"] = (last_idx, len(df) - 1)

    return phases, first_move_idx

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
        return None
    
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
    
    return front

def main(video_path, front_foot=None, gate_drop=None):
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
    detected_front = detect_front_foot(all_tracks, main_tid)

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

    # === Segmentation en phases ===
    print("Segmenting phases (Kalichová)...")
    ankle_col = f"{ankle_k}_y"  # ex: "R_ankle_y" ou "L_ankle_y"
    phases, first_move_idx = segment_phases(df, gate_drop, ankle_col)
    
    # Étiquette de phase pour chaque frame dans le CSV
    df["phase"] = "Unknown"
    for phase_name, (start, end) in phases.items():
        df.loc[start:end, "phase"] = phase_name

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
    react_time = df.loc[first_move_idx, "time"] - gate_drop
    print(f"\n=== TEMPS DE RÉACTION ===")
    print(f"  Premier mouvement détecté à t={df.loc[first_move_idx, 'time']:.2f}s")
    print(f"  Temps de réaction (depuis gate drop): {react_time*1000:.0f}ms")
    if react_time < 0:
        print(f"  ⚠️  Temps de réaction négatif = rider anticipait sur les bips (normal en BMX pro)")

    print(f"\nFichiers: output/{video_name}_annotated.mp4 + _landmarks.csv + _angles.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMX Start Analyzer MVP 2.0")
    parser.add_argument("video", help="Chemin vers la vidéo")
    parser.add_argument("--front-foot", choices=["L", "R"], default=None,
                        help="Pied avant du rider (L ou R)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gate-drop", type=float,
                       help="Temps (en secondes) où la grille commence à tomber")
    group.add_argument("--gate-frame", type=int,
                       help="Numéro de frame où la grille commence à tomber (plus précis)")
    args = parser.parse_args()

    # Si --gate-frame est utilisé, on convertit en temps
    if args.gate_frame is not None:
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(args.video)
        _fps = _cap.get(_cv2.CAP_PROP_FPS)
        _cap.release()
        gate_drop_time = args.gate_frame / _fps
        print(f"Gate frame {args.gate_frame} → t={gate_drop_time:.3f}s (fps={_fps:.2f})")
    else:
        gate_drop_time = args.gate_drop

    main(args.video, front_foot=args.front_foot, gate_drop=gate_drop_time)