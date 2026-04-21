"""
BMX Start Analyzer - MVP #1.4
Pose estimation + tracking + lissage + détection auto du côté filmé
+ support du pied avant pour analyse biomécaniquement correcte

Usage:
  python analyze.py videos/nom_video.mp4
  python analyze.py videos/nom_video.mp4 --front-foot L
  python analyze.py videos/nom_video.mp4 --front-foot R
"""

import sys
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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
    """Détecte le côté filmé par confiance moyenne des landmarks."""
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
    """Interpolation des petits trous + lissage Savitzky-Golay."""
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


def main(video_path, front_foot=None):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERREUR: fichier introuvable: {video_path}")
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(OUTPUT_DIR / f"{video_name}_annotated.mp4"),
        fourcc, fps, (width, height)
    )

    # === PASSE 1: Tracking ===
    print("Pass 1: tracking all persons...")
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
    print(f"Rider principal sélectionné: track_id = {main_tid}")

    # === Détection du côté filmé ===
    print("Detecting camera side...")
    camera_side = detect_visible_side(all_tracks, main_tid)
    camera_side_name = "GAUCHE" if camera_side == "L" else "DROIT"
    print(f"  → Côté filmé: {camera_side_name}")

    # Déterminer quel côté analyser
    if front_foot is not None:
        if front_foot != camera_side:
            print(f"  ⚠️  ATTENTION: pied avant déclaré ({front_foot}) "
                  f"≠ côté filmé ({camera_side})")
            print(f"  Le pied avant est caché par le vélo — l'analyse sera moins fiable.")
            print(f"  Recommandation: refilmer du côté {front_foot}.")
        side = front_foot
        side_name = "GAUCHE" if side == "L" else "DROIT"
        print(f"  Pied avant analysé: {side_name}")
    else:
        side = camera_side
        side_name = camera_side_name
        print(f"  (Pied avant non spécifié — analyse du côté filmé par défaut)")

    # Points du côté choisi
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
    n_frames_with_rider = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        row = {"frame": frame_idx, "time": frame_idx / fps}
        frame_tracks = all_tracks.get(frame_idx, {})
        if main_tid in frame_tracks:
            n_frames_with_rider += 1
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

    # === Lissage ===
    print("Smoothing landmarks...")
    for name in KEYPOINTS.values():
        df[f"{name}_x"] = smooth_series(df[f"{name}_x"].values)
        df[f"{name}_y"] = smooth_series(df[f"{name}_y"].values)

    # === Calcul des angles ===
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
    df["side_analyzed"] = side
    df["camera_side"] = camera_side
    df["front_foot"] = front_foot if front_foot else "auto"

    df.to_csv(OUTPUT_DIR / f"{video_name}_landmarks.csv", index=False)

    # === PASSE 3: Rendu vidéo ===
    print("Pass 3: rendering annotated video...")
    cap = cv2.VideoCapture(str(video_path))
    if side == "R":
        segments = [(6, 8), (8, 10),
                    (6, 12), (12, 14), (14, 16),
                    (5, 6), (11, 12), (5, 11)]
        points_to_draw = [6, 8, 10, 12, 14, 16, 0, 5, 11]
    else:
        segments = [(5, 7), (7, 9),
                    (5, 11), (11, 13), (13, 15),
                    (5, 6), (11, 12), (6, 12)]
        points_to_draw = [5, 7, 9, 11, 13, 15, 0, 6, 12]

    idx_to_name = {i: n for i, n in KEYPOINTS.items()}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        row = df.iloc[frame_idx]
        for idx in points_to_draw:
            name = idx_to_name[idx]
            x, y = row[f"{name}_x"], row[f"{name}_y"]
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        for s, e in segments:
            s_name = idx_to_name[s]
            e_name = idx_to_name[e]
            x1, y1 = row[f"{s_name}_x"], row[f"{s_name}_y"]
            x2, y2 = row[f"{e_name}_x"], row[f"{e_name}_y"]
            if not any(np.isnan(v) for v in [x1, y1, x2, y2]):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (0, 255, 0), 2)
        info = f"Rider {main_tid} | Analyse cote: {side_name}"
        cv2.putText(frame, info, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if front_foot:
            foot_info = f"Pied avant: {side_name}"
            cv2.putText(frame, foot_info, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out_video.write(frame)
        frame_idx += 1
    cap.release()
    out_video.release()

    # === Graphique ===
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    plot_defs = [
        (axes[0], "knee_angle",  "Angle du genou (°)"),
        (axes[1], "hip_angle",   "Angle de la hanche (°)"),
        (axes[2], "elbow_angle", "Angle du coude (°)"),
    ]
    for ax, col, title in plot_defs:
        ax.plot(df["time"], df[col], color='#1f77b4', linewidth=1.8)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Temps (s)")
    title_suffix = f"pied avant {side_name}" if front_foot else f"côté filmé {side_name}"
    fig.suptitle(f"Angles articulaires — {video_name} | {title_suffix}",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{video_name}_angles.png", dpi=120)
    plt.close()

    # === Stats ===
    tracking_rate = n_frames_with_rider / n_frames * 100
    print(f"\n=== STATS pour {video_name} ===")
    print(f"Côté caméra: {camera_side_name}")
    print(f"Côté analysé: {side_name} {'(pied avant)' if front_foot else '(côté filmé)'}")
    print(f"Rider principal suivi: {n_frames_with_rider}/{n_frames} ({tracking_rate:.1f}%)")
    print(f"\nAngles articulaires:")
    for col, label, bench in [("knee_angle", "Genou", "93°±12°"),
                              ("hip_angle", "Hanche", "62°±11°"),
                              ("elbow_angle", "Coude", "47°±15°")]:
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  {label}: moy={vals.mean():.1f}° "
                  f"| min={vals.min():.1f}° | max={vals.max():.1f}° "
                  f"| amplitude={vals.max()-vals.min():.1f}° "
                  f"(benchmark Grigg élite: {bench})")
    print(f"\nFichiers:")
    print(f"  output/{video_name}_annotated.mp4")
    print(f"  output/{video_name}_landmarks.csv")
    print(f"  output/{video_name}_angles.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMX Start Analyzer")
    parser.add_argument("video", help="Chemin vers la vidéo (ex: videos/test.mp4)")
    parser.add_argument("--front-foot", choices=["L", "R"], default=None,
                        help="Pied avant du rider: L (gauche) ou R (droit). "
                             "Si omis, le côté filmé est analysé par défaut.")
    args = parser.parse_args()
    main(args.video, front_foot=args.front_foot)