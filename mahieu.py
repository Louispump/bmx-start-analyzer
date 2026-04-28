"""
Alignement Mahieu — épaule-hanche-cheville pendant le Set.

Principe (R. Mahieu, bronze JO Paris 2024) : pour une transmission optimale
de la puissance, ces 3 points doivent être quasi colinéaires dans le plan
sagittal pendant la position de set, traduisant un dos solide.

Calcul :
  1. Pour une frame du Set : récupérer (shoulder, hip, ankle) du côté filmé
  2. Distance perpendiculaire signée du hip à la ligne shoulder→ankle,
     normalisée par la longueur de cette ligne (en %)
  3. Sign convention :
       deviation > 0  → hip vers l'AVANT du rider (sur-extension)
       deviation < 0  → hip vers l'ARRIÈRE du rider (dos arrondi / C-back)

Module utilisable :
  - depuis analyze.py via `mahieu_metric_from_keypoints(...)`
  - en standalone via `analyze_video(video_path)` qui charge YOLO,
    échantillonne le Set et retourne stats + image de debug
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Indices COCO 17 keypoints
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_HIP,      KP_R_HIP      = 11, 12
KP_L_ANKLE,    KP_R_ANKLE    = 15, 16

CONF_MIN = 0.3   # confiance minimale par keypoint pour être pris en compte

# Modèle YOLO chargé paresseusement
_YOLO = None


def _get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO("yolo11m-pose.pt")
    return _YOLO


# Référence empirique BMX (médiane des pros mesurés). Sert de cible pour le
# coaching et de centre des seuils.
PRO_REFERENCE_PCT = -27.0

# Seuils empiriques calibrés sur 22 vidéos (BMX race amateur + pro) :
#  - les pros tournent autour de −27%
#  - les amateurs tombent entre −35% et −45%
# La métrique reflète la stance BMX naturellement crouchée : 0% est
# physiquement irréaliste, on classe par rapport à la réalité du sport.
def _classify(deviation_pct: float) -> str:
    a = abs(deviation_pct)
    if a < 25:    return "excellent"      # niveau élite Mahieu
    if a < 32:    return "bon"             # niveau pro
    if a < 40:    return "moyen"
    return "faible"                         # C-back marqué


def mahieu_metric_from_points(shoulder, hip, ankle, direction: int):
    """Calcule la métrique Mahieu pour un seul triplet de points.

    `shoulder`, `hip`, `ankle` sont des couples (x, y) en pixels.
    `direction` ∈ {-1, +1} : sens du rider (+1 = face à droite).

    Retourne un dict :
      {"deviation_pct", "perp_px", "line_length_px",
       "projection_xy", "label"}
    ou None si la géométrie est dégénérée.
    """
    sh = np.asarray(shoulder, dtype=np.float64)
    hi = np.asarray(hip,      dtype=np.float64)
    an = np.asarray(ankle,    dtype=np.float64)

    sa_vec = an - sh
    sa_norm = float(np.linalg.norm(sa_vec))
    if sa_norm < 5.0:
        return None

    # Projection orthogonale du hip sur la ligne shoulder→ankle
    sh_to_hip = hi - sh
    t = float(np.dot(sh_to_hip, sa_vec) / (sa_norm * sa_norm))
    projection = sh + t * sa_vec

    disp = hi - projection                 # vecteur du hip à sa projection
    perp_px = float(np.linalg.norm(disp))  # toujours positif

    # Sens : projection du déplacement sur l'axe avant-rider (direction, 0)
    # forward = (direction, 0). dot(disp, forward) = disp_x * direction
    forward_dot = float(disp[0]) * direction
    sign = 1.0 if forward_dot > 0 else -1.0

    deviation_pct = sign * perp_px / sa_norm * 100.0

    return {
        "deviation_pct":   round(deviation_pct, 2),
        "perp_px":         round(perp_px,        1),
        "line_length_px":  round(sa_norm,        1),
        "projection_xy":   (float(projection[0]), float(projection[1])),
        "label":           _classify(deviation_pct),
        "shoulder":        (float(sh[0]), float(sh[1])),
        "hip":             (float(hi[0]), float(hi[1])),
        "ankle":           (float(an[0]), float(an[1])),
    }


def detect_side_and_direction(kpts: np.ndarray):
    """Décide quel côté analyser (L/R) à partir d'une frame de keypoints.

    Heuristique : on prend le côté avec la confiance moyenne la plus élevée
    sur (shoulder, hip, ankle).

    Direction : nez vs centre des hanches. nez_x > hip_center_x → +1 (droite).

    Retourne (side, direction) ou (None, None) si pas assez de confiance.
    """
    if kpts is None or kpts.shape != (17, 3):
        return None, None
    L_conf = np.mean([kpts[KP_L_SHOULDER, 2], kpts[KP_L_HIP, 2], kpts[KP_L_ANKLE, 2]])
    R_conf = np.mean([kpts[KP_R_SHOULDER, 2], kpts[KP_R_HIP, 2], kpts[KP_R_ANKLE, 2]])
    if max(L_conf, R_conf) < CONF_MIN:
        return None, None
    side = "L" if L_conf >= R_conf else "R"

    # Direction
    nose_x, nose_conf = kpts[0, 0], kpts[0, 2]
    L_hip_x, L_hip_conf = kpts[KP_L_HIP, 0], kpts[KP_L_HIP, 2]
    R_hip_x, R_hip_conf = kpts[KP_R_HIP, 0], kpts[KP_R_HIP, 2]
    if L_hip_conf > CONF_MIN and R_hip_conf > CONF_MIN and nose_conf > CONF_MIN:
        hip_cx = (L_hip_x + R_hip_x) / 2.0
        direction = 1 if nose_x > hip_cx else -1
    else:
        direction = 1   # défaut
    return side, direction


def compute_on_keypoints(kpts: np.ndarray, side: str, direction: int):
    """Wrapper : calcule la métrique depuis un (17,3) array et un side L/R."""
    if side == "L":
        sh = kpts[KP_L_SHOULDER, :2]
        hi = kpts[KP_L_HIP,      :2]
        an = kpts[KP_L_ANKLE,    :2]
        confs = (kpts[KP_L_SHOULDER, 2], kpts[KP_L_HIP, 2], kpts[KP_L_ANKLE, 2])
    else:
        sh = kpts[KP_R_SHOULDER, :2]
        hi = kpts[KP_R_HIP,      :2]
        an = kpts[KP_R_ANKLE,    :2]
        confs = (kpts[KP_R_SHOULDER, 2], kpts[KP_R_HIP, 2], kpts[KP_R_ANKLE, 2])
    if min(confs) < CONF_MIN:
        return None
    return mahieu_metric_from_points(tuple(sh), tuple(hi), tuple(an), direction)


def analyze_video(video_path: Path, set_fraction: float = 0.30,
                   sample_frames: int = 8) -> dict:
    """Échantillonne quelques frames du Set, calcule Mahieu sur chacune et
    retourne la médiane + la frame la plus représentative pour debug."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"detected": False, "reason": "cannot_open_video"}
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total < 5:
        cap.release()
        return {"detected": False, "reason": "video_too_short"}

    last_set_frame = max(2, int(n_total * set_fraction))
    sample_idx = set(np.linspace(0, last_set_frame - 1, sample_frames).astype(int).tolist())
    last_idx   = max(sample_idx)

    try:
        model = _get_yolo()
    except Exception as e:
        cap.release()
        return {"detected": False, "reason": f"yolo_load_failed:{e}"}

    samples = []  # list of {"frame": int, "metric": dict, "side": str, "direction": int}
    fi = 0
    while fi <= last_idx:
        if fi in sample_idx:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, verbose=False, imgsz=640, conf=0.4,
                                     device="mps")
            if results and len(results[0].boxes) > 0 and results[0].keypoints is not None:
                # Plus grande personne
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                i_main = int(np.argmax(areas))
                kpts = results[0].keypoints.data.cpu().numpy()[i_main]
                side, direction = detect_side_and_direction(kpts)
                if side is not None:
                    metric = compute_on_keypoints(kpts, side, direction)
                    if metric is not None:
                        samples.append({
                            "frame":     int(fi),
                            "metric":    metric,
                            "side":      side,
                            "direction": int(direction),
                        })
        else:
            if not cap.grab():
                break
        fi += 1
    cap.release()

    if not samples:
        return {"detected": False, "reason": "no_valid_pose_frames",
                "frames_sampled": len(sample_idx)}

    deviations = np.array([s["metric"]["deviation_pct"] for s in samples])
    median_dev = float(np.median(deviations))

    # Frame représentative = celle dont la déviation est la plus proche de la médiane
    best = min(samples, key=lambda s: abs(s["metric"]["deviation_pct"] - median_dev))

    return {
        "detected":         True,
        "deviation_pct":    round(median_dev, 2),
        "label":            _classify(median_dev),
        "n_detections":     len(samples),
        "frames_sampled":   len(sample_idx),
        "all_deviations":   [round(d, 2) for d in deviations.tolist()],
        "best_frame":       best,
    }


def render_debug_frame(video_path: Path, frame_idx: int, metric: dict,
                       side: str, output_path: Path) -> bool:
    """Rend une frame de la vidéo avec ligne shoulder→ankle + position
    réelle vs projetée du hip. Renvoie True si succès."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    sh = tuple(map(int, metric["shoulder"]))
    hi = tuple(map(int, metric["hip"]))
    an = tuple(map(int, metric["ankle"]))
    proj = tuple(map(int, metric["projection_xy"]))

    # Ligne épaule → cheville (jaune épais)
    cv2.line(frame, sh, an, (0, 220, 220), 4)
    # Marqueurs sur les keypoints
    cv2.circle(frame, sh, 8, (255, 255, 255), -1)
    cv2.circle(frame, sh, 8, (0, 0, 0), 2)
    cv2.circle(frame, an, 8, (255, 255, 255), -1)
    cv2.circle(frame, an, 8, (0, 0, 0), 2)
    # Hip réel — couleur selon les seuils BMX recalibrés
    abs_dev = abs(metric["deviation_pct"])
    if abs_dev < 32:
        hip_color = (0, 200, 0)
    elif abs_dev < 40:
        hip_color = (0, 165, 255)
    else:
        hip_color = (0, 0, 255)
    cv2.circle(frame, hi, 10, hip_color, -1)
    cv2.circle(frame, hi, 10, (0, 0, 0), 2)
    # Trait du hip réel à sa projection (rouge fin)
    cv2.line(frame, hi, proj, (0, 0, 255), 2)
    cv2.circle(frame, proj, 6, (0, 0, 255), 2)

    # Annotations texte
    h_img = frame.shape[0]
    label_text = f"{metric['label']} ({metric['deviation_pct']:+.1f}%)"
    cv2.rectangle(frame, (10, h_img - 60), (10 + 8 + 12 * len(label_text), h_img - 20),
                  (0, 0, 0), -1)
    cv2.putText(frame, label_text, (15, h_img - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"side={side}", (15, h_img - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), frame))


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python mahieu.py <video>")
        sys.exit(1)
    res = analyze_video(Path(sys.argv[1]))
    print(json.dumps(res, indent=2, default=str))
