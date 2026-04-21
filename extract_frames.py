"""
Outil helper pour identifier le gate drop.
Extrait les frames entre deux temps (en secondes) vers output/frames_<video>/
Usage: python extract_frames.py videos/ta_video.mp4 1.5 2.5
"""

import sys
import cv2
from pathlib import Path


def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_frames.py <video> <t_start> <t_end>")
        print("Exemple: python extract_frames.py videos/amateur1.mp4 1.5 2.5")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    t_start = float(sys.argv[2])
    t_end = float(sys.argv[3])

    if not video_path.exists():
        print(f"ERREUR: vidéo introuvable: {video_path}")
        sys.exit(1)

    out_dir = Path("output") / f"frames_{video_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    f_start = int(t_start * fps)
    f_end = int(t_end * fps)

    print(f"Vidéo: {video_path.name} @ {fps:.1f}fps")
    print(f"Extraction frames {f_start} à {f_end} (soit t={t_start}s à t={t_end}s)")
    print(f"Dossier de sortie: {out_dir}/")

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    for f_idx in range(f_start, f_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        t = f_idx / fps
        label = f"Frame {f_idx} | t={t:.3f}s"
        cv2.rectangle(frame, (10, 10), (500, 60), (0, 0, 0), -1)
        cv2.putText(frame, label, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        filename = f"frame_{f_idx:04d}_t{t:.3f}.jpg"
        cv2.imwrite(str(out_dir / filename), frame)

    cap.release()
    print(f"\n✓ {f_end - f_start + 1} frames extraites")
    print(f"Ouvre le dossier: open {out_dir}")


if __name__ == "__main__":
    main()