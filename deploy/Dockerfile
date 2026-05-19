# ────────────────────────────────────────────────────────────────────────
# BMX Start Analyzer — image Docker pour VPS CPU
# Python 3.12 slim + ffmpeg + torch CPU + ultralytics + YOLO pré-téléchargé
# ────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Dépendances système : ffmpeg pour la vidéo, libs OpenCV pour cv2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Étape 1 : installer torch CPU only (~200 MB au lieu de ~2 GB GPU) ────
# On l'installe AVANT le reste pour bénéficier du cache Docker
# (ça change rarement, le rebuild sera ultra-rapide quand seules les
# autres dépendances bougent).
RUN pip install --no-cache-dir \
    torch==2.11.0+cpu \
    torchvision==0.26.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Étape 2 : reste des dépendances Python ───────────────────────────────
COPY requirements.txt .
# On retire les lignes torch/torchvision pour ne pas réinstaller la version GPU
RUN grep -v -E '^(torch|torchvision)==' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

# ── Étape 3 : pré-télécharger le modèle YOLO pose ────────────────────────
# Évite que le 1er upload attende 5-10s pour télécharger 42 MB.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"

# ── Étape 4 : code de l'app ──────────────────────────────────────────────
COPY . .

# Le modèle YOLO est dans /root/.config/Ultralytics/, mais analyze.py
# fait YOLO("yolo11m-pose.pt") sans chemin absolu → on copie le fichier
# téléchargé dans /app pour qu'il soit trouvé en premier.
RUN cp /root/.config/Ultralytics/yolo11m-pose.pt /app/ 2>/dev/null \
    || cp /root/.cache/Ultralytics/yolo11m-pose.pt /app/ 2>/dev/null \
    || python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"

# Les dossiers uploads/ et output/ seront montés en volume.
RUN mkdir -p uploads output

EXPOSE 8000

# Mode prod : pas de --reload, plusieurs workers selon la RAM disponible.
# 2 workers suffisent pour usage solo + 1 user de réserve.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
