# BMX Start Analyzer

Outil d'analyse biomécanique automatique du départ en BMX race, basé sur la vision par ordinateur et la littérature scientifique (Grigg, Kalichová).

## Fonctionnalités

- Détection et tracking automatique du rider principal (YOLOv11-Pose + ByteTrack)
- Détection automatique du côté filmé
- Support du pied avant du rider pour une analyse biomécaniquement correcte
- Lissage des landmarks (Savitzky-Golay + interpolation)
- Calcul des angles articulaires clés: genou, hanche, coude
- Comparaison aux benchmarks de la recherche scientifique
- Export: vidéo annotée, CSV de données, graphiques d'angles

## Prérequis

- macOS (testé sur Apple Silicon)
- Python 3.12+
- ffmpeg

## Installation

```bash
git clone https://github.com/Louispump/bmx-start-analyzer.git
cd bmx-start-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir videos output
```

## Utilisation

Place une vidéo de départ BMX dans `videos/`, puis:

```bash
# Analyse automatique (détection du côté filmé)
python analyze.py videos/ta_video.mp4

# Analyse avec pied avant spécifié
python analyze.py videos/ta_video.mp4 --front-foot L
python analyze.py videos/ta_video.mp4 --front-foot R
```

## Sorties

Dans le dossier `output/` :
- `<nom>_annotated.mp4` — vidéo avec squelette overlayé
- `<nom>_landmarks.csv` — données brutes frame par frame
- `<nom>_angles.png` — graphique des angles articulaires

## Références scientifiques

- Grigg, J. (2020). *A biomechanical analysis of the BMX SX gate start.* Bond University.
- Grigg, J. et al. (2018). *Kinematics of the BMX SX gate start action.* ISBS Proceedings Archive, 36(1), 85-89.
- Kalichová, M. et al. (2013). *Biomechanics Analysis of Bicross Start.*

## État du projet

En développement actif.

## Auteur

Louis-Edouard Dube