# BMX Start Analyzer - Roadmap et documentation

## Vision

Devenir l'outil de référence pour l'analyse biomécanique automatisée du 
départ en BMX race. Combler un vide de marché identifié: il n'existe 
actuellement aucun outil qui combine analyse vidéo automatique, spécificité 
BMX, base scientifique, accessibilité web/mobile, et feedback actionnable.

## Positionnement

Les outils existants (Kinovea, Dartfish, Hudl) sont génériques et demandent 
un travail manuel considérable. Les systèmes pro (G-Cog, PerfAnalytics) 
sont inaccessibles au grand public. Le BMX race amateur mondial 
(~50-100k pratiquants sérieux) reçoit actuellement zéro feedback 
biomécanique automatisé.

## Références scientifiques (piliers techniques)

### Segmentation des phases
- Kalichová, M. et al. (2013). "Biomechanics Analysis of Bicross Start."
  International Journal of Medical, Health, Pharmaceutical and Biomedical 
  Engineering.
  → Framework des 5 phases utilisé dans notre segmentation.

### Kinématique et typologie du départ
- Grigg, J. (2020). "A biomechanical analysis of the BMX SX gate start." 
  Thèse, Bond University.
  → Typologie position de set (upright/back/angled) + trajectoire moyeu 
    (hairpin/up-and-over/demi-cercle). Position "back" corrélée aux 
    départs rapides.

- Grigg, J. et al. (2018). "Kinematics of the BMX SX gate start action." 
  ISBS Proceedings Archive, 36(1), 85-89.
  → Benchmarks d'amplitudes articulaires sur 10 athlètes élites:
    tronc 39°±6°, hanche 62°±11°, genou 93°±12°, cheville 58°±14°, 
    épaule 87°±7°, coude 47°±15°.

- Grigg, J. (2017). "Literature Review: Kinematics of the BMX Supercross 
  Gate Start." Journal of Science and Cycling, 6(1), 3-10.
  → Revue de littérature de référence, identifie les lacunes de recherche.

### Méthodologie mesure
- Grigg, J. et al. (2017). "The validity and intra-tester reliability of 
  markerless motion capture to analyse kinematics of the BMX Supercross 
  gate start."
  → Validation méthodologique Kinovea + GoPro à 120fps 720p.

### Recherche appliquée (labo de référence)
- Institut Pprime (CNRS/Université de Poitiers) - Mathieu Domalain
- Projet PerfAnalytics (INRIA Grenoble - Lionel Reveret)
- CAIPS (centre d'analyse d'images sportives)
- Partenariat avec Fédération Française de Cyclisme pour Paris 2024

### Coaching élite (source terrain)
- Romain Mahieu (bronze JO Paris 2024): principe d'alignement 
  épaules-bassin-chevilles, importance du dos fort pour transmission 
  de puissance.

### Observations statistiques clés
- 80% des coureurs qui mènent au premier saut gagnent le tour (analyse 
  équipe USA avant JO 2008, source bmxultra.com).

## Stack technique actuelle

- Python 3.12 + venv
- YOLOv11-Pose (Ultralytics) pour détection/tracking des personnes
- ByteTrack pour tracking multi-personnes avec ID persistants
- OpenCV pour I/O vidéo
- scipy (Savitzky-Golay + find_peaks) pour traitement signal
- pandas pour manipulation des données
- matplotlib pour visualisations
- Modèle: yolo11m-pose.pt (~40 MB)
- Device: Apple Silicon MPS

## Architecture cible (app web)