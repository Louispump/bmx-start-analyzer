# Prompt pour Claude chat — Module de coaching biomécanique avancé (BMX Start Analyzer)

> Copie tout ce qui suit (à partir de « Tu es un… ») dans le chat Claude.

---

Tu es un ingénieur biomécanique + développeur Python senior. Tu m'aides à coder le **module d'analyse coach le plus poussé** d'une app d'analyse vidéo du **départ en BMX race**. Je vais te donner tout le contexte technique et scientifique nécessaire — tu n'as pas accès au repo, donc base-toi uniquement sur ce document.

## 1. L'app et la stack

**BMX Start Analyzer** : outil web qui analyse des vidéos de départ BMX (la phase grille→première poussée, ~3 secondes).

- Backend : **FastAPI** (Python 3.12)
- Frontend : **Jinja2 + JavaScript vanilla** (pas de framework)
- Pose : **YOLOv11-Pose** (Ultralytics) → 17 keypoints COCO, lissés Savitzky-Golay
- Vidéo : OpenCV + ffmpeg
- Thème UI : **dark navy avec accent cyan `#00d4ff`**
- Langue : **tout en français**, vocabulaire de coach (pas de jargon scientifique dans le texte affiché)

L'app tourne déjà : upload vidéo → marquage du gate drop → analyse → page de résultats avec vidéo annotée, timeline des phases, comparaison côte-à-côte avec un pro.

## 2. Les données disponibles pour CHAQUE vidéo analysée

### Fichier CSV de landmarks (`{video}_landmarks.csv`)
Une ligne par frame. Colonnes :
```
frame, time, nose_x, nose_y, nose_conf,
L_shoulder_x/y/conf, R_shoulder_x/y/conf,
L_elbow_x/y/conf, R_elbow_x/y/conf,
L_wrist_x/y/conf, R_wrist_x/y/conf,
L_hip_x/y/conf, R_hip_x/y/conf,
L_knee_x/y/conf, R_knee_x/y/conf,
L_ankle_x/y/conf, R_ankle_x/y/conf,
knee_angle, phase
```
- Coordonnées en **pixels image** (origine en haut à gauche, y vers le bas).
- `conf` = confiance de détection 0–1 (filtrer < ~0.3).

### Métadonnées (`results` dict)
```python
{
  "fps": float, "total_frames": int, "duration_s": float,
  "gate_drop_t": float,      # instant T=0 (chute des grilles), en secondes
  "front_foot": "L" | "R",   # pied avant détecté
  "phases": [                # segmentation Kalichová
     {"name": "Set",     "start_t": ..., "end_t": ...},
     {"name": "Reaction","start_t": ..., "end_t": ...},
     {"name": "Push 1",  "start_t": ..., "end_t": ...},
     {"name": "Pull 1",  "start_t": ..., "end_t": ...},
     {"name": "Push 2",  "start_t": ..., "end_t": ...},
     {"name": "Post",    "start_t": ..., "end_t": ...},
  ],
  "reaction": {"first_move_t": ..., "from_gate_ms": ...},
}
```

### Convention de calcul d'angle (déjà utilisée partout)
```python
def calculate_angle(p1, p2, p3):
    """Angle intérieur au sommet p2, en degrés. 180° = aligné/tendu, 90° = plié."""
    v1, v2 = np.array(p1)-np.array(p2), np.array(p3)-np.array(p2)
    cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))
```
Direction du rider : `+1` s'il regarde vers la droite de l'image, `-1` sinon
(via `nose_x` vs centre des hanches `(L_hip_x+R_hip_x)/2`).

## 3. Ce qui est DÉJÀ codé — à NE PAS refaire, mais à compléter/articuler

1. **Module Explosivité (`/explosivity/{job_id}`)** : vitesses angulaires ω (°/s)
   hanche/genou/cheville sur [gate, gate+1s], séquence proximale-distale
   (hanche→genou→cheville), comparaison au référentiel perso de l'athlète,
   verdict coach + conseil.

2. **Module Posture v1 (`/posture/{job_id}`)** : consignes par phase, mesures
   caméra-invariantes :
   - SET (gate−0.05s) : position bassin vs pied avant, charge de la jambe
     avant (angle genou), inclinaison du buste.
   - PUSH 1 : extension complète de la jambe + alignement épaules-bassin-
     chevilles (Mahieu) au pic de poussée.
   - Seuils en constantes documentées (ex. `SET_KNEE_TOO_STRAIGHT=152`),
     première passe à calibrer.

**Ce module v1 est volontairement minimal. Je veux que tu le pousses BEAUCOUP plus loin** (voir §5).

## 4. Le cadre scientifique à intégrer À FOND (mes « docs de maîtrise »)

### Benchmarks d'amplitude articulaire — Grigg 2018
*(n=10, élites masculins Supercross, méthode markerless validée ±2°, plan sagittal)*

| Articulation | Amplitude (°) | SD (°) |
|---|---|---|
| Tronc    | 39 | ±6  |
| Hanche   | 62 | ±11 |
| Genou    | 93 | ±12 |
| Cheville | 58 | ±14 |
| Épaule   | 87 | ±7  |
| Coude    | 47 | ±15 |

⚠️ Limites à respecter : échantillon faible, 100% masculins élites SX, plan
sagittal seulement. **À présenter avec intervalles de confiance larges** et
étiqueté « repère élites SX, n=10 », jamais comme une vérité absolue.

### Typologie de la position de set — Grigg 2020
- **Back position** (la plus performante en moyenne) : bassin reculé / centre
  de masse vers l'arrière → favorise la trajectoire « hairpin » du moyeu.
- **Upright position** (la moins efficace) : COM trop haut.
- **Angled position** : intermédiaire.
→ Classer automatiquement via angle du tronc vs verticale + position du bassin
  relative au pied avant.

### Trajectoire du moyeu avant — Grigg 2020
Classification : **hairpin** / **up-and-over** / **demi-cercle**. La « hairpin »
est associée aux départs rapides. (Si trop dur à tracker en vidéo monoculaire,
note-le comme piste future plutôt que de l'inventer.)

### Alignement Mahieu (Romain Mahieu, bronze JO Paris 2024)
Colinéarité **épaules–bassin–chevilles** dans le plan sagittal **au pic
d'extension** (PAS en set, où le corps est volontairement plié). Le dos sert de
courroie de transmission : sans gainage, la puissance des jambes se dissipe dans
le tronc. Feedback directement actionnable.

### Séquence et timing — Gross 2017
- **Countermovement précoce** (retournement avant la chute du gate) =
  déterminant #1 de performance.
- Extension **hanche AVANT genou AVANT cheville** (chaîne proximale→distale).
- Au 1er coup de pédale : couple ET vitesse angulaire élevés simultanément.
- Vitesses d'extension comparables à un saut balistique chargé
  (~400–700°/s au genou chez les élites).

### Segmentation en phases — Kalichová 2013
Les 5 phases (Set, Reaction, Push 1, Pull 1, Push 2…) sont déjà segmentées dans
`results["phases"]`. Tu peux analyser chaque phase indépendamment.

## 5. CE QUE JE VEUX QUE TU CODES

Le **diagnostic biomécanique le plus complet possible, basé sur les angles**,
qui se comporte comme un **coach expert** : il regarde chaque phase, compare aux
repères de la littérature, et sort des **consignes précises et actionnables**.

Concrètement, va plus loin que la v1 sur ces axes :

1. **Analyse multi-phase complète** : pour Set, Push 1, Pull 1 (et plus si
   pertinent), calcule les angles clés (tronc, hanche, genou, cheville, épaule,
   coude côté pied avant) et leur évolution.

2. **Comparaison aux benchmarks Grigg** avec intervalles de confiance : pour
   chaque articulation, dire si le rider est dans / au-dessus / en-dessous de la
   plage élite, en gardant la prudence (n=10, etc.).

3. **Classification automatique de la position de set** (upright/back/angled)
   avec explication.

4. **Vérification de l'alignement Mahieu** au pic d'extension de Push 1
   (et Push 2 si dispo).

5. **Amplitudes de mouvement (ROM)** par articulation sur la phase de poussée
   (max − min de l'angle), comparées aux amplitudes Grigg.

6. **Détection de patterns/défauts** : buste qui se redresse trop tôt, hanche
   qui n'a plus de course, asymétrie avant/arrière, extension incomplète, etc.

7. **Consignes coach précises par phase**, du type :
   - « En set : recule ton bassin et plie davantage ta jambe avant. »
   - « Au push 1 : avance ton bassin en tirant sur le guidon, garde le dos gainé. »
   - « Ton genou ne se déplie qu'à X° (les élites vont à ~Y°), va chercher
     l'extension complète. »

## 6. Style et présentation (important)

- **Texte principal en langage coach**, zéro jargon (« coup de pédale » pas
  « ω genou », « ouverture des hanches » pas « extension de hanche »).
- **Chiffres techniques + sources** regroupés dans un repli « Détails / chiffres »
  (balise `<details>`), jamais dans la phrase principale.
- **Par phase** : une carte par phase (Set, Push 1, …) avec un badge de verdict
  (✅ OK / 🔧 à corriger / ⚠️ attention) et la liste des consignes.
- Chaque consigne = `{type, text (coach), why (explication courte)}`.
- Thème : dark navy, accent cyan `#00d4ff`, cohérent avec l'existant.

## 7. Contraintes de fiabilité (à respecter absolument)

- **Doit fonctionner sur une seule vidéo**, sans vidéo de pro.
- **Privilégier les mesures caméra-invariantes** : compare des positions du corps
  ENTRE ELLES dans la même image (angles de membres, alignement de segments,
  position relative bassin/pied), PAS des valeurs absolues sensibles à l'angle
  de caméra. Hypothèse : filmé approximativement de profil — mentionne-le.
- **Honnêteté** : si une mesure est peu fiable (ex. trajectoire du moyeu en
  monoculaire), dis-le ou marque-la « indicatif », n'invente pas une précision
  que tu n'as pas.
- **Tous les seuils** en constantes nommées et documentées en haut du module,
  faciles à calibrer (ce sont des premières estimations à ajuster sur de vraies
  vidéos).
- Robustesse aux trous : filtrer les keypoints `conf` faible, gérer les NaN,
  utiliser la médiane sur une petite fenêtre plutôt qu'une frame isolée.

## 8. Livrables attendus

1. **Le(s) fonction(s) Python** d'analyse (calcul + génération des consignes),
   bien commentées, avec les constantes de seuils en tête.
2. **Un endpoint FastAPI** `GET /coaching/{job_id}` qui renvoie le diagnostic
   en JSON (structure que tu proposes, claire et versionnée).
3. **La carte frontend** (HTML + JS vanilla + CSS) pour afficher le diagnostic
   sur la page de résultats, dans le thème dark navy / cyan.
4. **Un tableau récapitulatif** des seuils choisis avec leur justification
   (source + valeur de départ), pour qu'on puisse les calibrer ensemble.

Commence par me proposer **la structure JSON de sortie** et **le plan des mesures
par phase** avant d'écrire tout le code, qu'on valide l'architecture. Pose-moi
des questions si un repère biomécanique te manque.
