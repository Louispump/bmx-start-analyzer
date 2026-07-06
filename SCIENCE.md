# HOLESHOT — Base scientifique & seuils

> Chaque seuil de l'app vit ici, avec sa source et sa limite. Règle d'or :
> aucun chiffre affiché sans étiquette « mesuré / indicatif / calibrable ».
> Mis à jour le 11 juin 2026 après revue des sources.

## 1. La séquence de départ UCI / ProGate (le fait le plus important)

La cadence de départ UCI est **fixe une fois lancée** ; seul son *déclenchement*
est aléatoire :

- **Commandes vocales** : « OK riders, random start. Riders ready? Watch the gate. »
- **Délai aléatoire** : entre **0,1 et 2,7 s** après les commandes, avant le 1er feu/bip.
- **4 feux + 4 bips** espacés de **~120 ms** chacun (rouge, jaune, jaune, vert).
- **La grille tombe sur le 4e (vert)** — soit **~360 ms après le 1er bip**
  (3 × 120 ms).

→ Source : implémentation open-source conforme UCI
[gndean/bmx-start-gate](https://github.com/gndean/bmx-start-gate) — `random(100, 2700)` ms,
`delay(120)` entre feux, solénoïde de grille déclenché sur le feu vert.
Confirmé conceptuellement par la [doc ProGate](https://progate.net/) (fournisseur
officiel UCI) et l'[annonce UCI du random start](https://www.fatbmx.com/bmx-racing/item/3287-uci-announces-introduction-of-bmx-random-start-gate).

**Validation sur nos données** : les essais bien marqués montrent un intervalle
bip 1 → grille de **exactement 360 ms** — cohérent avec la spec. Les essais à
700–1700 ms trahissent une **erreur de marquage de la grille ou de détection du
bip 1** (signal de qualité, voir §2).

## 2. Le temps de réaction = 1er bip → 1er mouvement (et pourquoi)

**Définition (validée terrain, juin 2026)** : la réaction est le délai entre le
**1er bip** et le **tout premier mouvement** du rider — « dès qu'il y a un peu de
mouvement ». C'est la définition coach, et la bonne.

**Mais on ne mesure PAS ce délai sur le bip détecté à l'audio.** La détection
attrape souvent un écho ou la voix « watch the gate » comme « 1er bip », ce qui
gonflait la réaction à 800–1700 ms — des valeurs absurdes (un humain réagit en
~170 ms). Pire, c'était **incohérent** : on affichait « 562 ms de réaction » ET
« 200 ms avant la grille », impossible.

**Solution — ancrer le 1er bip sur la cadence (§1)** : la grille tombe toujours
360 ms après le 1er bip, donc **1er bip = grille − 360 ms**. La réaction devient
`réaction = (1er mvt − grille) + 360 ms`. Toujours cohérente, toujours plausible.
Sur nos 22 essais, ça ramène la fourchette de 160–1720 ms (absurde) à
**160–427 ms, médiane 327 ms** — pile la plage humaine.

**On ne peut PAS anticiper le 1er bip.** Il arrive après un délai ALÉATOIRE
(0,1–2,7 s, cf. §1) : le rider y **réagit**, il ne peut pas le devancer. Seule la
**grille** est anticipable, parce qu'elle tombe à un instant fixe (+360 ms) une
fois le 1er bip entendu. La compétence = réagir vite au bip ET caler son drive
sur la grille.

Repères physiologiques :
- **Plancher de réaction humaine** ~170 ms (auditif ; tactile 150, visuel 250).
  Un 1er mouvement **sous ~170 ms** après le bip est sous ce plancher → le rider
  n'a pas pu *réagir* au bip, il a **deviné** le départ (coup de poker, risque de
  faux départ).
- En BMX un bon départ part **AVEC la grille** (à 360 ms du bip), donc la
  réaction d'un bon rider est **~300–360 ms** : ce n'est pas un réflexe de 170 ms,
  c'est une réaction au bip + un drive calé sur la grille (le seul repère qu'on
  puisse anticiper).

| Réaction (1er bip → 1er mvt) | Lecture | Note |
|---|---|---|
| avant le 1er bip | **Faux départ** (illégal) | 0 |
| < 230 ms (sous le plancher ~170) | Coup de poker — deviné, pas réagi | 52–84 |
| 230–300 ms | Réaction vive, parti avant la grille | 86–99 |
| **300–385 ms** | **Idéal** — parti avec la grille | 100 |
| 385–470 ms | Un peu tard (après la grille) | 95→70 |
| > 470 ms | Réaction lente | décroît vers 20 |

**Détection du 1er mouvement (raffinée, juillet 2026)** : `_detect_first_move`
dans app.py — multi-articulaire (genou + hanche + inclinaison du tronc), calculée
depuis le CSV de landmarks déjà stocké (aucun re-passage YOLO). Baseline robuste
= médiane/MAD sur [grille−1,5 s, grille−0,6 s] (avant le 1er bip) ; onset = 1re
excursion au-delà d'un seuil adaptatif (max(6×MAD, 5° genou/hanche, 4° tronc))
soutenue 2 frames, dans [grille−0,55 s, grille+0,8 s]. Confiance = nb de signaux
qui confirment dans les 150 ms. Validation sur nos 22 essais : 15/20 détectés à
±33 ms (1 frame) de la valeur pipeline — les deux méthodes se recoupent.

**Confirmation coach (juillet 2026) — la règle finale** : un temps de réaction
n'est **évalué** (noté dans la scorecard, compté dans les stats et la
progression) que si le coach a **confirmé à l'œil les deux instants** qui le
composent : la chute de la grille ET le premier mouvement. Le logiciel pré-place
les marqueurs (détection raffinée ci-dessus) et affiche une **estimation
clairement étiquetée « ~X ms · à confirmer »**, jamais notée. Panneau de
vérification sur la page résultats : vidéo image par image, touches G/M,
recalcul automatique des phases si la grille est corrigée. « Continuer sans » =
estimation affichée, jamais comptée. (`reaction.verified` dans results,
endpoints `/jobs/{id}/reaction_verify` · `reaction_skip`.)

**Honnêteté — réaction « non mesurable »** : si aucun signal ne bouge franchement
autour de la grille (rider trop petit, caméra qui suit), la réaction est marquée
**« non mesurable »** (exclue de la note et des graphes) plutôt qu'inventée. En
plus, si un mouvement net existe bien APRÈS la fenêtre (scan étendu à +2,5 s),
l'app le signale : « la grille est peut-être mal placée sur cette vidéo » —
c'est le cas des 2 essais non mesurables (mouvement à +866 et +1500 ms).

Bandes **indicatives et calibrables** : il n'existe pas de barème public universel
pour l'amateur. Elles encodent la spec (§1) + le plancher auditif.

**Contrôle de cohérence du gate** : si le bip 1 est détecté, on connaît la grille
théorique (bip 1 + 360 ms). Si la grille marquée s'en écarte de > ~150 ms, on
affiche un avertissement de fiabilité plutôt qu'un faux temps précis.

## 3. Biomécanique du départ (déterminants)

| Source | Ce qu'on en tire | Limite |
|---|---|---|
| **Gross 2017** — *Where is time lost in the BMX SX gate start?* | Le **countermovement précoce** est le déterminant n°1 ; extension proximale-distale (hanche→genou→cheville) ; genou balistique 400–700°/s **en capture marqueurs** | Labo, marqueurs, petit n |
| **Grigg 2018/2020** (thèse Bond Univ., n=10 élites, markerless ±2°) | Amplitudes articulaires de référence ; **position de set « back » = la plus performante** (variable n°1) ; trajectoire du moyeu (signal n°2) | n=10, élites masculins SX, 120 fps |
| **Grigg & Haakonssen** — *Kinematics of the BMX SX gate start* (markerless validé) | Le markerless est **valide et fiable** pour ce geste (justifie notre approche sans capteur) | Conditions contrôlées |
| **Zabala 2009** | Le **feedback audiovisuel** améliore le temps de départ (~200 ms) | — |
| **Mahieu** (bronze JO 2024) | Alignement épaules-bassin-chevilles pour la transmission | Anecdotique, pas une étude |

→ Sources : [Bond University — thèse Grigg](https://research.bond.edu.au/en/studentTheses/a-biomechanical-analysis-of-the-bmx-sx-gate-start),
[Where is time lost (Gross/Grigg)](https://research.bond.edu.au/en/publications/where-is-time-lost-in-the-bmx-sx-gate-start),
[Kinematics of the BMX SX gate start (JSC)](https://jsc-journal.com/index.php/JSC/article/download/249/479),
[Markerless validity (PubMed)](https://pubmed.ncbi.nlm.nih.gov/29129121/),
[Gate start position & performance (ResearchGate)](https://www.researchgate.net/publication/315636245).

## 4. Échelle markerless ≠ échelle labo (rappel critique)

Nos vitesses angulaires viennent de **pose markerless à ~30 fps** : elles sont
**systématiquement plus basses** que les 400–700°/s de Gross (marqueurs, haute
cadence) à cause du lissage + de la différentiation. **On ne note jamais sur
l'échelle absolue de Gross** ; nos bandes d'explosivité sont calibrées pour le
régime markerless et étiquetées comme telles. Gross reste la référence
*conceptuelle* (l'ordre et le caractère balistique), pas l'étalon numérique.

## 5. Constantes calibrables (où les trouver)

| Constante | Fichier | Valeur | Base |
|---|---|---|---|
| Cadence bip / grille | `audio_gate.py` | 120 ms / +360 ms | §1 |
| Tolérance cadence (qualité gate) | `audio_gate.py` `CADENCE_TOL_MS` | ±55 ms vs 120 ms | §1 |
| Offset audio→visuel grille | `audio_gate.py` | +0,095 s | calibré /gate_calibration |
| Pas de temps / précision | `app.py` `precision_tier` | 30 fps=±33 ms · 60=±17 · 120=±8 | 1000/fps |
| Bandes de réaction | `app.py` `REACT_*` | voir §2 | spec + plancher |
| Bandes explosivité (markerless) | `app.py` `EXPLO_*` | genou 110→360°/s | §4 |
| Séquence (scores) | `app.py` `SEQ_SCORES` | 100/65/45/25 | Gross 2017 |
| Contre-mouvement | `app.py` `CMV_*` | profondeur ≥10°, load ≤+60 ms | §3 |
| Position de set | `app.py` `SETPOS_*` | offset ≤ −0,30 ; trunk < 25° | §3 |
| Pondérations scorecard (5 dim) | `app.py` `SCORE_WEIGHTS` | réaction 0,24 · contre-mvt 0,22 · explosivité 0,20 · séquence 0,17 · posture 0,17 | réaction = timing course ; contre-mvt = déterminant #1 |
