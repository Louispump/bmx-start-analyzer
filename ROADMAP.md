# ROADMAP — Améliorations biomécaniques app analyse départ BMX Race
> Généré : mai 2026 | Basé sur revue de littérature 2017-2025

---

## 1. État de l'art — Mise à jour des benchmarks Grigg

### 1.1 Statut des benchmarks Grigg (2017-2020)

Les études Grigg de référence (publiées dans *Sports Biomechanics* et *Journal of Science and Cycling*)
restent les seules à fournir des angles articulaires normalisés sur départ SX avec capture vidéo markerless.
Leurs chiffres (n=10 élites masculins SX) sont :

| Articulation | Amplitude (°) | SD (°) |
|---|---|---|
| Tronc | 39 | ±6 |
| Hanche | 62 | ±11 |
| Genou | 93 | ±12 |
| Cheville | 58 | ±14 |
| Épaule | 87 | ±7 |
| Coude | 47 | ±15 |

**Limites identifiées :**
- N très faible (n=10), 100 % masculins, 100 % élites SX
- Méthode markerless validée à ±2° (acceptable mais pas gold standard)
- Plan sagittal uniquement — asymétrie frontale non capturée
- Aucune comparaison SX 8 m vs piste classique 5 m publiée
- Aucune donnée junior, féminine ou amateur dans ces études

### 1.2 Publications post-Grigg pertinentes (2020-2025)

**Princelle D., Monnet T., Domalain M. et al. — Thèse & série de papers (2021-2023)**
*Institut Pprime / CNRS / Université de Poitiers*
- Titre : « Analyse biomécanique multi corps et évaluation de la performance lors du départ en BMX race »
- Méthode : modèle multi-corps + OpenSim + pédales instrumentées, n=8 élites nationaux
- Apports : cinématique + dynamique articulaire (hanche), cadence ~62 rpm au départ, trajectoire
  du centre de masse calculée, interaction pilote-vélo modélisée
- Référence HAL : [tel-04131317](https://theses.hal.science/tel-04131317)
- Paper connexe : « Dynamic analysis of the BMX start: interactions between riders and their bike »
  *Computer Methods in Biomechanics and Biomedical Engineering*, 2020
  DOI : [10.1080/10255842.2020.1714924](https://doi.org/10.1080/10255842.2020.1714924)

**Équipe Pprime — « BMX Race-Start Dynamics: Coupling On-Track Measurements and Physics-Based
Modelling for Performance Optimisation » (2024-2025)**
*ResearchGate pub. 398881230*
- Track : piste olympique de Saint-Quentin-en-Yvelines (Paris 2024)
- Modèle Hill (torque-cadence) + bilan énergétique (aéro + frottement) + pente mesurée
- Prédit le profil de vitesse sur rampe → optimisation du braquet individuel
- Premier papier à coupler mesures terrain haute fréquence + modèle physique validé

**Becerra-Patiño B.A. et al. — Systematic Review (2025, OPEN ACCESS)**
*Journal of Functional Morphology and Kinesiology*, 2 juin 2025, 10(2):205
DOI : [10.3390/jfmk10020205](https://doi.org/10.3390/jfmk10020205) | PMC12194740
- Première revue systématique (PRISMA) sur les déterminants de performance BMX
- 21 études, 287 athlètes
- Conclusion : Pmax (puissance max), cadence, capacité neuromusculaire, feedback audiovisuel,
  accélérométrie + vidéo, et profil force-vitesse sont les facteurs déterminants
- Qualité de preuve : modérée (petits échantillons, pas de randomisation)

**Gross M., Schellenberg J., Lüthi A., Baker J., Lorenzetti S. (2017)**
« Performance determinants and leg kinematics in the BMX supercross start »
*Journal of Science and Cycling*, 6, 3-12
- Méthode : manivelle modifiée (power meter) + capture 3D (lab)
- Résultats clés :
  - Timing précoce du retournement (countermovement) = déterminant #1 de performance
  - Premier coup de pédale : couple ET vitesse angulaire élevés simultanément
  - Vitesses d'extension similaires à des sauts balistiques avec charges élevées

**Gross M. & Gross (2019)**
*Force-velocity (Fv) et torque-cadence (Tc) — BMX vs sprints plat*
- Les profils Fv diffèrent entre départ sur rampe et sprint sur plat
- Les paramètres haute vitesse montrent des divergences → l'entraînement spécifique rampe est irremplaçable

---

## 2. Métriques manquantes — Variables biomécaniques prédictives non calculées

### 2.1 Timing du countermovement (pre-gate maneuver)

**Source** : Gross et al. (2017), JSC
**Métrique** : Délai entre la chute du gate et le début de l'extension genou/hanche (ms)
**Verdict** : Déterminant #1 selon Gross et al. — un retournement précoce et rapide est la
variable qui discrimine le plus les performances au départ.
**À coder** : Détecter le pic de flexion maximale (genou/hanche) avant la chute du gate
→ calculer le délai en ms par rapport à l'audio gate.
**Impact** : Directement corrélable au temps au bout de la rampe.

### 2.2 Vitesses angulaires articulaires (explosivité)

**Source** : Gross et al. (2017) ; Cowell et al. (2020)
**Métrique** : Vitesse angulaire (°/s) au genou, hanche et cheville au moment du premier coup de pédale
**Verdict** : Extensions hanche/genou/cheville au départ BMX sont comparables à des
sauts balistiques (≈ 500-700°/s genou). La *vitesse* d'extension est aussi importante
que l'amplitude.
**À coder** : Dériver les angles articulaires par rapport au temps (Δangle/Δframe × fps).
Afficher un index "explosivité" = vitesse angulaire max sur les 3 premières images après
extension déclenchée.
**Impact** : Permet de distinguer un angle correct exécuté lentement vs correctement ET
de façon explosive — dimension absente des benchmarks Grigg.

### 2.3 Asymétrie L/R de la cinématique du haut du corps

**Source** : Grigg J. et al. (2017) — « Validity and Reliability of a 2D Kinematics Method
for Measuring Athlete Symmetry during the BMX Gate Start »
*ResearchGate pub. 319881896*
**Métrique** : Index d'asymétrie = |angle_gauche - angle_droit| / moyenne × 100 %
sur épaule, coude, hanche (plan sagittal et frontal)
**Verdict** : Les élites ne sont PAS symétriques — l'asymétrie du haut du corps ne peut pas
être supposée nulle même chez les meilleurs. L'inégalité avant/arrière dans la posture de set
est fonctionnelle (pied avant ≠ pied arrière).
**À coder** : Déjà détecté le pied avant. Étendre : calculer l'index d'asymétrie épaule
et coude G vs D. Afficher un score 0-100 (0 = parfait symétrique) comme indicateur
de "biais postural".
**Impact** : Coaching ciblé côté faible ; suivi des déséquilibres dans le temps.

### 2.4 Trajectoire du centre de masse (COM)

**Source** : Princelle et al. (2020-2023) — Pprime / OpenSim
**Métrique** : Trajectoire 2D du COM dans le plan sagittal (x, y) en fonction du temps
depuis gate drop
**Verdict** : Le COM encode la stratégie globale de transfert de poids mieux que la
trajectoire du moyeu seul. Sa vitesse horizontale en sortie de rampe corrèle avec la
position en course.
**À coder** : Estimation simplifiée via moyenne pondérée des segments (modèle 7 segments,
masses de Dempster/Winter). Tracer la courbe COM superposée à l'image annotée.
**Note** : Approximation 2D acceptable pour coaching, pas besoin de modèle multi-corps complet.

### 2.5 Puissance mécanique estimée (proxy)

**Source** : Becerra-Patiño et al. (2025) ; Moya-Ramón et al. ; Daneshfar et al.
**Métrique** : Pmax (proxy) = f(vitesse angulaire genou × index de charge estimée)
OU estimation via accélération COM × masse rider (Newton 2e loi)
**Verdict** : Pmax explique 78 % de la variabilité de performance sur tour complet
(Moya-Ramón et al., r=0.87). Sans capteur de force, un proxy cinématique (accélération COM)
est la meilleure approximation utilisable en vidéo.
**À coder** : P_proxy = m_rider × ||a_COM|| × v_COM (puissance instantanée COM).
Nécessite la masse du rider (champ à saisir) et le delta-t entre frames.
**Impact** : Proxy imparfait mais unique indicateur de puissance accessible sans instrumentation.

### 2.6 Variabilité inter-essais de la position de set (consistency score)

**Source** : Cowell et al. ; scoping review JSC (2019)
**Métrique** : Écart-type des angles de set (tronc, genou, hanche) entre N essais d'une même session
**Verdict** : La reproductibilité de la posture de set est reconnue comme un facteur
de performance stable. Les études notent que la variabilité inter-essais est sous-étudiée
mais corrèle à la fiabilité du départ.
**À coder** : Pour une session comportant ≥3 essais avec le même tag, calculer
σ (angle_set) pour chaque articulation. Afficher une carte de chaleur "consistency" par
articulation (vert = stable, rouge = variable).
**Impact** : Identifie si le rider a un problème de technique répétable ou de position de set.

### 2.7 Timing de déclenchement du premier coup de pédale (reaction + motor delay)

**Source** : Zabala et al. (2009) ; études sur feedback BMX
**Métrique** : Délai en ms entre audio gate (bip UCI corrigé +95 ms) et premier mouvement
détectable (déplacement genou > seuil)
**Verdict** : La réduction de ~200 ms sur ce timing améliore significativement le résultat.
Le feedback audiovisuel réduit ce délai de façon significative (Zabala et al.).
**À coder** : Tu as déjà l'audio détection. Ajouter : frame du premier mouvement articulaire
détectable → calculer reaction time (ms). Afficher en overlay avec la ligne de référence pro.

### 2.8 Ratio knee/hip extension (séquence proximale-distale)

**Source** : Cowell et al. (2020) ; Gross et al. (2017)
**Métrique** : Délai en frames entre pic de vitesse d'extension hanche vs pic genou vs pic cheville
**Verdict** : La coordination proximale→distale (hanche avant genou avant cheville) est le
pattern optimal. Une inversion (genou avant hanche) est un signe de technique inefficace.
**À coder** : Détecter l'ordre temporel des pics de vitesse angulaire max des 3 articulations.
Scorer : 0 = séquence inversée, 1 = simultané, 2 = proximal-distal correct.

---

## 3. Validation méthodologique — Labos actifs et collaborations

### 3.1 Institut Pprime — Poitiers (CNRS UPR 3346)

**Équipe RoBioSS** (Robotique, Biomécanique, Sport et Santé)
- **Tony Monnet** (MC, Université de Poitiers) — référent BMX race
  Profil ResearchGate : [Tony Monnet](https://www.researchgate.net/profile/Tony-Monnet)
- **Marc Domalain** — biomécanique articulaire hanche au départ BMX
- **Domitille Princelle** — thèse 2021, maintenant research engineer (Movella)
- **Collaboration active** avec l'équipe de France de BMX (FFC) et l'INS — confirmée pour Paris 2024
- **Technologie** : OpenSim + pédales instrumentées + capture inertielle terrain

**Statut 2024-2026** : Actif. Papers récents sur modélisation physique de la rampe olympique SQY.

### 3.2 Bond University — Gold Coast, Australie

**Josephine Grigg** — auteure des études 2017-2020 de référence
- Thèse : « A biomechanical analysis of the BMX SX gate start »
  [Bond University Research Portal](https://research.bond.edu.au/en/studentTheses/a-biomechanical-analysis-of-the-bmx-sx-gate-start)
- Méthode : markerless MoCap (GoPro 120fps + Kinovea), validée ±2°, intra-tester fiabilité ±6°

**Limite** : Focus SX uniquement, échantillons masculins élites, pas de suivi post-2020 identifié.

### 3.3 ETH Zürich / Haute École de sport de Macolin (HEFSM)

**Micah Gross, Silvio Lorenzetti** — paper 2017 JSC (performance determinants + Fv)
- Seule étude avec capture 3D lab + power meter instrumenté sur rampe BMX
- Référence pour vitesses angulaires et timing du countermovement

### 3.4 Universidad de Murcia / Universidad de Granada — Espagne

**Mikel Zabala, Manuel Mateo** — nombreux papers sur feedback, bicarbonate, puissance en course
- Série d'études sur feedback audiovisuel pour amélioration du départ (2009)
- Études de puissance terrain avec SRM

### 3.5 Universidad de Los Lagos — Chili / Universidad de Murcia

**Becerra-Patiño B.A., López-Gil J.F.** — Systematic Review 2025 (JFMK)
Première revue systématique PRISMA sur BMX performance.

---

## 4. Critiques et limites des frameworks existants

### 4.1 Critiques de Kalichová et al. (2013)

- N=2 (1 élite masc. + 1 élite fém.) — pas généralisable
- Analyse qualitative + vidéo standard — pas de cinématique quantitative précise
- Segmentation en phases valide conceptuellement mais seuils temporels arbitraires
- **Pas de publication de suivi** — pas validé sur cohorte

**Alternative** : La segmentation par phases de Gross et al. (2017) basée sur le cycle de pédale
et le countermovement est plus robuste mécaniquement. Peut compléter Kalichová.

### 4.2 Critiques de Grigg (2017-2020)

- **N=10 uniquement**, tous masculins, tous SX élite → extrapolation très limitée
- Markerless MoCap validée à ±2° sur les articulations, MAIS la méthode repose sur le positionnement
  manuel des marqueurs → variabilité inter-testeur non négligeable
- Plan sagittal 2D uniquement → asymétrie frontale non mesurée
- Mesures statiques (position de set) → peu de données sur la phase dynamique complète
- Grigg 2020 (thèse) est plus complète mais reste non publiée sous forme d'article peer-reviewed

**Impact sur l'app** : Les benchmarks Grigg sont utilisables comme reference de départ mais doivent
être présentés avec intervalles de confiance larges et étiquetés « élites SX masculins, n=10 ».

### 4.3 Approches alternatives crédibles

**Machine learning / pose estimation (actuel)**
- YOLO-Pose, MediaPipe, OpenPose : précision ~5-10° en condition non-contrôlée (angle de caméra,
  occlusions, vêtements amples) — validé pour coaching mais pas pour recherche
- Papier 2025 sur AI Pose Analysis in Resistance Training (arXiv:2510.20012) montre la faisabilité
  de l'extraction de trajectoires angulaires par deep learning depuis vidéo monoculaire

**Analyse fonctionnelle des données (FDA)**
- Utilisée dans la thèse Princelle pour analyser les courbes temporelles d'angle articulaire
  plutôt que des scalaires ponctuels → capture mieux la dynamique du mouvement
- Implémentable en Python (scikit-fda) pour comparer profils complets rider vs référence

**Modélisation physique (Pprime 2024)**
- Couplage mesures terrain (GPS/IMU haute fréquence) + modèle Hill → optimisation braquet
- Moins applicable en app vidéo mais ouvre la voie à un module de recommandation matériel

---

## 5. Coaching élite — Points techniques clés

### 5.1 Points consensuels dans la littérature et le coaching

**Séquence temporelle** (Gross 2017, coaching biomécanique)
- Retournement (countermovement) le plus tôt et le plus rapide possible avant gate drop
- Extension hanche AVANT genou AVANT cheville (chaine proximale→distale)
- Simultanéité couple × vitesse au premier coup de pédale

**Position de set** (Grigg 2020 — 3 typologies)
- *Back position* : le plus performant en moyenne → ramène le COM vers l'arrière, favorise
  la trajectoire "hairpin" du moyeu et le transfert vers l'avant
- *Upright position* : moins efficace (COM trop haut, moins d'inertie à libérer)
- *Angled position* : intermédiaire

**Transmission de puissance** (Mahieu / Princelle / Pprime)
- Alignement épaules-bassin-chevilles au moment du pic d'extension
- Le dos (gainage) sert de courroie de transmission : sans rigidité lombaire, la puissance
  des membres inférieurs est dissipée dans le tronc
- Appui actif sur le guidon pendant la phase de poussée (réaction opposée)

**Points coaching USA / British Cycling**
- *Greg Romero (bmxtraining.com)* : « The same principles that make humans efficient at walking,
  running, and jumping are the same principles elite riders use in their starts. The angles of
  joints in the front leg close and open similarly when running and riding. »
- *British Cycling coaching resources* : Importance de la progressivité — ne pas changer plusieurs
  éléments à la fois, muscular memory d'abord

**Feedback audiovisuel**
- Zabala et al. (2009) : réduction de 200 ms sur le temps de départ après programme de feedback
  audiovisuel (n=6 équipe nationale espagnole)
- Le feedback externe-focus (attention sur le résultat, pas le mouvement) est supérieur

### 5.2 Facteurs moins documentés (lacunes de recherche)

- Position de la tête / regard : non étudiée en BMX — analogie avec athlétisme (effets sur posture
  pelvienne et angle de tronc) mais pas de données BMX publiées
- Différences SX 8 m vs piste 5 m : aucune comparaison cinématique publiée — la pente différente
  implique logiquement des angles optimaux différents
- Données féminines et juniors : quasiment absentes de la littérature biomécanique

---

## 6. Recommandations prioritaires — Améliorations à coder

### PRIORITÉ 1 — Vitesses angulaires articulaires (explosivité index)

**Source** : Gross et al. (2017), JSC ; Cowell et al. (2020)
**Métrique exacte** : ω_max (°/s) pour genou, hanche, cheville, calculé sur fenêtre [gate drop, +0.5s]
**Verdict / seuil** : Extension genou ~400-700°/s pour élites (analogie saut balistique) ; pas de
benchmark BMX publié → afficher en percentile relatif à la base de données de l'app
**Pourquoi plus impactant** : Les angles seuls (Grigg) ne distinguent pas un athlète raide d'un
athlète explosif. L'explosivité est précisément ce qui discrimine les performances au départ.
**Complexité** : Faible (dérivée temporelle des angles déjà calculés)

---

### PRIORITÉ 2 — Timing du countermovement (pre-gate delay)

**Source** : Gross et al. (2017) — déterminant #1
**Métrique exacte** : T_countermovement = timestamp(pic flexion max genou) − timestamp(gate drop)
en ms, positif si avant gate drop
**Verdict / seuil** : Plus c'est précoce, mieux c'est — aucun seuil publié, mais la corrélation
avec le temps de sortie de rampe est forte
**Pourquoi plus impactant** : C'est le déterminant #1 selon la seule étude avec capture 3D + power
meter. Ton pipeline détecte déjà les angles → il suffit d'ajouter la détection du pic de flexion.
**Complexité** : Moyenne (nécessite un algorithme de détection de pic robuste)

---

### PRIORITÉ 3 — Séquence proximale-distale et index de coordination

**Source** : Gross et al. (2017) ; Cowell et al. (2020) — pattern dominant en BMX
**Métrique exacte** : Ordre temporel des pics ω_max : hanche (t1) → genou (t2) → cheville (t3)
Coordinating Index CI = 1 si t_hanche < t_genou < t_cheville, 0 sinon
**Verdict / seuil** : CI = 1 est le pattern élite. CI = 0 indique une inversion à corriger en priorité.
**Pourquoi plus impactant** : La séquence d'activation est un indicateur de technique directement
actionnable par le coaching, contrairement aux amplitudes qui peuvent être compensées.
**Complexité** : Faible (dérivé de #1)

---

### PRIORITÉ 4 — Score de consistance inter-essais

**Source** : Scoping Review JSC (2019) ; Cowell et al.
**Métrique exacte** : Pour N≥3 départs d'une même session :
CV(%) = (σ_angle / μ_angle) × 100 pour chaque articulation en phase Set
**Verdict / seuil** : CV < 5 % = excellent, 5-10 % = acceptable, > 10 % = inconsistant
(seuils issus de la littérature sport en général, pas spécifique BMX)
**Pourquoi plus impactant** : Un rider avec de bons angles mais un CV élevé ne progressera pas
en compétition — la reproductibilité est une dimension orthogonale à la technique.
**Complexité** : Faible (calcul statistique sur les données de session existantes)

---

### PRIORITÉ 5 — Profil asymétrie postural (symmetry index)

**Source** : Grigg J. et al. (2017) — « Validity and Reliability of a 2D Kinematics Method for
Measuring Athlete Symmetry during the BMX Gate Start »
**Métrique exacte** : SI(%) = |angle_cote_dominant − angle_cote_non_dominant| / moyenne × 100
Sur : épaule, coude (déjà calculés) + hanche côté avant vs côté arrière
**Verdict / seuil** : Grigg confirme que les élites ne sont pas symétriques → pas de seuil "zéro".
Objectif : tracker l'évolution du SI dans le temps plutôt qu'un absolu.
**Pourquoi plus impactant** : La détection du pied avant est déjà faite → l'asymétrie L/R
est calculable sans développement majeur, et renseigne sur les déséquilibres posturaux chroniques.
**Complexité** : Faible (extension de la détection de pied avant)

---

## 7. Axes de recherche restant à explorer

Ces pistes ne sont pas encore mûres pour être codées mais méritent une veille :

- **Modélisation physique de la rampe** (Pprime 2024) → recommandation de braquet individualisée :
  nécessite masse rider + cadence mesurée + pente piste
- **Analyse fonctionnelle des données (FDA)** → comparaison de profils temporels complets (courbes)
  rider vs référence, plutôt que scalaires ponctuels — plus riche que la comparaison angle-par-angle
- **Position de la tête / regard** → non documentée en BMX, mais inférable via YOLO-pose (keypoint
  tête + cou) → piste pour une feature future
- **Estimation de puissance proxy** via accélération COM → nécessite validation terrain (correlation
  avec SRM power meter) avant d'afficher un chiffre

---

## 8. Bibliographie complète

| # | Référence | Disponibilité |
|---|---|---|
| 1 | Kalichová M. et al. (2013). Biomechanics Analysis of Bicross Start. *Kinesiology*, 45(1). | ResearchGate |
| 2 | Grigg J. (2017). Kinematics of the BMX SX Gate Start. *JSC*, 6(3). | [JSC open access](https://jsc-journal.com/index.php/JSC/article/view/249) |
| 3 | Grigg J. (2017). Validity & Reliability of 2D kinematics / BMX Gate Start Symmetry. *Sports Biomechanics*. | [ResearchGate](https://www.researchgate.net/publication/319881896) |
| 4 | Grigg J. (2018). Validity & intra-tester reliability of markerless MoCap. *Sports Biomechanics*, 17(3). | [PubMed 29129121](https://pubmed.ncbi.nlm.nih.gov/29129121/) |
| 5 | Gross M. et al. (2017). Performance determinants and leg kinematics in the BMX SX start. *JSC*. | [JSC](https://www.jsc-journal.com/index.php/JSC/article/view/312) |
| 6 | Princelle D. et al. (2020). Dynamic analysis of the BMX start: interactions between riders and bike. *Comput. Methods Biomech.* | [DOI](https://doi.org/10.1080/10255842.2020.1714924) |
| 7 | Princelle D. (2021/2023). Thèse — Analyse biomécanique multi corps et évaluation de la performance lors du départ BMX race. Univ. Poitiers / Pprime. | [HAL tel-04131317](https://theses.hal.science/tel-04131317) |
| 8 | Équipe Pprime (2024-2025). BMX Race-Start Dynamics: Coupling On-Track Measurements and Physics-Based Modelling. | [ResearchGate 398881230](https://www.researchgate.net/publication/398881230) |
| 9 | Becerra-Patiño B.A. et al. (2025). A Systematic Review of Bicycle Motocross: Influence of Physiological, Biomechanical, Physical, and Psychological Indicators. *JFMK*, 10(2):205. | [PMC12194740](https://pmc.ncbi.nlm.nih.gov/articles/PMC12194740/) |
| 10 | Zabala M. et al. (2009). Effects of Feedback on Performance of the BMX Cycling Gate Start. *JSSM*. | [PMC3763285](https://pmc.ncbi.nlm.nih.gov/articles/PMC3763285/) |
| 11 | Cowell J. et al. (2020). Power Analysis of Field-Based BMX. *OAJSM*. | [PMC7360409](https://pmc.ncbi.nlm.nih.gov/articles/PMC7360409/) |
| 12 | Moya-Ramón M. et al. Sprint characteristics & prediction of performance in elite BMX. | Via Becerra-Patiño 2025 (ref. [42]) |
| 13 | Robert et al. Relationship vertical jump and BMX performance. | Via Becerra-Patiño 2025 (ref. [45]) |
| 14 | Gross M. & Gross (2019). Cyclic vs Non-Cyclic Force-Velocity in BMX cyclists. *Sports*, 7(11):232. | [MDPI open access](https://www.mdpi.com/2075-4663/7/11/232) |
| 15 | Grigg J. (2020). Thèse complète — A biomechanical analysis of the BMX SX gate start. Bond University. | [Bond Research Portal](https://research.bond.edu.au/en/studentTheses/a-biomechanical-analysis-of-the-bmx-sx-gate-start) |

---

*Document généré pour le projet d'analyse biomécanique automatique du départ BMX Race.*
*À mettre à jour dès publication de nouveaux papers — veille recommandée sur JSC, Sports Biomechanics, JFMK.*
