# HOLESHOT — Vision produit

> Le départ BMX se joue en 300 millisecondes. HOLESHOT est le seul outil qui
> les voit, les mesure et les explique — au bord de la piste, sans capteur,
> en langage de coach.

## 1. Ce que devient l'app

**De** : un outil d'analyse fonctionnel mais brut (16 pages, navigation plate,
chiffres bruts).
**Vers** : un produit en trois couches, chacune pour un moment différent :

| Couche | Moment | Utilisateur |
|---|---|---|
| **Capture & verdict** (« au gate ») | Pendant la session, entre deux runs | Coach sur iPad, 30 s d'attention |
| **Studio d'analyse** (« au calme ») | Le soir, débrief | Coach + athlète, comparaison, annotations |
| **Progression** (« la saison ») | Hebdo/mensuel | Coach multi-athlètes, parents, athlète |

### Parcours coach (le payeur)
1. Arrive au bord de la piste → `/` = upload immédiat, gate auto-détecté au son,
   verdict 3 couleurs en ~2 min. Zéro friction.
2. Le soir → page résultats : explosivité, posture, comparaison pro synchronisée,
   annotations. Exporte un rapport PDF par rider.
3. Fin de semaine → tableau de bord de progression : tendances de réaction,
   régularité inter-essais, asymétries, plan de travail.

### Parcours athlète
Reçoit un lien de rapport (lecture seule) : sa vidéo annotée, son verdict,
ses 2 consignes de la semaine, sa courbe de progression. Gamification légère
(records personnels, badges de régularité — jamais de classement public).

### Identité
La marque proposée : **HOLESHOT** (le terme exact pour « gagner le départ »),
avec « BMX Start Analyzer » en sous-titre. Réversible si tu préfères garder le
nom actuel — toute la maquette fonctionne avec les deux.

## 2. Le système de design — « HOLESHOT DS »

Fichier : `static/design-system.css` (tokens + composants). Cinq principes :

1. **La piste comme métaphore** — l'axe horizontal = le temps de la course.
   Les reveals glissent en X (gauche → droite), jamais en Y.
2. **Telemetry, pas dashboard** — chiffres en mono `tabular-nums`, unités en
   petit, présentés comme un écran de course (F1/MotoGP).
3. **Explosif mais précis** — durées 120–320 ms, `ease-out-expo` : départ
   violent, arrivée douce. Rien > 600 ms hors hero.
4. **Le verdict en 3 couleurs** — vert/orange/rouge, lisible à 2 m en plein
   soleil. Aucune 4e couleur de jugement, jamais.
5. **Honnêteté scientifique visible** — chaque benchmark porte un badge source
   (`.ds-source` : « Grigg 2018 · n=10 »). Le doute est une feature.

Concrètement :
- **Typo** : Archivo 800/900 (display, caractère sport), Inter (UI, conservé),
  JetBrains Mono (données).
- **Couleurs** : navy/cyan conservés (c'est l'identité), enrichis d'un gradient
  « chrono » cyan→vert (CTA, succès) et d'une **rampe de 6 couleurs de phases**
  (Set, Réaction, Push 1, Pull, Push 2, Post) utilisée partout : timeline,
  graphes, chips, vidéo annotée. Une phase = une couleur, sur toute l'app.
- **Composants** : `.ds-kpi` (brique telemetry), `.ds-verdict`, `.ds-phase`,
  `.ds-source`, `.ds-btn`, `.ds-card`, `.ds-skeleton` (shimmer horizontal),
  `.ds-reveal`. Cibles tactiles ≥ 44 px, `prefers-reduced-motion` respecté.

## 3. La landing page

Fichiers : `templates/landing.html` + `static/landing.css`, route `/landing`
(l'app de travail reste sur `/` — un coach qui ouvre l'app tous les jours ne
doit pas retraverser une page marketing).

Pièce maîtresse : la **scène telemetry** — un squelette de rider SVG animé en
JS (interpolation de poses clés) qui exécute un départ complet en boucle :
bips UCI → chute du gate → réaction → push 1 → pull → push 2, avec la trace du
moyeu avant (clin d'œil Grigg 2020), les chips de phases qui s'allument, et un
feed de métriques qui tombent en direct (réaction 142 ms, ω genou 624 °/s,
séquence H→G→C ✓, verdict). Pause automatique hors viewport (perf iPad),
version statique si `prefers-reduced-motion`.

Sections : hero → scène → méthode (3 temps) → 6 mesures sourcées → teaser
comparaison rider/pro → science (4 papiers + bandeau d'honnêteté) → témoignage
(étiqueté « programme pilote ») → CTA → footer.

## 4. Analyse — ce qui vient ensuite (après alignement)

1. **Scorecard du départ** : une note synthétique par essai (réaction,
   explosivité, séquence, posture), pondérations calibrables, affichée en
   « écran de course ». Base du rapport PDF.
2. **Tableau de bord de progression** : courbes de réaction/ω dans le temps,
   PB, régularité inter-essais (écart-type des 5 derniers), par piste.
3. **Résurrection des modules dormants** : trajectoire du moyeu (hairpin /
   up-and-over), classification du set (back/upright/angled), alignement
   Mahieu — les trois sont déjà codés ou semi-codés.
4. **Asymétrie gauche/droite** : les 17 points COCO sont bilatéraux, le
   pipeline les a déjà — il suffit de comparer les deux côtés.
5. **Rapport PDF exportable** : vidéo-vignettes + scorecard + consignes,
   partageable aux parents.
6. **Refonte des pages clés** (accueil, résultats, comparaison) sur le DS.

## 5. Architecture cible (trajectoire cloud)

- **Aujourd'hui** : FastAPI + JSON locaux. On garde — c'est simple et ça marche.
- **Étape 1** : SQLite via SQLModel (mêmes schémas que les JSON, migration
  douce), comptes coach simples, déploiement Fly.io avec volume.
- **Étape 2** : file d'attente d'analyse (le GPU/CPU lourd reste un worker),
  stockage vidéo S3-compatible, PWA installable sur iPad (déjà presque le cas
  avec les meta apple-mobile-web-app).
- Le backend Python d'analyse reste le joyau : il ne bouge pas, il s'expose.

## 6. Roadmap — état au 11 juin 2026

### Livré ✅
- **Rebrand HOLESHOT + design system** appliqué au shell (`design-system.css`,
  Archivo, rampe de phases).
- **Scorecard du départ** : endpoint `/scorecard/{job_id}` (4 dimensions notées
  + note globale pondérée, bandes recalibrées pour le markerless), affichée en
  tête de la page résultats (anneau + cartes sourcées).
- **Dashboard de progression** : endpoint `/athletes/{id}/progression`
  (séries note/réaction/ω, records, tendance, régularité), graphes SVG vanilla.
- **Rapport imprimable** : `/report/{job_id}`, print-to-PDF, CSS `@media print`
  (encre claire), zéro dépendance nouvelle.
- **Contre-mouvement (Gross 2017, déterminant #1)** : `/countermovement/{job_id}`,
  détection load→extension (profondeur + timing), carte résultats + rapport.
- **Position de set (Grigg 2020, variable #1)** : `/set_position/{job_id}`,
  back / upright / angled depuis la géométrie corps, carte + rapport.

### Décisions d'honnêteté scientifique
- **Asymétrie G/D** : NON livrée. Non mesurable de façon fiable en vue de profil
  (jambe arrière occultée, front/rear font des gestes différents). Remplacée
  par le contre-mouvement (déterminant #1, lui mesurable).
- **Trajectoire du moyeu (Grigg signal #2)** : NON livrée via proxy poignet —
  le poignet capte le rock-back du corps, pas le moyeu (tout sortait « hairpin »).
  Remplacée par la position de set (variable #1, mesurable depuis le corps).
  Vraie résurrection = tracking du moyeu (Hough sur vidéo) → backlog.

### Backlog priorisé (impact × effort)
| # | Feature | Impact | Effort |
|---|---|---|---|
| 1 | Refonte comparaison (DS) | ★★ | M |
| 2 | Intégrer contre-mouvement comme 5e dimension de la scorecard | ★★ | S |
| 3 | Coach IA conversationnel (explique le départ) | ★★ | M |
| 4 | Multilingue FR/EN/ES | ★★ | M |
| 5 | Vraie trajectoire du moyeu (Hough, opt-in, post-analyse) | ★★ | M |
| 6 | Comptes + SQLite + déploiement cloud | ★★ | L |
| 7 | Analyse de la course complète (1er virage) | ★ | L |
