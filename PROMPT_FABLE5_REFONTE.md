# Prompt pour Fable 5 — Refonte totale & ambition maximale (BMX Start Analyzer)

> Copie tout à partir de « Tu es… » dans le chat Fable 5.

---

Tu es un **directeur produit + designer UI/UX de classe mondiale + ingénieur full-stack + biomécanicien du sport**. Je te confie un projet et je te donne **carte blanche totale**. Je veux que tu construises **la meilleure application de tous les temps pour le départ en BMX race — et pour le BMX en général**. Sors de la boîte. Propose des choses auxquelles personne n'a pensé. Tu peux tout changer.

## La mission

Le BMX race amateur mondial (~50–100k pratiquants sérieux) reçoit **zéro feedback biomécanique automatisé**. Les outils existants (Kinovea, Dartfish, Hudl) sont génériques et demandent un travail manuel énorme. Les systèmes pro (G-Cog, PerfAnalytics) sont inaccessibles. **Mon app comble ce vide** : analyse vidéo automatique, spécifique BMX, base scientifique, feedback actionnable pour coaches et athlètes.

Aujourd'hui c'est un outil fonctionnel mais brut. **Je veux que tu en fasses un produit dont les gens tombent amoureux.** Une page d'accueil de feu, un fini visuel irréprochable, une analyse plus profonde, et toutes les features intelligentes que tu juges utiles.

Repo public (pour le contexte, lis-le si tu peux) : **https://github.com/Louispump/bmx-start-analyzer**

## Stack actuelle (tu peux la faire évoluer, mais garde le backend Python qui est le joyau)

- **Backend** : FastAPI (Python 3.12). C'est lui qui fait toute l'analyse — NE LE CASSE PAS, c'est des mois de travail biomécanique.
- **Frontend** : Jinja2 + JavaScript **vanilla** + un seul `static/style.css` (4200 lignes). Tu peux tout refondre côté front : proposer une vraie architecture CSS (design tokens, composants), voire un build moderne si tu justifies. Mais ça doit rester servable par FastAPI et simple à déployer.
- **Vision par ordinateur** : YOLOv11-Pose (17 keypoints COCO), ByteTrack, OpenCV, ffmpeg, scipy (Savitzky-Golay, find_peaks).
- **Données** : JSON + fichiers locaux (pas encore de vraie DB). Persistance : `jobs_db.json`, `pros_db.json`, `athletes_db.json`, `tracks_db.json`.
- **Cible** : coaches + athlètes BMX, surtout sur **iPad au bord de la piste**, aussi desktop. Tout en **français**.

## Identité visuelle actuelle (point de départ — tu peux l'élever ou la réinventer)

- Fond dark navy très sombre `#060a0f`, surfaces `#0e1420`, accent **cyan `#00d4ff`**.
- Vert `#2dcd6e`, orange `#f59c18`, rouge `#ff4d4a` pour les verdicts.
- Sidebar à gauche, contenu à droite. Responsive iPad/mobile.
- C'est correct mais ça manque d'âme, de mouvement, de "wow". Donne-lui une vraie personnalité — vitesse, explosivité, précision, sport de haut niveau.

## Ce que l'app fait déjà (16 pages, ~15k lignes)

**Pipeline d'analyse** : upload vidéo → marquage du gate drop (manuel + détection audio des 4 bips UCI + détection visuelle) → pose estimation → segmentation en phases (Set, Reaction, Push 1, Pull 1, Push 2, Post — framework Kalichová) → vidéo annotée + CSV de landmarks.

**Pages & features existantes** :
- **Accueil / upload** : drag & drop, barre de progression de téléversement, sélection du gate.
- **Page de résultats** (`/result`) : temps de réaction, timeline des phases, vidéo annotée avec outils d'annotation (flèche, crayon, cercle, texte, couleurs), export PNG.
- **Module Explosivité** : vitesses angulaires ω (°/s) hanche/genou/cheville, séquence proximale-distale (hanche→genou→cheville), comparaison au référentiel perso de l'athlète, verdict coach en langage simple + conseil d'entraînement.
- **Module Consignes de position** : analyse posturale par phase (set, push 1) avec consignes précises (« recule tes hanches », « plie ta jambe », « avance ton bassin »).
- **Comparaison côte-à-côte** (`/compare`) : rider vs pro (ou vs un autre départ), barre de lecture synchronisée sur le gate drop, play/pause sync (driver-follower pour iOS), 3 modes de cadrage (naturel / même hauteur / recadré sur le rider), annotations, vitesses jusqu'à 0.1×.
- **Banque de pros** : upload de vidéos de référence + previews H.264 720p compacts (pour fluidité iPad).
- **Dossiers athlètes** : historique, tags (Course/Entraînement/Warmup/PB), notes, pistes, stats.
- **Calibration audio du gate**, **outils gear/pression de pneus**, **backup auto des données**.

## Fonctionnalités qui ont été retirées / mises en dormance — tu peux les ressusciter et les faire marcher si tu les juges utiles

- **Trajectoire du moyeu avant** : classification hairpin / up-and-over / demi-cercle (signal #2 de la littérature Grigg). Code existant mais sorti du pipeline.
- **Classification de la position de set** : upright / back / angled (variable #1 prédictive selon Grigg).
- **Métrique d'alignement Mahieu** (épaules-bassin-chevilles) : `mahieu.py` existe encore.
- **Multilingue FR / EN / ES** : avait été ajouté puis revert. Pourrait ouvrir le marché international.

## Le cadre scientifique (à intégrer/exploiter à fond)

- **Grigg 2018** (n=10 élites SX, markerless ±2°) — benchmarks d'amplitude articulaire : tronc 39°±6, hanche 62°±11, genou 93°±12, cheville 58°±14, épaule 87°±7, coude 47°±15. (À présenter avec prudence : petit échantillon, masculins élites.)
- **Grigg 2020** — typologie position de set (back = la plus performante) + trajectoire du moyeu.
- **Gross 2017** — countermovement précoce = déterminant #1 ; séquence proximale-distale ; extension balistique ~400-700°/s au genou.
- **Mahieu** (bronze JO Paris 2024) — alignement épaules-bassin-chevilles pour la transmission de puissance.
- **Kalichová 2013** — segmentation en phases.
- **Zabala 2009** — le feedback audiovisuel réduit le temps de départ de ~200 ms.
- **Becerra-Patiño 2025** (revue systématique) — Pmax, cadence, capacité neuromusculaire, feedback comme déterminants.

## CE QUE JE VEUX DE TOI

**1. Une page d'accueil / landing page de fou.** Pas juste un formulaire d'upload. Une vraie page qui vend le rêve : hero percutant, démonstration visuelle de ce que l'app fait (avant/après, squelette animé, métriques qui s'affichent), preuve scientifique, témoignage type coach, call-to-action clair. Quelque chose qui donne envie à un coach de BMX de dire « il me faut ça ».

**2. Un fini visuel irréprochable sur TOUTE l'app.** Système de design cohérent (typographie, espacements, composants, micro-animations, transitions, états de chargement soignés, feedback tactile iPad). Que chaque page respire le produit premium, pas le prototype.

**3. Une analyse encore meilleure.** Améliore ce qui existe (explosivité, posture, comparaison) ET ajoute tout ce que tu juges utile : nouvelles métriques, meilleures visualisations (graphes, overlays squelette, heatmaps, courbes temporelles), un vrai tableau de bord de progression de l'athlète dans le temps, des rapports exportables (PDF), une vue « scorecard » d'un départ, etc.

**4. Des features hors-boîte.** Pense en grand. Quelques pistes pour t'inspirer (mais surprends-moi avec les tiennes) : coach IA conversationnel qui explique le départ, plans d'entraînement personnalisés générés depuis les faiblesses détectées, comparaison à une « base de données idéale », détection d'asymétrie gauche/droite, suivi de cohérence inter-essais, gamification/badges de progression, partage de rapports avec les parents, mode « side-by-side » contre un champion du monde, overlay de la trajectoire idéale, analyse de la course complète (pas juste le départ) puisque tu peux élargir au **BMX en général**.

**5. Pense produit complet, pas juste outil.** Onboarding, navigation repensée, hiérarchie de l'information, parcours coach vs parcours athlète, multi-athlètes pour un coach, et la trajectoire vers une vraie app (cloud, comptes, mobile) — propose l'architecture.

## Contraintes (les seules)

- **Le backend Python et le pipeline d'analyse sont sacrés** : tu peux les améliorer/étendre, mais l'app doit continuer de tourner. Tout ce qui touche YOLO/pose/angles/phases/gate doit rester fonctionnel.
- **Français** par défaut (mais tu peux rajouter le multilingue).
- **iPad-first** : utilisable au bord de la piste, tactile, vidéos lourdes, lecture fluide.
- Doit rester **servable par FastAPI** et **déployable simplement** (aujourd'hui local, demain Fly.io/Railway).
- **Honnêteté scientifique** : ne jamais inventer une précision qu'on n'a pas. Les seuils/benchmarks doivent rester calibrables et étiquetés.

## Livrables attendus

1. **Une vision produit** : ce que devient l'app, pour qui, le parcours utilisateur idéal (coach et athlète).
2. **Un système de design complet** : palette, typographie, tokens, composants clés, principes de motion. En code (CSS/variables) prêt à utiliser.
3. **La landing page complète** : HTML + CSS + JS, prête à intégrer.
4. **La refonte des pages clés** (au minimum : accueil, résultats, comparaison) : code complet.
5. **Les améliorations & nouvelles features d'analyse** : pour chacune, l'idée, la valeur, et le code (Python backend + endpoint + frontend) quand c'est réaliste.
6. **Une roadmap priorisée** (impact × effort) de tout le reste que tu recommandes.

Commence par me présenter **ta vision globale + le système de design + le concept de la landing page** (avec maquette décrite ou code), qu'on s'aligne sur la direction avant que tu produises tout le reste. Ensuite déroule. **Ne te bride pas — je veux la meilleure app BMX qui ait jamais existé.**
