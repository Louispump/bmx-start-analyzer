# Mission : précision & réalisme de l'analyse (11 juin 2026)

> Auto-brief — carte blanche utilisateur : « du précis et du réel pour
> l'analyse », nouvelles sources, vidéos de pros, temps de réaction qui fait
> sens, design complet.

## 1. Temps de réaction qui FAIT SENS (priorité absolue)
- [ ] Rechercher la spec réelle du start UCI/ProGate : délai aléatoire, cadence
      des 4 bips, instant où la grille tombe vs bips, durée de chute (~0.3 s ?).
- [ ] Vérifier la cadence bip1→gate sur NOS données (tous les jobs avec audio).
- [ ] Le plancher humain de réaction auditive (~140–160 ms) : tout mouvement
      < plancher après un bip = anticipation TIMÉE sur la cadence (c'est le but
      du départ BMX), pas une « réaction ». Le verdict doit le dire.
- [ ] Re-modéliser le score réaction : faux départ (avant bip 1) / anticipation
      excessive / fenêtre optimale (parti AVEC la grille) / réactif-tardif.
      Bandes justifiées par la spec gate + littérature, étiquetées.
- [ ] La carte réaction de la page résultats doit expliquer les deux référentiels
      (vs bip 1 = cadence totale ; vs grille = timing du départ).

## 2. Nouvelles sources scientifiques
- [ ] Chercher la littérature 2017–2025 : déterminants du départ SX, temps de
      réponse au gate, anticipation, set position, countermovement.
- [ ] Mettre à jour les bandes de la scorecard avec citations propres.
- [ ] Documenter dans SCIENCE.md : chaque seuil → sa source → sa limite.

## 3. Scorecard plus réelle
- [ ] Intégrer le contre-mouvement comme 5e dimension (déterminant #1 ≠ absent
      de la note, c'est incohérent).
- [ ] Re-pondérer (réaction/CM en tête, posture/séquence ensuite).
- [ ] Afficher les délais réels de la séquence H→G→C en ms (pas juste l'ordre).

## 4. Vidéos de référence pros
- [ ] Curater des sources publiques de départs élites filmés de profil
      (UCI SX, finales JO, contenus d'équipes) avec ce qu'il faut regarder.
- [ ] Les intégrer à la Banque de pros comme guide d'import (pas de
      téléchargement automatique — droits + ToS).

## 5. Design complet
- [ ] Page comparaison : passage au DS (header, boutons, tags rider/pro,
      sync-bar) sans toucher à la logique driver-follower.
- [ ] Unifier les couleurs de phases PARTOUT (la timeline résultats utilise
      encore les pastels du backend) via la rampe DS.
- [ ] Pass de cohérence : titres, boutons, badges, états vides.

## Garde-fous
- Backend pipeline sacré (YOLO/phases/gate) : on lit, on n'altère pas.
- Aucun chiffre sans source ou sans étiquette « indicatif/calibrable ».
- Tout vérifié dans le navigateur avant de déclarer fini.
