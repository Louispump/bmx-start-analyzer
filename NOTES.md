Hosting prévus:
- Frontend: Vercel (gratuit)
- API: Fly.io ou Railway (~$5/mois)
- Compute: Modal (pay per second, idéal pour usage sporadique)
- Coût total phase pré-revenus: $0-20/mois

## Métriques à implémenter (par tier de priorité)

### Tier 1 - Indispensables (prochaines sessions)

- **Position de set (Grigg)**: classification automatique upright/back/angled
  basée sur angle du tronc vs vertical et position du bassin relative au 
  pied avant. Variable #1 prédictive de vitesse selon Grigg.

- **Trajectoire du moyeu avant**: tracking du centre de la roue avant 
  (détection de cercle via Hough ou modèle dédié) pour classification 
  hairpin/up-and-over/demi-cercle. Signal #2 de la littérature.

- **Alignement Mahieu**: vérification colinéarité épaule-hanche-cheville 
  dans le plan sagittal. Feedback actionnable directement.

- **Timing des phases**: durée exacte de chaque phase segmentée, 
  comparaison aux percentiles élites.

### Tier 2 - Métriques avancées

- **Vitesses angulaires**: dérivée temporelle des angles pour mesurer 
  l'explosivité (°/s).

- **Cadence de pédalage**: fréquence des cycles de genou sur les 4 
  premiers coups.

- **Vitesse/accélération du rider**: déplacement centre de masse, 
  calibré en m/s via dimensions de vélo connues.

- **Symétrie gauche-droite**: comparaison pieds visibles simultanément 
  dans les cas favorables.

- **Position de la tête et regard**: les pros regardent où ils vont.

### Tier 3 - Métriques pro

- **Puissance estimée**: via masse rider + accélération + vitesse angulaire.

- **Index de snap au départ**: delta entre position set et position 
  100ms plus tard.

- **Trajectoire du centre de masse**: moyenne pondérée des landmarks.

- **Angle cuisse vs cadre vélo**: efficacité de transmission de puissance.

- **Temps de réaction pur**: via détection audio du voice-box.

### Tier 4 - Fonctionnalités outil

- **Comparaison côte-à-côte**: synchronisation auto avec vidéo de 
  référence (Mahieu, Daudet).

- **Tracking de progression**: évolution temporelle des métriques pour 
  un rider donné.

- **Détection audio automatique du gate drop**: FFT sur les 4 beeps 
  UCI (cadence rapide, fréquences calibrées).

- **Calibration automatique**: via dimensions standards vélo BMX 
  (roue 20" = 508mm).

- **Export Kinovea**: génération fichier .kva pour utilisateurs avancés.

### Tier 5 - Machine learning custom

- **Modèle prédictif "score de départ"**: régression temps de course à 
  partir des métriques. Nécessite 100+ vidéos annotées avec chronos.

- **Coach IA contextualisé**: Claude avec profil utilisateur (âge, 
  niveau, historique), génération plan d'entraînement personnalisé.

- **Détection anomalies biomécaniques**: identifier les patterns 
  d'asymétrie ou de compensation risqués.

## Plan de déploiement en phases

### Phase 1 - Prouver localement (en cours)
- Script Python fonctionnel de bout en bout
- Validation sur 10-20 vidéos diverses
- Itération sur métriques et algo

### Phase 2 - Web MVP minimal (1-2 mois)
- Page Next.js unique: upload → traitement → résultat
- Backend FastAPI sur VPS Hetzner
- Beta invite-only: 10-20 BMXeurs québécois
- Objectif: valider UX et demande réelle

### Phase 3 - Vraie app cloud (2-3 mois)
- Comptes utilisateurs, historique, progression
- Comparaison vidéos de référence
- Jobs async avec notifications
- Paiement Stripe si demande validée

### Phase 4 - Scale
- Mobile: React Native (réutilise 80% du code web)
- Comptes coach (multi-athlètes)
- Partenariats clubs/fédérations (cible initiale: FQSC au Québec, 
  puis Cyclisme Canada)

## Modèle économique envisagé

- **Freemium**: 1 analyse/mois gratuite, illimité à ~10-15$/mois
- **Tier Coach**: ~40$/mois pour gérer 10+ athlètes
- **Partenariats fédérations**: licence annuelle club ou fédération
- **Partenariats recherche**: fourniture outil en échange de données 
  de validation scientifique

## Moat stratégique

1. **Spécificité BMX**: les outils génériques ne rivaliseront pas sur 
   la pertinence des métriques.

2. **Base de données propriétaire**: à 1000+ utilisateurs, plus de 
   données cinématiques que toute la recherche publiée combinée.

3. **Rigueur scientifique**: fidélité aux frameworks Grigg/Kalichová 
   comme garant de crédibilité.

4. **Connaissance du domaine**: l'auteur est à la fois ingénieur méca 
   (biomécanique de formation), développeur IA, et familier de la 
   scène BMX.

## Pistes de validation scientifique

- Contact **Mathieu Domalain** à l'Institut Pprime (Poitiers) pour 
  collaboration.
- Département kinésiologie de l'**Université de Sherbrooke**.
- **INS Québec** (Institut national du sport).
- **FQSC** (Fédération québécoise des sports cyclistes).

Approche: fournir l'outil logiciel en échange de données de validation 
sur équipements labo (motion capture, plate-forme de force).

## État actuel du projet (avril 2026)

✓ Pipeline complet de pose estimation opérationnel
✓ Tracking multi-personnes avec ByteTrack
✓ Détection auto du côté filmé et du pied avant
✓ Lissage Savitzky-Golay des landmarks
✓ Segmentation en 5 phases Kalichová
✓ Métriques angles articulaires avec benchmarks Grigg
✓ Repo GitHub avec versioning
✓ CLI fonctionnelle