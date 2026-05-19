# Déploiement BMX Start Analyzer

App FastAPI + YOLO empaquetée en Docker, derrière Caddy (HTTPS auto + Basic Auth).

## Pré-requis

- Un VPS Ubuntu 24.04 LTS (testé sur OVH VPS-1 : 4 vCore / 8 GB RAM / 75 GB SSD)
- Un sous-domaine pointant vers l'IP du VPS (DuckDNS gratuit, ou un domaine acheté)

## Étape 1 — Sous-domaine DuckDNS (gratuit, 2 min)

1. Va sur https://www.duckdns.org → connecte-toi (GitHub/Google)
2. Choisis un sous-domaine (ex: `bmx-louis`) → ça donne `bmx-louis.duckdns.org`
3. Dans la case **"current ip"**, mets l'IP de ton VPS OVH → bouton **update ip**
4. Note ton **token** affiché en haut (pas nécessaire pour la suite, juste pour mémoire)

> ℹ️ L'IP OVH est fixe à vie pour ton VPS, pas besoin de cron de mise à jour.

## Étape 2 — Connexion SSH au VPS

Depuis ton Mac, dans un Terminal :

```bash
ssh root@TON_IP_VPS
```

Le mot de passe est dans l'email de bienvenue OVH. À la 1ère connexion, OVH va te demander de le changer.

> 💡 Pour ne plus taper le mot de passe : on configurera une clé SSH après l'install.

## Étape 3 — Installation automatique (1 commande)

Sur le VPS, en root :

```bash
curl -fsSL https://raw.githubusercontent.com/Louispump/bmx-start-analyzer/main/deploy/install.sh | bash
```

Le script va te demander :
- **Domaine** : `bmx-louis.duckdns.org`
- **E-mail** : ton email (pour notifs Let's Encrypt, optionnel)
- **Utilisateur web** : `louis` (ce que tu veux)
- **Mot de passe web** : choisis-le (Safari le retient à la 1ère connexion)

Puis il installe Docker, build l'image (5-10 min la 1ère fois), démarre tout.

À la fin, ouvre dans Safari iPad :

```
https://bmx-louis.duckdns.org
```

## Modifier l'app après déploiement

### Workflow normal

Depuis ton Mac :
```bash
# tu codes, tu testes localement
git push origin main
```

Sur le VPS (1 commande) :
```bash
ssh root@TON_IP "cd /opt/bmx && git pull && cd deploy && docker compose up -d --build"
```

Le rebuild est rapide grâce au cache Docker (10-60 s selon ce qui a changé).

### Logs en direct

```bash
ssh root@TON_IP "cd /opt/bmx/deploy && docker compose logs -f app"
```

### Redémarrer sans modifier le code

```bash
ssh root@TON_IP "cd /opt/bmx/deploy && docker compose restart"
```

## Sauvegarde des vidéos

Les dossiers persistés sont :
- `/opt/bmx/uploads` (vidéos originales)
- `/opt/bmx/output` (vidéos annotées + CSV + DBs JSON)

OVH fait une sauvegarde quotidienne automatique (incluse dans le plan VPS-1).

Pour une sauvegarde manuelle sur ton Mac :
```bash
rsync -avz root@TON_IP:/opt/bmx/uploads ~/bmx-backup/
rsync -avz root@TON_IP:/opt/bmx/output  ~/bmx-backup/
```

## Désinstaller / résilier

1. Sur manager.ovhcloud.com → désactiver le renouvellement du VPS
2. (Avant la date d'échéance) Récupérer les vidéos avec `rsync` (voir ci-dessus)
3. Laisser OVH supprimer le VPS à l'échéance

Le code reste sur GitHub, redéployable n'importe quand.
