#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────
# BMX Start Analyzer — installation initiale sur VPS Ubuntu 24.04 LTS.
# Usage (en root sur le VPS, après une connexion SSH fraîche) :
#
#   curl -fsSL https://raw.githubusercontent.com/Louispump/bmx-start-analyzer/main/deploy/install.sh | bash
#
# Ce script :
#   1. met à jour le système et installe Docker + Docker Compose
#   2. clone le repo dans /opt/bmx
#   3. crée un user "bmx" non-root et lui donne accès Docker
#   4. te demande les infos (domaine DuckDNS, user/mot de passe Basic Auth)
#   5. génère .env, build l'image, démarre les services
#   6. configure un firewall basique (UFW) — 22/80/443 seuls ouverts
# ────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_URL="https://github.com/Louispump/bmx-start-analyzer.git"
INSTALL_DIR="/opt/bmx"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  BMX Start Analyzer — installation sur ce VPS"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# ── 0. Sanity check ─────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
   echo "❌ Ce script doit être lancé en root (ou via sudo)."
   exit 1
fi

# ── 1. Mise à jour système + paquets de base ────────────────────────────
echo "→ Mise à jour du système..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    ca-certificates curl gnupg git ufw unattended-upgrades

# ── 2. Docker + Docker Compose (depuis le repo officiel) ─────────────────
if ! command -v docker &> /dev/null; then
    echo "→ Installation de Docker..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
    echo "✓ Docker déjà installé"
fi

# ── 3. Firewall basique : SSH + HTTP + HTTPS uniquement ─────────────────
echo "→ Configuration du pare-feu..."
ufw --force reset > /dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# ── 4. Mises à jour de sécurité automatiques ────────────────────────────
echo "→ Activation des mises à jour de sécurité auto..."
dpkg-reconfigure --priority=low unattended-upgrades > /dev/null 2>&1 || true

# ── 5. Clone du repo ────────────────────────────────────────────────────
if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    echo "→ Clone du repo dans $INSTALL_DIR..."
    git clone "$REPO_URL" "$INSTALL_DIR"
else
    echo "→ Repo déjà cloné, on met à jour..."
    git -C "$INSTALL_DIR" pull
fi

cd "$INSTALL_DIR/deploy"

# ── 6. Configuration interactive (.env) ─────────────────────────────────
if [[ ! -f caddy.env ]]; then
    echo ""
    echo "─── Configuration ──────────────────────────────────────────────"
    # IMPORTANT : on lit explicitement sur /dev/tty pour que les prompts
    # marchent même quand le script est piped depuis curl ... | sudo bash.
    read -p "Domaine (ex: bmx-louis.duckdns.org) : " DOMAIN < /dev/tty
    read -p "Utilisateur web (ex: louis) : " BASIC_AUTH_USER < /dev/tty
    read -s -p "Mot de passe web : " BASIC_AUTH_PASS < /dev/tty
    echo ""

    if [[ -z "$DOMAIN" || -z "$BASIC_AUTH_USER" || -z "$BASIC_AUTH_PASS" ]]; then
        echo "❌ Domaine, user et mot de passe sont obligatoires."
        exit 1
    fi

    echo "→ Génération du hash bcrypt..."
    BASIC_AUTH_HASH=$(docker run --rm caddy:2 caddy hash-password --plaintext "$BASIC_AUTH_PASS")

    # On écrit dans caddy.env (PAS .env) pour éviter que docker-compose
    # interprète les `$` du hash bcrypt à la lecture automatique de .env.
    cat > caddy.env <<EOF
DOMAIN=$DOMAIN
BASIC_AUTH_USER=$BASIC_AUTH_USER
BASIC_AUTH_HASH=$BASIC_AUTH_HASH
EOF
    chmod 600 caddy.env
    echo "✓ caddy.env créé"
fi

# ── 7. Build + démarrage ────────────────────────────────────────────────
echo ""
echo "→ Build de l'image Docker (5-10 min la 1ère fois)..."
docker compose build

echo "→ Démarrage des services..."
docker compose up -d

# ── 8. Récap ────────────────────────────────────────────────────────────
DOMAIN=$(grep '^DOMAIN=' caddy.env | cut -d= -f2-)
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ✅ Installation terminée"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Ouvre : https://$DOMAIN"
echo ""
echo "  (Premier accès : Caddy met 10-30s à obtenir le certificat HTTPS.)"
echo ""
echo "  Commandes utiles :"
echo "    cd $INSTALL_DIR/deploy"
echo "    docker compose logs -f app    # logs de l'app"
echo "    docker compose restart        # redémarrer"
echo "    git -C $INSTALL_DIR pull && docker compose up -d --build  # MAJ"
echo ""
