#!/bin/bash
# Script de démarrage automatique du serveur Flask

cd /home/nvernot/projets/loto_keno
source venv/bin/activate

echo "🚀 Démarrage du serveur Loto/Keno..."
echo "🌐 Dashboard disponible sur: http://localhost:5000"
echo "⚡ Appuyez sur Ctrl+C pour arrêter"
echo ""

# Fonction pour gérer l'arrêt propre
cleanup() {
    echo ""
    echo "🛑 Arrêt du serveur..."
    exit 0
}

# Intercepter Ctrl+C
trap cleanup SIGINT

# Démarrer le serveur
python api/app.py
