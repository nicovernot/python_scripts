#!/bin/bash

# Script de lancement de l'API Flask
# =================================

# Configuration
PORT=${1:-5000}
HOST=${2:-"127.0.0.1"}
DEBUG=${3:-"true"}

echo "🚀 Lancement de l'API Loto/Keno"
echo "================================="
echo "Port: $PORT"
echo "Host: $HOST"
echo "Debug: $DEBUG"
echo ""

# Vérification Python
python_path=$(which python3)
if [ -z "$python_path" ]; then
    echo "❌ Python 3 non trouvé"
    exit 1
fi

echo "✅ Python: $python_path"

# Vérification des dépendances
echo "🔍 Vérification des dépendances..."

required_packages=("flask" "flask-cors" "duckdb" "pandas")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "❌ Packages manquants: ${missing_packages[*]}"
    echo "📦 Installation des dépendances..."
    pip install flask flask-cors
    if [ $? -ne 0 ]; then
        echo "❌ Échec de l'installation des dépendances"
        exit 1
    fi
fi

echo "✅ Toutes les dépendances sont installées"

# Vérification du fichier de configuration
if [ ! -f ".env" ]; then
    echo "⚠️ Fichier .env manquant - Utilisation des valeurs par défaut"
fi

# Vérification de la structure API
if [ ! -f "api/app.py" ]; then
    echo "❌ Fichier API principal manquant (api/app.py)"
    exit 1
fi

echo "✅ Structure API valide"

# Variables d'environnement Flask
export FLASK_APP=api/app.py
export FLASK_ENV=development
export FLASK_DEBUG=$DEBUG

# Lancement de l'API
echo ""
echo "🌐 Lancement du serveur Flask..."
echo "   URL: http://$HOST:$PORT"
echo "   Documentation: http://$HOST:$PORT/"
echo ""
echo "   Endpoints principaux:"
echo "   - POST /api/loto/generate    : Génération de grilles Loto"
echo "   - POST /api/keno/analyze     : Analyse Keno"
echo "   - POST /api/data/update      : Mise à jour des données"
echo "   - GET  /api/health           : Statut de santé"
echo ""
echo "⏹️  Appuyez sur Ctrl+C pour arrêter"
echo ""

# Lancement
cd "$(dirname "$0")"
python3 -m flask run --host=$HOST --port=$PORT

# Cleanup
echo ""
echo "🔄 API arrêtée"
