#!/bin/bash
# Script de lancement du Menu CLI Loto/Keno
# Usage: ./lancer_menu.sh

echo "🎲🎰 Lancement du Menu Loto/Keno..."

# Vérification de l'environnement virtuel
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Activation de l'environnement virtuel..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "../venv/bin/activate" ]; then
        source ../venv/bin/activate
    else
        echo "❌ Environnement virtuel non trouvé"
        echo "Créez-le avec: python -m venv venv"
        echo "Puis activez-le avec: source venv/bin/activate"
        exit 1
    fi
fi

# Vérification de Python
if ! command -v python &> /dev/null; then
    echo "❌ Python non trouvé"
    exit 1
fi

# Vérification des dépendances
if ! python -c "import pandas, duckdb, matplotlib" &> /dev/null; then
    echo "⚠️  Installation des dépendances..."
    pip install -r requirements.txt
fi

# Lancement du menu
echo "🚀 Démarrage du menu interactif..."
python cli_menu.py
