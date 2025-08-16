#!/bin/bash
# Script de lancement pour le générateur Keno

echo "🎲 Générateur Intelligent de Grilles Keno v2.0"
echo "================================================"

# Répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Vérification de l'environnement virtuel
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Activation de l'environnement virtuel..."
    source ../venv/bin/activate
fi

# Menu de choix
echo ""
echo "Choisissez une option:"
echo "1. Test rapide (5 grilles)"
echo "2. Génération standard (50 grilles)"
echo "3. Génération complète (100 grilles)"
echo "4. Réentraîner les modèles et générer"
echo "5. Mode silencieux (10 grilles)"
echo ""

read -p "Votre choix (1-5): " choice

case $choice in
    1)
        echo "🚀 Test rapide..."
        python keno_generator_advanced.py --quick
        ;;
    2)
        echo "🚀 Génération standard..."
        python keno_generator_advanced.py --grids 50
        ;;
    3)
        echo "🚀 Génération complète..."
        python keno_generator_advanced.py --grids 100
        ;;
    4)
        echo "🚀 Réentraînement + génération..."
        python keno_generator_advanced.py --retrain --grids 50
        ;;
    5)
        echo "🚀 Mode silencieux..."
        python keno_generator_advanced.py --grids 10 --silent
        ;;
    *)
        echo "❌ Choix invalide. Utilisation du mode par défaut."
        python keno_generator_advanced.py --quick
        ;;
esac

echo ""
echo "✅ Terminé ! Vérifiez le dossier keno_output pour les résultats."
