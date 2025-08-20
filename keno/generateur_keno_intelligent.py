#!/usr/bin/env python3
"""
Générateur de grilles Keno basé sur l'analyse FDJ

Ce script génère des grilles Keno en utilisant les analyses 
statistiques des tirages précédents.
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Ajout du path parent pour accéder aux modules
sys.path.append(str(Path(__file__).parent.parent))

# Import des modules locaux
from grilles.generateur_grilles import GenerateurGrilles
import pandas as pd
import numpy as np
from collections import Counter

def analyser_donnees_keno():
    """Analyse les données Keno et retourne les numéros recommandés"""
    from pathlib import Path
    
    # Charger les données
    KENO_DATA_DIR = Path(__file__).parent / "keno_data"
    csv_files = list(KENO_DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        return None, None
    
    try:
        df = pd.read_csv(csv_files[0], delimiter=';')
        ball_columns = [f'boule{i}' for i in range(1, 21)]
        
        # Analyser les fréquences
        all_numbers = Counter()
        for col in ball_columns:
            if col in df.columns:
                for num in df[col].dropna():
                    if pd.notna(num):
                        all_numbers[int(num)] += 1
        
        # Analyser les retards (dernière apparition)
        recent_draws = df.head(100)
        last_seen = {}
        
        for num in range(1, 71):
            found = False
            for idx in range(len(recent_draws)):
                row = recent_draws.iloc[idx]
                for col in ball_columns:
                    if col in row and pd.notna(row[col]) and int(row[col]) == num:
                        last_seen[num] = idx
                        found = True
                        break
                if found:
                    break
            if not found:
                last_seen[num] = 100
        
        # Extraire les numéros chauds et en retard
        most_common = all_numbers.most_common(15)
        delayed_numbers = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)[:15]
        
        hot_numbers = [num for num, count in most_common]
        cold_numbers = [num for num, delay in delayed_numbers]
        
        return hot_numbers, cold_numbers
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        return None, None

def generer_grilles_keno_intelligentes(nb_grilles=5, taille_grille=10, utiliser_analyse=True):
    """Génère des grilles Keno basées sur l'analyse statistique"""
    
    print("🎯 GÉNÉRATEUR DE GRILLES KENO INTELLIGENT")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"🎲 Génération de {nb_grilles} grilles de {taille_grille} numéros")
    print(f"🧠 Analyse intelligente: {'Activée' if utiliser_analyse else 'Désactivée'}")
    print("-"*60)
    
    # 1. Analyse des données (simulées pour la démo)
    numeros_recommandes = []
    
    if utiliser_analyse:
        print("\n📊 Analyse des tirages précédents...")
        
        # Analyser les données réelles
        hot_numbers, cold_numbers = analyser_donnees_keno()
        
        if hot_numbers is not None and cold_numbers is not None:
            # Combiner pour avoir une liste de numéros recommandés
            numeros_recommandes.extend(hot_numbers[:10])
            for numero in cold_numbers[:10]:
                if numero not in numeros_recommandes:
                    numeros_recommandes.append(numero)
            
            print(f"✅ {len(numeros_recommandes)} numéros recommandés identifiés")
            print(f"🔥 Numéros chauds: {hot_numbers[:5]}")
            print(f"❄️ Numéros en retard: {cold_numbers[:5]}")
        else:
            print("⚠️ Impossible de charger les données, génération aléatoire")
            utiliser_analyse = False
    
    # 2. Génération des grilles
    print(f"\n🎲 Génération de {nb_grilles} grilles...")
    
    # Créer le générateur pour le Keno
    numeros_keno = list(range(1, 71))  # Numéros Keno 1-70
    
    grilles = []
    
    for i in range(nb_grilles):
        if utiliser_analyse and numeros_recommandes and i < 3:
            # Pour les 3 premières grilles, utiliser l'analyse
            grille = generer_grille_avec_analyse(numeros_recommandes, taille_grille)
        else:
            # Grilles aléatoires pour la diversité
            import random
            grille = random.sample(numeros_keno, taille_grille)
        
        grilles.append(sorted(grille))
        
        # Afficher la grille
        numeros_str = ' - '.join(f'{num:2d}' for num in sorted(grille))
        strategie = "🧠 Analysée" if (utiliser_analyse and numeros_recommandes and i < 3) else "🎲 Aléatoire"
        print(f"   Grille {i+1:2d}: {numeros_str} {strategie}")
    
    # 3. Statistiques et conseils
    print(f"\n📈 Statistiques des grilles générées:")
    
    # Calculer la répartition par zones
    zones = {'1-23': 0, '24-46': 0, '47-70': 0}
    total_numeros = sum(len(grille) for grille in grilles)
    
    for grille in grilles:
        for numero in grille:
            if 1 <= numero <= 23:
                zones['1-23'] += 1
            elif 24 <= numero <= 46:
                zones['24-46'] += 1
            else:
                zones['47-70'] += 1
    
    for zone, count in zones.items():
        pourcentage = (count / total_numeros) * 100
        print(f"   Zone {zone}: {count:2d} numéros ({pourcentage:.1f}%)")
    
    # Conseils
    print(f"\n💡 Conseils de jeu:")
    print(f"   • Variez vos mises selon votre budget")
    print(f"   • Les grilles analysées utilisent les tendances récentes")
    print(f"   • Les grilles aléatoires apportent de la diversité")
    print(f"   • Aucune méthode ne garantit un gain")
    
    return grilles

def generer_grille_avec_analyse(numeros_recommandes, taille_grille):
    """Génère une grille en utilisant l'analyse statistique"""
    
    import random
    
    grille = set()
    
    # 60% de la grille avec des numéros recommandés
    nb_recommandes = int(taille_grille * 0.6)
    recommandes_choisis = random.sample(numeros_recommandes[:20], 
                                      min(nb_recommandes, len(numeros_recommandes)))
    grille.update(recommandes_choisis)
    
    # Compléter avec des numéros aléatoires
    while len(grille) < taille_grille:
        numero = random.randint(1, 70)
        grille.add(numero)
    
    return list(grille)

def main():
    """Fonction principale"""
    
    parser = argparse.ArgumentParser(
        description="Générateur de grilles Keno intelligent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python generateur_keno_intelligent.py                     # 5 grilles de 10 numéros
  python generateur_keno_intelligent.py --grilles 10        # 10 grilles
  python generateur_keno_intelligent.py --taille 8          # Grilles de 8 numéros
  python generateur_keno_intelligent.py --sans-analyse      # Mode purement aléatoire
        """
    )
    
    parser.add_argument(
        '--grilles',
        type=int,
        default=5,
        help='Nombre de grilles à générer (défaut: 5)'
    )
    
    parser.add_argument(
        '--taille',
        type=int,
        default=10,
        choices=range(3, 21),
        help='Nombre de numéros par grille (3-20, défaut: 10)'
    )
    
    parser.add_argument(
        '--sans-analyse',
        action='store_true',
        help='Génération purement aléatoire (sans analyse)'
    )
    
    args = parser.parse_args()
    
    try:
        grilles = generer_grilles_keno_intelligentes(
            nb_grilles=args.grilles,
            taille_grille=args.taille,
            utiliser_analyse=not args.sans_analyse
        )
        
        print(f"\n✅ {len(grilles)} grilles générées avec succès!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
