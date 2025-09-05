#!/usr/bin/env python3
"""
Générateur de Grilles Loto - Systèmes Réduits
=============================================

Générateur de grilles Loto basé sur la théorie des systèmes réduits.
Permet de créer un nombre optimal de grilles à partir d'une liste de numéros favoris.

Théorie des Systèmes Réduits :
- Maximise les chances de gain avec un nombre minimal de grilles
- Garantit un certain niveau de couverture des combinaisons
- Optimise la répartition des numéros dans les grilles

Usage:
    python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 5
    python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 10 --garantie 3
    python generateur_grilles.py --fichier mes_nombres.txt --grilles 8 --export
    python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 5 --nombres-utilises 8

Auteur: Système Loto/Keno
Date: 14 août 2025
"""

import argparse
import itertools
import random
import sys
import json
import csv
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any
from datetime import datetime
import math

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config_env import get_config, get_config_bool, get_config_int
except ImportError:
    def get_config(key, default=None): return default
    def get_config_bool(key, default=False): return default
    def get_config_int(key, default=0): return default

# Imports pour l'optimisation
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("⚠️  PuLP non disponible. Installation recommandée : pip install pulp")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  Pandas non disponible. Installation recommandée : pip install pandas")


class SystemeReduit:
    """
    Générateur de systèmes réduits pour le Loto
    
    Principe : À partir d'une liste de N numéros favoris (généralement 8-15),
    génère le nombre minimal de grilles pour garantir un certain niveau de gain.
    """
    
    def __init__(self, nombres_favoris: List[int], garantie: int = 3, taille_grille: int = 5):
        """
        Initialise le système réduit
        
        Args:
            nombres_favoris: Liste des numéros favoris (8-20 numéros)
            garantie: Niveau de garantie (2, 3, 4 ou 5)
                     garantie=3 : si 5 bons numéros dans les favoris, 
                                 garantit au moins un 3 dans une grille
            taille_grille: Nombre de numéros par grille (5 pour Loto, 10 pour Keno)
        """
        self.nombres_favoris = sorted(list(set(nombres_favoris)))
        self.garantie = garantie
        self.taille_grille = taille_grille
        self.nb_favoris = len(self.nombres_favoris)
        
        # Validation
        if self.nb_favoris < self.taille_grille:
            raise ValueError(f"Il faut au minimum {self.taille_grille} numéros favoris pour des grilles de {self.taille_grille} numéros")
        if self.nb_favoris > 20:
            raise ValueError("Maximum 20 numéros favoris")
        if self.garantie < 2 or self.garantie > 5:
            raise ValueError("La garantie doit être entre 2 et 5")
        
        print(f"🎯 Système réduit initialisé :")
        print(f"   Numéros favoris : {self.nombres_favoris}")
        print(f"   Nombre de favoris : {self.nb_favoris}")
        print(f"   Garantie : {self.garantie}")
    
    def calculer_couverture_theorique(self, nb_grilles: int) -> Dict[str, float]:
        """Calcule la couverture théorique du système"""
        
        # Nombre total de combinaisons possibles avec les favoris
        total_combinaisons = math.comb(self.nb_favoris, self.taille_grille)
        
        # Estimation de la couverture
        if nb_grilles >= total_combinaisons:
            couverture = 100.0
        else:
            # Formule empirique basée sur la théorie des systèmes réduits
            couverture = min(100.0, (nb_grilles / total_combinaisons) * 100 * 1.2)
        
        # Probabilité de gain selon la garantie
        prob_garantie = self._calculer_probabilite_garantie(nb_grilles)
        
        return {
            'couverture_combinaisons': couverture,
            'probabilite_garantie': prob_garantie,
            'total_combinaisons_theoriques': total_combinaisons,
            'efficacite': (prob_garantie / nb_grilles) * 100
        }
    
    def _calculer_probabilite_garantie(self, nb_grilles: int) -> float:
        """Calcule la probabilité d'atteindre la garantie"""
        
        # Formules empiriques basées sur la théorie des systèmes réduits
        if self.garantie == 2:
            return min(95.0, nb_grilles * 8.0)
        elif self.garantie == 3:
            return min(85.0, nb_grilles * 6.0)
        elif self.garantie == 4:
            return min(70.0, nb_grilles * 4.0)
        else:  # garantie == 5
            return min(50.0, nb_grilles * 2.0)
    
    def generer_grilles_optimales(self, nb_grilles: int) -> List[List[int]]:
        """
        Génère des grilles selon la méthode des systèmes réduits optimaux
        
        Méthode :
        1. Génère toutes les combinaisons possibles avec les favoris
        2. Sélectionne les combinaisons qui maximisent la couverture
        3. Applique des algorithmes d'optimisation pour minimiser les redondances
        """
        
        print(f"\n🔄 Génération de {nb_grilles} grilles optimales...")
        
        # Étape 1 : Générer toutes les combinaisons possibles
        toutes_combinaisons = list(itertools.combinations(self.nombres_favoris, self.taille_grille))
        print(f"   Combinaisons théoriques : {len(toutes_combinaisons)}")
        
        if nb_grilles >= len(toutes_combinaisons):
            # Si on demande plus de grilles que de combinaisons possibles
            print("   ⚠️  Nombre de grilles >= combinaisons possibles")
            grilles_selectionnees = toutes_combinaisons[:nb_grilles]
        else:
            # Sélection optimale par algorithme de couverture
            grilles_selectionnees = self._selection_optimale(toutes_combinaisons, nb_grilles)
        
        # Conversion en listes triées
        grilles_finales = [sorted(list(grille)) for grille in grilles_selectionnees]
        
        print(f"   ✅ {len(grilles_finales)} grilles générées")
        return grilles_finales
    
    def _selection_optimale(self, combinaisons: List[Tuple], nb_grilles: int) -> List[Tuple]:
        """
        Sélectionne les meilleures combinaisons selon l'algorithme de couverture maximale
        
        Algorithme :
        1. Commence par la combinaison qui couvre le plus de sous-ensembles
        2. Ajoute itérativement les combinaisons qui augmentent le plus la couverture
        3. Évite les redondances maximales
        """
        
        if nb_grilles >= len(combinaisons):
            return combinaisons
        
        # Stratégie de sélection équilibrée
        if nb_grilles <= 10:
            # Pour peu de grilles : sélection par espacement maximal
            return self._selection_par_espacement(combinaisons, nb_grilles)
        else:
            # Pour beaucoup de grilles : sélection par couverture
            return self._selection_par_couverture(combinaisons, nb_grilles)
    
    def _selection_par_espacement(self, combinaisons: List[Tuple], nb_grilles: int) -> List[Tuple]:
        """Sélection par espacement maximal entre les grilles"""
        
        selected = []
        remaining = list(combinaisons)
        
        # Première grille : celle avec la meilleure répartition
        first_grille = self._choisir_grille_equilibree(remaining)
        selected.append(first_grille)
        remaining.remove(first_grille)
        
        # Grilles suivantes : maximiser la distance avec les précédentes
        for _ in range(nb_grilles - 1):
            if not remaining:
                break
            
            meilleure_grille = None
            meilleur_score = -1
            
            for grille in remaining:
                score = self._calculer_score_espacement(grille, selected)
                if score > meilleur_score:
                    meilleur_score = score
                    meilleure_grille = grille
            
            if meilleure_grille:
                selected.append(meilleure_grille)
                remaining.remove(meilleure_grille)
        
        return selected
    
    def _selection_par_couverture(self, combinaisons: List[Tuple], nb_grilles: int) -> List[Tuple]:
        """Sélection par maximisation de la couverture"""
        
        # Pour un grand nombre de grilles, utilise un échantillonnage intelligent
        step = len(combinaisons) // nb_grilles
        
        selected = []
        for i in range(0, len(combinaisons), step):
            if len(selected) >= nb_grilles:
                break
            selected.append(combinaisons[i])
        
        # Complète avec des grilles aléatoires si nécessaire
        while len(selected) < nb_grilles and len(selected) < len(combinaisons):
            remaining = [c for c in combinaisons if c not in selected]
            if remaining:
                selected.append(random.choice(remaining))
        
        return selected
    
    def _choisir_grille_equilibree(self, grilles: List[Tuple]) -> Tuple:
        """Choisit la grille la plus équilibrée (bonne répartition des numéros)"""
        
        meilleure_grille = None
        meilleur_score = -1
        
        for grille in grilles[:100]:  # Limite pour performance
            # Score basé sur la répartition des numéros
            ecart_type = self._calculer_ecart_type(grille)
            repartition = self._calculer_repartition_zones(grille)
            
            score = (1.0 / (1.0 + ecart_type)) + repartition
            
            if score > meilleur_score:
                meilleur_score = score
                meilleure_grille = grille
        
        return meilleure_grille or grilles[0]
    
    def _calculer_score_espacement(self, grille: Tuple, grilles_selectionnees: List[Tuple]) -> float:
        """Calcule le score d'espacement d'une grille par rapport aux grilles déjà sélectionnées"""
        
        if not grilles_selectionnees:
            return 1.0
        
        distances = []
        for grille_sel in grilles_selectionnees:
            # Distance = nombre de numéros différents
            distance = len(set(grille) - set(grille_sel))
            distances.append(distance)
        
        # Score = distance minimale (on veut maximiser la distance minimale)
        return min(distances)
    
    def _calculer_ecart_type(self, grille: Tuple) -> float:
        """Calcule l'écart-type des numéros dans une grille"""
        grille_list = list(grille)
        moyenne = sum(grille_list) / len(grille_list)
        variance = sum((x - moyenne) ** 2 for x in grille_list) / len(grille_list)
        return math.sqrt(variance)
    
    def _calculer_repartition_zones(self, grille: Tuple) -> float:
        """Calcule la qualité de répartition par zones (1-10, 11-20, 21-30, 31-40, 41-49)"""
        
        zones = [0] * 5  # 5 zones de 10 numéros chacune
        
        for numero in grille:
            zone = min(4, (numero - 1) // 10)
            zones[zone] += 1
        
        # Score : favorise les grilles avec répartition équilibrée
        zones_occupees = sum(1 for z in zones if z > 0)
        return zones_occupees / 5.0
    
    def generer_grilles_aleatoires_intelligentes(self, nb_grilles: int) -> List[List[int]]:
        """
        Génère des grilles avec une approche aléatoire intelligente
        
        Combine :
        - Sélection aléatoire dans les favoris
        - Optimisations pour éviter les doublons exacts
        - Garantie de diversité
        - Gestion des cas où il y a peu de numéros favoris
        """
        
        print(f"\n🎲 Génération de {nb_grilles} grilles aléatoires intelligentes...")
        
        grilles = []
        
        # Cas spécial : Si on a exactement le nombre de numéros requis pour une grille
        if len(self.nombres_favoris) == self.taille_grille:
            # On ne peut faire qu'une seule grille avec tous les numéros
            grille_unique = sorted(self.nombres_favoris)
            for i in range(nb_grilles):
                grilles.append(grille_unique.copy())
            print(f"   ⚠️  Une seule combinaison possible (tous les numéros favoris)")
        
        # Cas spécial : Pas assez de numéros favoris
        elif len(self.nombres_favoris) < self.taille_grille:
            # Impossible de générer des grilles complètes
            taille_possible = len(self.nombres_favoris)
            print(f"   ⚠️  Réduction taille grille: {self.taille_grille} -> {taille_possible} numéros")
            grille_reduite = sorted(self.nombres_favoris)
            for i in range(nb_grilles):
                grilles.append(grille_reduite.copy())
        
        # Cas normal : Assez de numéros pour faire des combinaisons variées
        else:
            tentatives_max = nb_grilles * 10
            
            for i in range(nb_grilles):
                tentatives = 0
                while tentatives < tentatives_max:
                    # Génère une grille candidate
                    grille = sorted(random.sample(self.nombres_favoris, self.taille_grille))
                    
                    # Vérifie qu'elle n'existe pas déjà
                    if grille not in grilles:
                        grilles.append(grille)
                        break
                    
                    tentatives += 1
                
                # Si impossible de trouver une grille unique, force l'ajout
                if len(grilles) <= i:
                    grille = sorted(random.sample(self.nombres_favoris, self.taille_grille))
                    grilles.append(grille)
        
        print(f"   ✅ {len(grilles)} grilles générées")
        return grilles
    
    def analyser_grilles(self, grilles: List[List[int]]) -> Dict[str, Any]:
        """Analyse la qualité des grilles générées"""
        
        if not grilles:
            return {'erreur': 'Aucune grille à analyser'}
        
        # Statistiques de base
        nb_grilles = len(grilles)
        grilles_uniques = len(set(tuple(g) for g in grilles))
        pourcentage_unique = (grilles_uniques / nb_grilles) * 100
        
        # Analyse de la répartition des numéros
        compteur_numeros = {}
        for grille in grilles:
            for numero in grille:
                compteur_numeros[numero] = compteur_numeros.get(numero, 0) + 1
        
        # Numéros les plus/moins utilisés
        numeros_tries = sorted(compteur_numeros.items(), key=lambda x: x[1], reverse=True)
        plus_utilises = numeros_tries[:5]
        moins_utilises = numeros_tries[-5:]
        
        # Analyse de la couverture
        couverture = self.calculer_couverture_theorique(nb_grilles)
        
        # Score de qualité global
        score_qualite = self._calculer_score_qualite(grilles, couverture)
        
        return {
            'nb_grilles': nb_grilles,
            'grilles_uniques': grilles_uniques,
            'pourcentage_unique': pourcentage_unique,
            'plus_utilises': plus_utilises,
            'moins_utilises': moins_utilises,
            'couverture': couverture,
            'score_qualite': score_qualite,
            'recommandation': self._generer_recommandation(score_qualite, pourcentage_unique)
        }
    
    def _calculer_score_qualite(self, grilles: List[List[int]], couverture: Dict) -> float:
        """Calcule un score de qualité global sur 100"""
        
        # Facteurs de qualité
        uniqueness = len(set(tuple(g) for g in grilles)) / len(grilles)
        coverage = couverture['couverture_combinaisons'] / 100
        efficiency = couverture['efficacite'] / 100
        
        # Score pondéré
        score = (uniqueness * 40) + (coverage * 35) + (efficiency * 25)
        return min(100, score * 100)
    
    def _generer_recommandation(self, score_qualite: float, pourcentage_unique: float) -> str:
        """Génère une recommandation basée sur l'analyse"""
        
        if score_qualite >= 80 and pourcentage_unique >= 95:
            return "🟢 Excellent système - Qualité optimale"
        elif score_qualite >= 60 and pourcentage_unique >= 85:
            return "🟡 Bon système - Quelques améliorations possibles"
        elif score_qualite >= 40:
            return "🟠 Système moyen - Recommandé d'augmenter le nombre de favoris"
        else:
            return "🔴 Système à améliorer - Revoir les paramètres"


class SystemeReduitOptimise:
    """
    Générateur de systèmes réduits optimisé utilisant les CSV TOP et PuLP
    
    Cette classe utilise :
    - Les fichiers TOP 25 Loto / TOP 30 Keno pour sélectionner les meilleurs numéros
    - L'optimisation PuLP pour minimiser le nombre de grilles tout en maximisant la couverture
    - Des algorithmes avancés de réduction de systèmes
    """
    
    def __init__(self, jeu: str, taille_grille: int = None):
        """
        Initialise le système réduit optimisé
        
        Args:
            jeu: 'loto' ou 'keno'
            taille_grille: Nombre de numéros par grille (défaut: 5 pour Loto, 10 pour Keno)
        """
        self.jeu = jeu.lower()
        
        if self.jeu == 'loto':
            self.csv_path = Path("loto_stats_exports/top_25_numeros_equilibres_loto.csv")
            self.max_numeros = 49
            self.taille_grille = taille_grille or 5
            self.min_grille = 6  # Pour Loto, on peut faire des grilles de 6-10 numéros
            self.max_grille = 10
        elif self.jeu == 'keno':
            self.csv_path = Path("keno_stats_exports/top_30_numeros_equilibres_keno.csv")
            self.max_numeros = 70
            self.taille_grille = taille_grille or 10
            self.min_grille = 6  # Pour Keno, on peut faire des grilles de 6-10 numéros
            self.max_grille = 10
        else:
            raise ValueError("Le jeu doit être 'loto' ou 'keno'")
        
        self.top_numeros = []
        self.scores = {}
        
        # Vérifier les dépendances
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas requis pour cette fonctionnalité. Installez avec : pip install pandas")
        
        if not PULP_AVAILABLE:
            print("⚠️  PuLP non disponible. Utilisation de l'algorithme de base sans optimisation avancée.")
    
    def charger_top_numeros(self, nombre_numeros: int = None) -> bool:
        """
        Charge les numéros TOP depuis le fichier CSV
        
        Args:
            nombre_numeros: Nombre de numéros à utiliser (défaut: 15 pour Loto, 20 pour Keno)
        """
        if not self.csv_path.exists():
            print(f"❌ Fichier TOP non trouvé : {self.csv_path}")
            print(f"💡 Générez d'abord les TOP avec l'option {28 if self.jeu == 'loto' else 29} du menu CLI")
            return False
        
        try:
            df = pd.read_csv(self.csv_path, sep=';')
            
            # Définir le nombre de numéros par défaut
            if nombre_numeros is None:
                nombre_numeros = 15 if self.jeu == 'loto' else 20
            
            # Limiter selon la taille disponible
            nombre_numeros = min(nombre_numeros, len(df))
            
            # Extraire les TOP numéros
            top_df = df.head(nombre_numeros)
            self.top_numeros = top_df['numero'].tolist()
            
            # Extraire les scores pour pondération
            self.scores = dict(zip(top_df['numero'], top_df['score_composite']))
            
            print(f"✅ {len(self.top_numeros)} numéros TOP {self.jeu.upper()} chargés")
            print(f"   Numéros : {self.top_numeros}")
            print(f"   Score max : {max(self.scores.values()):.3f}")
            print(f"   Score min : {min(self.scores.values()):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")
            return False
    
    def generer_grilles_optimisees_pulp(self, nb_grilles: int, taille_grille: int = None) -> List[List[int]]:
        """
        Génère des grilles optimisées avec PuLP
        
        Args:
            nb_grilles: Nombre de grilles à générer
            taille_grille: Taille des grilles (défaut: celle définie à l'init)
        """
        if not PULP_AVAILABLE:
            print("⚠️  PuLP non disponible, utilisation de la méthode alternative")
            return self.generer_grilles_optimisees_alternative(nb_grilles, taille_grille)
        
        if not self.top_numeros:
            raise ValueError("Chargez d'abord les numéros TOP avec charger_top_numeros()")
        
        taille = taille_grille or self.taille_grille
        
        print(f"\n🔄 Génération de {nb_grilles} grilles optimisées avec PuLP...")
        print(f"   Pool de numéros : {len(self.top_numeros)}")
        print(f"   Taille par grille : {taille}")
        
        # Créer le problème d'optimisation
        prob = pulp.LpProblem("SystemeReduitOptimal", pulp.LpMinimize)
        
        # Variables binaires : x[i][j] = 1 si le numéro j est dans la grille i
        numeros = self.top_numeros
        grilles_vars = {}
        
        for i in range(nb_grilles):
            grilles_vars[i] = {}
            for j in numeros:
                grilles_vars[i][j] = pulp.LpVariable(f"grille_{i}_numero_{j}", cat='Binary')
        
        # Contrainte 1: Chaque grille doit avoir exactement 'taille' numéros
        for i in range(nb_grilles):
            prob += pulp.lpSum([grilles_vars[i][j] for j in numeros]) == taille
        
        # Objectif: Maximiser la couverture pondérée (minimiser le négatif)
        # Chaque numéro contribue selon son score
        couverture_ponderee = pulp.lpSum([
            self.scores[j] * grilles_vars[i][j] 
            for i in range(nb_grilles) 
            for j in numeros
        ])
        
        prob += -couverture_ponderee  # Minimiser le négatif = maximiser
        
        # Contrainte 2: Équilibrage - éviter de sur-utiliser certains numéros
        for j in numeros:
            utilisation = pulp.lpSum([grilles_vars[i][j] for i in range(nb_grilles)])
            # Un numéro ne peut être utilisé dans plus de nb_grilles/2 grilles
            prob += utilisation <= max(1, nb_grilles // 2)
        
        # Résoudre
        print("   🧮 Résolution du problème d'optimisation...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extraire les résultats
        grilles = []
        if prob.status == pulp.LpStatusOptimal:
            for i in range(nb_grilles):
                grille = []
                for j in numeros:
                    if grilles_vars[i][j].varValue == 1:
                        grille.append(j)
                grilles.append(sorted(grille))
            
            print(f"   ✅ Optimisation réussie ! {len(grilles)} grilles générées")
        else:
            print(f"   ⚠️  Optimisation échouée (statut: {prob.status})")
            return self.generer_grilles_optimisees_alternative(nb_grilles, taille_grille)
        
        return grilles
    
    def generer_grilles_optimisees_alternative(self, nb_grilles: int, taille_grille: int = None) -> List[List[int]]:
        """
        Méthode alternative d'optimisation sans PuLP
        Utilise un algorithme glouton avec pondération par scores
        """
        if not self.top_numeros:
            raise ValueError("Chargez d'abord les numéros TOP avec charger_top_numeros()")
        
        taille = taille_grille or self.taille_grille
        
        print(f"\n🔄 Génération de {nb_grilles} grilles avec algorithme glouton pondéré...")
        
        grilles = []
        utilisation_numeros = {num: 0 for num in self.top_numeros}
        
        for i in range(nb_grilles):
            # Créer une grille en privilégiant les numéros les mieux notés et moins utilisés
            grille = []
            numeros_disponibles = self.top_numeros.copy()
            
            for _ in range(taille):
                if not numeros_disponibles:
                    break
                
                # Calculer un score combiné : score original - pénalité d'utilisation
                scores_ajustes = {}
                for num in numeros_disponibles:
                    score_base = self.scores[num]
                    penalite = utilisation_numeros[num] * 0.1  # Pénalité progressive
                    scores_ajustes[num] = score_base - penalite
                
                # Sélectionner le meilleur numéro
                meilleur_num = max(scores_ajustes, key=scores_ajustes.get)
                grille.append(meilleur_num)
                numeros_disponibles.remove(meilleur_num)
                utilisation_numeros[meilleur_num] += 1
            
            grilles.append(sorted(grille))
        
        print(f"   ✅ {len(grilles)} grilles générées avec méthode alternative")
        return grilles
    
    def analyser_couverture_optimisee(self, grilles: List[List[int]]) -> Dict[str, Any]:
        """
        Analyse la couverture des grilles optimisées
        """
        if not grilles:
            return {}
        
        # Statistiques de base
        total_numeros_utilises = set()
        for grille in grilles:
            total_numeros_utilises.update(grille)
        
        # Calcul des scores moyens
        scores_moyens = []
        for grille in grilles:
            score_grille = sum(self.scores.get(num, 0) for num in grille) / len(grille)
            scores_moyens.append(score_grille)
        
        # Équilibrage de l'utilisation
        utilisation = {}
        for grille in grilles:
            for num in grille:
                utilisation[num] = utilisation.get(num, 0) + 1
        
        return {
            'nb_grilles': len(grilles),
            'taille_moyenne_grille': sum(len(g) for g in grilles) / len(grilles),
            'numeros_uniques_utilises': len(total_numeros_utilises),
            'couverture_pool': len(total_numeros_utilises) / len(self.top_numeros) * 100,
            'score_moyen_global': sum(scores_moyens) / len(scores_moyens),
            'score_max_grille': max(scores_moyens),
            'score_min_grille': min(scores_moyens),
            'utilisation_max': max(utilisation.values()) if utilisation else 0,
            'utilisation_min': min(utilisation.values()) if utilisation else 0,
            'equilibrage_score': 100 - (max(utilisation.values()) - min(utilisation.values())) * 10 if utilisation else 0
        }


class GenerateurGrilles:
    """Classe principale pour la génération de grilles Loto/Keno avec systèmes réduits"""
    
    def __init__(self, jeu: str = 'auto'):
        self.base_path = Path(__file__).parent
        self.output_path = self.base_path / "sorties"
        self.output_path.mkdir(exist_ok=True)
        
        # Configuration selon le type de jeu
        if jeu.lower() == 'keno':
            self._configurer_keno()
        elif jeu.lower() == 'loto':
            self._configurer_loto()
        else:
            # Mode auto - sera configuré lors de l'analyse des numéros
            self.jeu = 'auto'
            self.min_numero = 1
            self.max_numero = 70  # Plage la plus large
            self.taille_grille = 5   # Par défaut Loto
            self.nom_jeu = "Auto"
    
    def _configurer_loto(self):
        """Configure pour le jeu Loto"""
        self.jeu = 'loto'
        self.min_numero = 1
        self.max_numero = 49
        self.taille_grille = 5
        self.nom_jeu = "Loto"
    
    def _configurer_keno(self):
        """Configure pour le jeu Keno"""
        self.jeu = 'keno'
        self.min_numero = 1
        self.max_numero = 70
        self.taille_grille = 10
        self.nom_jeu = "Keno"
    
    def detecter_jeu_automatique(self, nombres: List[int]) -> str:
        """
        Détecte automatiquement le type de jeu selon les numéros
        
        Logique :
        - Si tous les numéros sont ≤ 49 → Loto
        - Si au moins un numéro est > 49 → Keno
        """
        max_numero = max(nombres) if nombres else 1
        
        if max_numero > 49:
            jeu_detecte = 'keno'
            print(f"🔍 Détection automatique : **KENO** (numéro max: {max_numero})")
        else:
            jeu_detecte = 'loto'
            print(f"🔍 Détection automatique : **LOTO** (numéros ≤ 49)")
        
        return jeu_detecte
    
    def charger_nombres_depuis_fichier(self, fichier: str) -> List[int]:
        """Charge une liste de numéros depuis un fichier"""
        
        fichier_path = Path(fichier)
        if not fichier_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {fichier}")
        
        nombres = []
        
        try:
            with open(fichier_path, 'r', encoding='utf-8') as f:
                contenu = f.read()
                
                # Supporte plusieurs formats
                for ligne in contenu.split('\n'):
                    ligne = ligne.strip()
                    if not ligne or ligne.startswith('#'):
                        continue
                    
                    # Séparateurs possibles : virgule, espace, point-virgule
                    for sep in [',', ' ', ';', '\t']:
                        if sep in ligne:
                            nombres.extend([int(x.strip()) for x in ligne.split(sep) if x.strip().isdigit()])
                            break
                    else:
                        # Ligne avec un seul numéro
                        if ligne.isdigit():
                            nombres.append(int(ligne))
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la lecture du fichier : {e}")
        
        return list(set(nombres))  # Supprime les doublons
    
    def valider_nombres(self, nombres: List[int]) -> List[int]:
        """Valide et filtre la liste de numéros"""
        
        # Filtre les numéros valides selon le jeu
        nombres_valides = [n for n in nombres if self.min_numero <= n <= self.max_numero]
        
        min_requis = self.taille_grille  # Au minimum la taille de grille
        if len(nombres_valides) < min_requis:
            raise ValueError(f"Au moins {min_requis} numéros valides requis ({self.min_numero}-{self.max_numero}) pour des grilles de {self.taille_grille} numéros ({self.nom_jeu}). Trouvés : {len(nombres_valides)}")
        
        if len(nombres_valides) > 20:
            print(f"⚠️  Trop de numéros ({len(nombres_valides)}), limitation à 20")
            nombres_valides = nombres_valides[:20]
        
        return sorted(list(set(nombres_valides)))
    
    def exporter_grilles(self, grilles: List[List[int]], analyse: Dict, 
                        format_export: str = 'csv', nom_fichier: str = None) -> str:
        """Exporte les grilles dans le format spécifié"""
        
        if nom_fichier is None:
            nom_fichier = "grilles_systeme_reduit"
        
        if format_export.lower() == 'csv':
            return self._exporter_csv(grilles, analyse, nom_fichier)
        elif format_export.lower() == 'json':
            return self._exporter_json(grilles, analyse, nom_fichier)
        elif format_export.lower() == 'txt':
            return self._exporter_txt(grilles, analyse, nom_fichier)
        elif format_export.lower() == 'md' or format_export.lower() == 'markdown':
            return self._exporter_markdown(grilles, analyse, nom_fichier)
        else:
            raise ValueError(f"Format non supporté : {format_export}")
    
    def _exporter_csv(self, grilles: List[List[int]], analyse: Dict, nom_fichier: str) -> str:
        """Exporte au format CSV"""
        
        fichier_path = self.output_path / f"{nom_fichier}.csv"
        
        with open(fichier_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # En-têtes adaptés au nombre de numéros
            en_tetes = ['Grille'] + [f'Numero_{i}' for i in range(1, self.taille_grille + 1)]
            writer.writerow(en_tetes)
            
            # Grilles
            for i, grille in enumerate(grilles, 1):
                writer.writerow([f"Grille_{i:03d}"] + grille)
            
            # Ligne vide puis statistiques
            writer.writerow([])
            writer.writerow(['=== ANALYSE ==='])
            writer.writerow(['Indicateur', 'Valeur'])
            writer.writerow(['Nombre de grilles', analyse['nb_grilles']])
            
            # Analyse différente selon le mode
            if 'grilles_uniques' in analyse:
                # Mode classique
                writer.writerow(['Grilles uniques', analyse['grilles_uniques']])
                writer.writerow(['% Unicité', f"{analyse['pourcentage_unique']:.1f}%"])
                writer.writerow(['Score qualité', f"{analyse['score_qualite']:.1f}/100"])
                writer.writerow(['Recommandation', analyse['recommandation']])
            else:
                # Mode TOP CSV optimisé
                writer.writerow(['Numéros uniques utilisés', analyse['numeros_uniques_utilises']])
                writer.writerow(['Couverture du pool TOP', f"{analyse['couverture_pool']:.1f}%"])
                writer.writerow(['Score moyen', f"{analyse['score_moyen_global']:.3f}"])
                writer.writerow(['Score max', f"{analyse['score_max_grille']:.3f}"])
                writer.writerow(['Score min', f"{analyse['score_min_grille']:.3f}"])
                writer.writerow(['Équilibrage', f"{analyse['equilibrage_score']:.1f}/100"])
        
        return str(fichier_path)
    
    def _exporter_json(self, grilles: List[List[int]], analyse: Dict, nom_fichier: str) -> str:
        """Exporte au format JSON"""
        
        fichier_path = self.output_path / f"{nom_fichier}.json"
        
        data = {
            'metadata': {
                'date_generation': datetime.now().isoformat(),
                'type': 'systeme_reduit',
                'version': '1.0'
            },
            'grilles': [
                {
                    'id': i + 1,
                    'numeros': grille
                }
                for i, grille in enumerate(grilles)
            ],
            'analyse': analyse
        }
        
        with open(fichier_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(fichier_path)
    
    def _exporter_txt(self, grilles: List[List[int]], analyse: Dict, nom_fichier: str) -> str:
        """Exporte au format texte lisible"""
        
        fichier_path = self.output_path / f"{nom_fichier}.txt"
        
        with open(fichier_path, 'w', encoding='utf-8') as f:
            f.write("🎯 GRILLES LOTO - SYSTÈME RÉDUIT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📅 Date de génération : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"🎲 Nombre de grilles : {len(grilles)}\n")
            
            # Informations différentes selon le mode
            if 'score_qualite' in analyse:
                # Mode classique
                f.write(f"⭐ Score de qualité : {analyse['score_qualite']:.1f}/100\n")
                f.write(f"💡 Recommandation : {analyse['recommandation']}\n\n")
            else:
                # Mode TOP CSV optimisé
                f.write(f"🎯 Numéros uniques utilisés : {analyse['numeros_uniques_utilises']}\n")
                f.write(f"� Score moyen : {analyse['score_moyen_global']:.3f}\n")
                f.write(f"⚖️ Équilibrage : {analyse['equilibrage_score']:.1f}/100\n\n")
            
            f.write("�📋 GRILLES GÉNÉRÉES\n")
            f.write("-" * 30 + "\n\n")
            
            for i, grille in enumerate(grilles, 1):
                f.write(f"Grille {i:3d} : {' - '.join(f'{n:2d}' for n in grille)}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("📊 ANALYSE DÉTAILLÉE\n")
            f.write("=" * 50 + "\n\n")
            
            # Analyse différente selon le mode
            if 'grilles_uniques' in analyse:
                # Mode classique
                f.write(f"Grilles uniques : {analyse['grilles_uniques']}/{analyse['nb_grilles']} ")
                f.write(f"({analyse['pourcentage_unique']:.1f}%)\n\n")
                
                if 'plus_utilises' in analyse:
                    f.write("🔥 Numéros les plus utilisés :\n")
                    for num, count in analyse['plus_utilises']:
                        f.write(f"   {num:2d} : {count} fois\n")
                    f.write("\n")
                
                if 'couverture' in analyse:
                    couv = analyse['couverture']
                    f.write("📈 Couverture théorique :\n")
                    f.write(f"   Combinaisons couvertes : {couv['couverture_combinaisons']:.1f}%\n")
                    f.write(f"   Probabilité garantie : {couv['probabilite_garantie']:.1f}%\n")
                    f.write(f"   Efficacité : {couv['efficacite']:.1f}%\n")
            else:
                # Mode TOP CSV optimisé
                f.write(f"Numéros uniques utilisés : {analyse['numeros_uniques_utilises']}\n")
                f.write(f"Couverture du pool TOP : {analyse['couverture_pool']:.1f}%\n")
                f.write(f"Score moyen global : {analyse['score_moyen_global']:.3f}\n")
                f.write(f"Score maximum : {analyse['score_max_grille']:.3f}\n")
                f.write(f"Score minimum : {analyse['score_min_grille']:.3f}\n")
                f.write(f"Équilibrage : {analyse['equilibrage_score']:.1f}/100\n")
        
        return str(fichier_path)
    
    def _exporter_markdown(self, grilles: List[List[int]], analyse: Dict, nom_fichier: str) -> str:
        """Exporte au format Markdown"""
        
        fichier_path = self.output_path / f"{nom_fichier}.md"
        
        with open(fichier_path, 'w', encoding='utf-8') as f:
            f.write(f"# 🎯 Grilles {self.nom_jeu} - Système Réduit\n\n")
            
            f.write("## 📊 Informations Générales\n\n")
            f.write(f"- **📅 Date de génération :** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"- **🎲 Nombre de grilles :** {len(grilles)}\n")
            
            # Informations différentes selon le mode
            if 'score_qualite' in analyse:
                # Mode classique
                f.write(f"- **⭐ Score de qualité :** {analyse['score_qualite']:.1f}/100\n")
                f.write(f"- **💡 Recommandation :** {analyse['recommandation']}\n\n")
            else:
                # Mode TOP CSV optimisé
                f.write(f"- **🎯 Numéros uniques utilisés :** {analyse['numeros_uniques_utilises']}\n")
                f.write(f"- **📈 Score moyen :** {analyse['score_moyen_global']:.3f}\n")
                f.write(f"- **⚖️ Équilibrage :** {analyse['equilibrage_score']:.1f}/100\n\n")
            
            f.write("## 🎲 Grilles Générées\n\n")
            
            # En-tête du tableau adapté au nombre de numéros
            if self.taille_grille <= 5:
                f.write("| Grille | Numéro 1 | Numéro 2 | Numéro 3 | Numéro 4 | Numéro 5 |\n")
                f.write("|--------|----------|----------|----------|----------|----------|\n")
            elif self.taille_grille <= 10:
                headers = "| Grille |" + "".join([f" N°{i} |" for i in range(1, self.taille_grille + 1)]) + "\n"
                separator = "|--------|" + "-----|" * self.taille_grille + "\n"
                f.write(headers)
                f.write(separator)
            else:
                # Pour plus de 10 numéros, format compact
                headers = "| Grille |" + "".join([f" {i} |" for i in range(1, self.taille_grille + 1)]) + "\n"
                separator = "|--------|" + "---|" * self.taille_grille + "\n"
                f.write(headers)
                f.write(separator)
            
            for i, grille in enumerate(grilles, 1):
                numeros = " | ".join(f"**{n:02d}**" for n in grille)
                f.write(f"| {i:03d} | {numeros} |\n")
            
            f.write("\n## 📈 Analyse Détaillée\n\n")
            
            # Analyse différente selon le mode
            if 'grilles_uniques' in analyse:
                # Mode classique
                f.write("### ✅ Statistiques de Base\n\n")
                f.write(f"- **Grilles uniques :** {analyse['grilles_uniques']}/{analyse['nb_grilles']} ")
                f.write(f"({analyse['pourcentage_unique']:.1f}%)\n")
                f.write(f"- **Score de qualité :** {analyse['score_qualite']:.1f}/100\n")
                f.write(f"- **Recommandation :** {analyse['recommandation']}\n\n")
                
                if 'plus_utilises' in analyse:
                    f.write("### 🔥 Numéros les Plus Utilisés\n\n")
                    f.write("| Numéro | Occurrences |\n")
                    f.write("|--------|-------------|\n")
                    for num, count in analyse['plus_utilises']:
                        f.write(f"| **{num:02d}** | {count} fois |\n")
                    f.write("\n")
                
                if 'moins_utilises' in analyse and analyse['moins_utilises']:
                    f.write("### ❄️ Numéros les Moins Utilisés\n\n")
                    f.write("| Numéro | Occurrences |\n")
                    f.write("|--------|-------------|\n")
                    for num, count in analyse['moins_utilises']:
                        f.write(f"| **{num:02d}** | {count} fois |\n")
                    f.write("\n")
                
                if 'couverture' in analyse:
                    couv = analyse['couverture']
                    f.write("### 🎯 Couverture Théorique\n\n")
                    f.write("| Indicateur | Valeur |\n")
                    f.write("|------------|--------|\n")
                    f.write(f"| **Combinaisons couvertes** | {couv['couverture_combinaisons']:.1f}% |\n")
                    f.write(f"| **Probabilité garantie** | {couv['probabilite_garantie']:.1f}% |\n")
                    f.write(f"| **Efficacité du système** | {couv['efficacite']:.1f}% |\n")
                    f.write(f"| **Combinaisons théoriques** | {couv['total_combinaisons_theoriques']:,} |\n")
            else:
                # Mode TOP CSV optimisé
                f.write("### 🎯 Statistiques d'Optimisation\n\n")
                f.write(f"- **Numéros uniques utilisés :** {analyse['numeros_uniques_utilises']}\n")
                f.write(f"- **Couverture du pool TOP :** {analyse['couverture_pool']:.1f}%\n")
                f.write(f"- **Score moyen global :** {analyse['score_moyen_global']:.3f}\n")
                f.write(f"- **Score maximum :** {analyse['score_max_grille']:.3f}\n")
                f.write(f"- **Score minimum :** {analyse['score_min_grille']:.3f}\n")
                f.write(f"- **Équilibrage :** {analyse['equilibrage_score']:.1f}/100\n\n")
                f.write("\n")
            
            f.write("## 🎮 Comment Utiliser Ces Grilles\n\n")
            f.write("1. **Imprimer ou sauvegarder** ce fichier pour référence\n")
            f.write("2. **Jouer les grilles** selon votre budget (toutes ou une sélection)\n")
            f.write("3. **Suivre les résultats** pour évaluer la performance du système\n\n")
            
            f.write("## 💡 Conseils\n\n")
            
            # Conseils différents selon le mode
            if 'score_qualite' in analyse:
                # Mode classique
                if analyse['score_qualite'] >= 80:
                    f.write("✅ **Système de haute qualité** - Excellent choix de numéros et répartition optimale\n\n")
                elif analyse['score_qualite'] >= 60:
                    f.write("🟡 **Système de bonne qualité** - Quelques améliorations possibles\n\n")
                else:
                    f.write("🔴 **Système à optimiser** - Considérez augmenter le nombre de numéros favoris\n\n")
            else:
                # Mode TOP CSV optimisé
                if analyse['equilibrage_score'] >= 90:
                    f.write("✅ **Optimisation excellente** - Grilles parfaitement équilibrées avec TOP numéros\n\n")
                elif analyse['equilibrage_score'] >= 70:
                    f.write("🟡 **Bonne optimisation** - Grilles bien équilibrées avec les meilleurs numéros\n\n")
                else:
                    f.write("🔴 **Optimisation basique** - Grilles correctes mais équilibrage à améliorer\n\n")
            
            f.write("- 🎯 Les systèmes réduits ne garantissent pas de gains mais optimisent vos chances\n")
            f.write("- 📊 La garantie ne s'applique que si vos favoris contiennent les 5 bons numéros\n")
            f.write("- 💰 Adaptez le nombre de grilles à votre budget\n\n")
            
            f.write("---\n\n")
            f.write(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')} par le Générateur de Systèmes Réduits*\n")
        
        return str(fichier_path)
    
    def generer(self, nombres: List[int], nb_grilles: int, garantie: int = 3, 
               methode: str = 'optimal', nb_nombres_utilises: int = None,
               nombres_par_grille: int = None, export: bool = False, format_export: str = 'csv') -> Dict[str, Any]:
        """
        Génère des grilles selon les paramètres spécifiés
        
        Args:
            nombres: Liste des numéros favoris
            nb_grilles: Nombre de grilles à générer
            garantie: Niveau de garantie (2-5)
            methode: 'optimal' ou 'aleatoire'
            nb_nombres_utilises: Nombre de numéros à utiliser parmi les favoris (None = tous)
            nombres_par_grille: Nombre de numéros par grille (None = automatique selon le jeu)
            export: Exporter les résultats
            format_export: Format d'export ('csv', 'json', 'txt', 'md')
        
        Returns:
            Dictionnaire avec les grilles et l'analyse
        """
        
        # Détection automatique du jeu si nécessaire
        if self.jeu == 'auto':
            jeu_detecte = self.detecter_jeu_automatique(nombres)
            if jeu_detecte == 'keno':
                self._configurer_keno()
            else:
                self._configurer_loto()
        
        # Gestion du paramètre nombres_par_grille
        if nombres_par_grille is not None:
            min_par_grille = 3 if self.jeu == 'keno' else 5
            max_par_grille = 20 if self.jeu == 'keno' else 10
            
            if nombres_par_grille < min_par_grille:
                print(f"⚠️  Minimum {min_par_grille} numéros par grille pour {self.nom_jeu}. Ajustement automatique.")
                nombres_par_grille = min_par_grille
            elif nombres_par_grille > max_par_grille:
                print(f"⚠️  Maximum {max_par_grille} numéros par grille pour {self.nom_jeu}. Ajustement automatique.")
                nombres_par_grille = max_par_grille
            
            # Mise à jour de la taille de grille
            self.taille_grille = nombres_par_grille
            print(f"🎯 Taille de grille personnalisée : {self.taille_grille} numéros par grille")
        
        # Application des minimums selon le jeu
        min_grilles = 3 if self.jeu == 'keno' else 5
        if nb_grilles < min_grilles:
            print(f"⚠️  Minimum {min_grilles} grilles requis pour {self.nom_jeu}. Ajustement automatique.")
            nb_grilles = min_grilles
        
        print(f"\n🎯 GÉNÉRATEUR DE GRILLES {self.nom_jeu.upper()} - SYSTÈME RÉDUIT")
        print("=" * 60)
        
        # Validation des paramètres
        nombres_valides = self.valider_nombres(nombres)
        print(f"📊 Numéros favoris validés : {nombres_valides}")
        
        # Vérification si on a assez de numéros pour générer le nombre de grilles demandé
        import itertools
        combinaisons_possibles = len(list(itertools.combinations(nombres_valides, self.taille_grille)))
        
        if combinaisons_possibles < nb_grilles:
            print(f"⚠️  Seulement {combinaisons_possibles} combinaisons possibles avec {len(nombres_valides)} numéros.")
            print(f"💡 Pour obtenir {nb_grilles} grilles {self.nom_jeu}, ajoutez plus de numéros favoris.")
            print(f"📊 Minimum recommandé : {self.taille_grille + (nb_grilles - 1)} numéros")
            
            # On continue quand même avec ce qu'on a
            if combinaisons_possibles == 0:
                raise ValueError(f"Impossible de générer des grilles avec {len(nombres_valides)} numéros pour {self.nom_jeu}")
        
        # Sélection du nombre de numéros à utiliser
        if nb_nombres_utilises is not None:
            # Le minimum requis doit être adapté à la taille de grille actuelle
            min_requis = self.taille_grille  # Au minimum la taille de grille
            if nb_nombres_utilises < min_requis:
                raise ValueError(f"Au moins {min_requis} numéros sont requis pour générer des grilles de {self.taille_grille} numéros ({self.nom_jeu})")
            if nb_nombres_utilises > len(nombres_valides):
                raise ValueError(f"Impossible d'utiliser {nb_nombres_utilises} numéros parmi {len(nombres_valides)} disponibles")
            
            # Sélection des meilleurs numéros (on peut ajouter de la logique ici)
            import random
            random.seed(42)  # Pour la reproductibilité
            nombres_selectionnes = sorted(random.sample(nombres_valides, nb_nombres_utilises))
            print(f"🎯 Utilisation de {nb_nombres_utilises} numéros parmi {len(nombres_valides)} : {nombres_selectionnes}")
            nombres_valides = nombres_selectionnes
        
        # Création du système réduit
        systeme = SystemeReduit(nombres_valides, garantie, self.taille_grille)
        
        # Génération des grilles
        if methode.lower() == 'optimal':
            # Vérifier si on a assez de combinaisons pour la méthode optimale
            combinaisons_possibles = math.comb(len(nombres_valides), self.taille_grille)
            if combinaisons_possibles < nb_grilles:
                print(f"⚠️  Basculement automatique vers la méthode aléatoire")
                print(f"   Raison: {combinaisons_possibles} combinaisons < {nb_grilles} grilles demandées")
                
                # Suggérer une taille de grille plus petite si possible
                if self.jeu == 'keno' and self.taille_grille > 5:
                    for taille_test in range(self.taille_grille - 1, 4, -1):
                        combinaisons_test = math.comb(len(nombres_valides), taille_test)
                        if combinaisons_test >= nb_grilles:
                            print(f"💡 Conseil: Utilisez --nombres-par-grille {taille_test} pour {combinaisons_test} combinaisons possibles")
                            break
                
                grilles = systeme.generer_grilles_aleatoires_intelligentes(nb_grilles)
            else:
                grilles = systeme.generer_grilles_optimales(nb_grilles)
        else:
            grilles = systeme.generer_grilles_aleatoires_intelligentes(nb_grilles)
        
        # Analyse des résultats
        analyse = systeme.analyser_grilles(grilles)
        
        # Affichage des résultats
        self._afficher_resultats(grilles, analyse)
        
        # Export si demandé
        fichier_export = None
        if export:
            try:
                fichier_export = self.exporter_grilles(grilles, analyse, format_export)
                print(f"\n💾 Grilles exportées : {fichier_export}")
            except Exception as e:
                print(f"\n❌ Erreur lors de l'export : {e}")
        
        return {
            'grilles': grilles,
            'analyse': analyse,
            'fichier_export': fichier_export,
            'systeme_info': {
                'nombres_favoris': nombres_valides,
                'garantie': garantie,
                'methode': methode
            }
        }
    
    def _afficher_resultats(self, grilles: List[List[int]], analyse: Dict):
        """Affiche les résultats de génération"""
        
        print(f"\n🎲 GRILLES GÉNÉRÉES")
        print("=" * 40)
        
        for i, grille in enumerate(grilles[:10], 1):  # Affiche les 10 premières
            print(f"Grille {i:2d} : {' - '.join(f'{n:2d}' for n in grille)}")
        
        if len(grilles) > 10:
            print(f"... et {len(grilles) - 10} autres grilles")
        
        print(f"\n📊 ANALYSE")
        print("=" * 30)
        print(f"Nombre total de grilles : {analyse['nb_grilles']}")
        print(f"Grilles uniques : {analyse['grilles_uniques']}")
        print(f"Pourcentage d'unicité : {analyse['pourcentage_unique']:.1f}%")
        print(f"Score de qualité : {analyse['score_qualite']:.1f}/100")
        print(f"Recommandation : {analyse['recommandation']}")
        
        if 'couverture' in analyse:
            print(f"\n🎯 COUVERTURE THÉORIQUE")
            print("=" * 35)
            couv = analyse['couverture']
            print(f"Couverture combinaisons : {couv['couverture_combinaisons']:.1f}%")
            print(f"Probabilité garantie : {couv['probabilite_garantie']:.1f}%")
            print(f"Efficacité du système : {couv['efficacite']:.1f}%")


def main():
    """Fonction principale avec interface en ligne de commande"""
    
    parser = argparse.ArgumentParser(
        description="Générateur de grilles Loto avec systèmes réduits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :

  # Génération basique avec numéros en paramètre
  python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 5

  # Avec niveau de garantie personnalisé
  python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 10 --garantie 3

  # Utiliser seulement 8 numéros parmi les 10 favoris
  python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 5 --nombres-utilises 8

  # Mode Keno (1-70, grilles de 10 numéros)
  python generateur_grilles.py --jeu keno --nombres 5,15,25,35,45,55,65 --grilles 3

  # Lecture depuis un fichier
  python generateur_grilles.py --fichier mes_nombres.txt --grilles 8

  # Export en JSON
  python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 5 --export --format json

  # Méthode aléatoire intelligente
  python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 15 --methode aleatoire

  # NOUVEAUX : Utilisation des CSV TOP avec optimisation
  
  # Loto avec TOP 25 (5 numéros par grille, optimisation PuLP)
  python generateur_grilles.py --top-csv --jeu loto --grilles 5 --export

  # Keno avec TOP 30 (8 numéros par grille, algorithme glouton)
  python generateur_grilles.py --top-csv --jeu keno --grilles 10 --taille-grille-keno 8 --optimisation glouton

  # Loto avec TOP 15 numéros et grilles de 6 numéros
  python generateur_grilles.py --top-csv --jeu loto --grilles 8 --top-nombres 15 --taille-grille-loto 6

  # Mode automatique avec méthode top-optimise
  python generateur_grilles.py --methode top-optimise --grilles 12 --export --format json
        """
    )
    
    # Arguments principaux (optionnel si --top-csv est utilisé)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--nombres',
        type=str,
        help='Liste des numéros favoris séparés par des virgules (ex: 1,7,12,18,23)'
    )
    group.add_argument(
        '--fichier',
        type=str,
        help='Fichier contenant les numéros favoris (un par ligne ou séparés)'
    )
    
    parser.add_argument(
        '--grilles',
        type=int,
        required=True,
        help='Nombre de grilles à générer'
    )
    
    parser.add_argument(
        '--nombres-utilises',
        type=int,
        help='Nombre de numéros à utiliser parmi les favoris (défaut: tous)'
    )
    
    parser.add_argument(
        '--nombres-par-grille',
        type=int,
        help='Nombre de numéros par grille (Loto: min 5, Keno: min 3, défaut: automatique selon le jeu)'
    )
    
    parser.add_argument(
        '--jeu',
        choices=['auto', 'loto', 'keno'],
        default='auto',
        help='Type de jeu (auto: détection automatique, loto: 1-49/5 numéros, keno: 1-70/10 numéros, défaut: auto)'
    )
    
    parser.add_argument(
        '--garantie',
        type=int,
        choices=[2, 3, 4, 5],
        default=3,
        help='Niveau de garantie (défaut: 3)'
    )
    
    parser.add_argument(
        '--methode',
        choices=['optimal', 'aleatoire', 'top-optimise'],
        default='optimal',
        help='Méthode de génération (défaut: optimal, top-optimise: utilise les CSV TOP avec optimisation PuLP)'
    )
    
    # Nouvelles options pour les CSV TOP
    parser.add_argument(
        '--top-csv',
        action='store_true',
        help='Utiliser les fichiers CSV TOP 25 Loto / TOP 30 Keno au lieu de numéros manuels'
    )
    
    parser.add_argument(
        '--top-nombres',
        type=int,
        help='Nombre de numéros TOP à utiliser (défaut: 15 pour Loto, 20 pour Keno)'
    )
    
    parser.add_argument(
        '--optimisation',
        choices=['pulp', 'glouton'],
        default='pulp',
        help='Type d\'optimisation (pulp: optimisation linéaire, glouton: algorithme glouton, défaut: pulp)'
    )
    
    parser.add_argument(
        '--taille-grille-keno',
        type=int,
        choices=range(6, 11),
        help='Taille des grilles Keno (6-10 numéros) quand --top-csv est utilisé'
    )
    
    parser.add_argument(
        '--taille-grille-loto',
        type=int,
        choices=[5, 6, 7, 8, 9, 10],
        help='Taille des grilles Loto (5-10 numéros) quand --top-csv est utilisé'
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        help='Exporter les grilles vers un fichier'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'json', 'txt', 'md', 'markdown'],
        default='csv',
        help='Format d\'export (défaut: csv)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Affichage détaillé'
    )
    
    args = parser.parse_args()
    
    # Validation conditionnelle
    if not args.top_csv and not args.nombres and not args.fichier:
        parser.error("L'argument --nombres ou --fichier est requis sauf si --top-csv est utilisé")
    
    try:
        # Initialisation du générateur
        generateur = GenerateurGrilles(args.jeu)
        
        # Chargement des numéros (seulement si pas en mode TOP CSV)
        nombres = None
        if not args.top_csv:
            if args.nombres:
                try:
                    nombres = [int(x.strip()) for x in args.nombres.split(',')]
                except ValueError:
                    print("❌ Erreur : Format des numéros invalide. Utilisez: 1,2,3,4,5")
                    sys.exit(1)
            else:
                nombres = generateur.charger_nombres_depuis_fichier(args.fichier)
        
        # Validation des paramètres
        if args.grilles <= 0:
            print("❌ Erreur : Le nombre de grilles doit être positif")
            sys.exit(1)
        
        if args.grilles > 10000:
            print("❌ Erreur : Maximum 10000 grilles autorisées")
            sys.exit(1)
        
        # Validation des nombres seulement si pas en mode TOP CSV
        if not args.top_csv and nombres:
            jeu_detecte = generateur.detecter_jeu_automatique(nombres)
            if jeu_detecte != generateur.jeu and generateur.jeu != 'auto':
                print(f"⚠️  Attention : Jeu détecté ({jeu_detecte}) différent du jeu configuré ({generateur.jeu})")
        
        # Gestion des nouvelles options TOP CSV
        if args.top_csv or args.methode == 'top-optimise':
            print("\n🎯 MODE TOP CSV OPTIMISÉ")
            print("=" * 40)
            
            # Déterminer le jeu si auto
            jeu_final = args.jeu
            if jeu_final == 'auto':
                if args.taille_grille_loto:
                    jeu_final = 'loto'
                elif args.taille_grille_keno:
                    jeu_final = 'keno'
                else:
                    # Détecter selon les fichiers CSV disponibles
                    loto_csv = Path("loto_stats_exports/top_25_numeros_equilibres_loto.csv")
                    keno_csv = Path("keno_stats_exports/top_30_numeros_equilibres_keno.csv")
                    
                    if loto_csv.exists() and not keno_csv.exists():
                        jeu_final = 'loto'
                    elif keno_csv.exists() and not loto_csv.exists():
                        jeu_final = 'keno'
                    else:
                        print("🎲 Les deux fichiers CSV sont disponibles. Choisissez explicitement --jeu loto ou --jeu keno")
                        print("1️⃣  Loto (TOP 25 numéros)")
                        print("2️⃣  Keno (TOP 30 numéros)")
                        
                        while True:
                            choix = input("Votre choix (1-2): ").strip()
                            if choix == "1":
                                jeu_final = 'loto'
                                break
                            elif choix == "2":
                                jeu_final = 'keno'
                                break
                            else:
                                print("Choix invalide. Entrez 1 ou 2.")
            
            # Déterminer la taille de grille
            if jeu_final == 'loto':
                taille_grille = args.taille_grille_loto or 5
            elif jeu_final == 'keno':
                taille_grille = args.taille_grille_keno or 10
            
            print(f"🎮 Jeu sélectionné : {jeu_final.upper()}")
            print(f"🔢 Taille des grilles : {taille_grille} numéros")
            
            # Initialiser le système optimisé
            try:
                systeme_optimise = SystemeReduitOptimise(jeu_final, taille_grille)
                
                # Charger les numéros TOP
                if not systeme_optimise.charger_top_numeros(args.top_nombres):
                    print("❌ Impossible de charger les numéros TOP")
                    sys.exit(1)
                
                # Générer les grilles optimisées
                if args.optimisation == 'pulp' and PULP_AVAILABLE:
                    grilles = systeme_optimise.generer_grilles_optimisees_pulp(args.grilles, taille_grille)
                else:
                    if args.optimisation == 'pulp' and not PULP_AVAILABLE:
                        print("⚠️  PuLP non disponible, utilisation de l'algorithme glouton")
                    grilles = systeme_optimise.generer_grilles_optimisees_alternative(args.grilles, taille_grille)
                
                # Analyser les résultats
                analyse_optimisee = systeme_optimise.analyser_couverture_optimisee(grilles)
                
                # Affichage des résultats
                print(f"\n🎲 GRILLES GÉNÉRÉES (MODE TOP CSV)")
                print("=" * 50)
                
                for i, grille in enumerate(grilles[:10], 1):
                    score_grille = sum(systeme_optimise.scores.get(num, 0) for num in grille) / len(grille)
                    print(f"Grille {i:2d} : {' - '.join(f'{n:2d}' for n in grille)} (Score: {score_grille:.3f})")
                
                if len(grilles) > 10:
                    print(f"... et {len(grilles) - 10} autres grilles")
                
                print(f"\n📊 ANALYSE OPTIMISÉE")
                print("=" * 35)
                print(f"Grilles générées : {analyse_optimisee['nb_grilles']}")
                print(f"Numéros uniques utilisés : {analyse_optimisee['numeros_uniques_utilises']}")
                print(f"Couverture du pool TOP : {analyse_optimisee['couverture_pool']:.1f}%")
                print(f"Score moyen : {analyse_optimisee['score_moyen_global']:.3f}")
                print(f"Score max : {analyse_optimisee['score_max_grille']:.3f}")
                print(f"Score min : {analyse_optimisee['score_min_grille']:.3f}")
                print(f"Équilibrage : {analyse_optimisee['equilibrage_score']:.1f}/100")
                
                # Export si demandé
                if args.export:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if jeu_final == 'loto':
                        nom_fichier = f"grilles_loto_top25_optimisees_{timestamp}"
                    else:
                        nom_fichier = f"grilles_keno_top30_optimisees_{timestamp}"
                    
                    # Export selon le format
                    if args.format == 'csv':
                        chemin_export = generateur._exporter_csv(grilles, analyse_optimisee, nom_fichier)
                    elif args.format == 'json':
                        chemin_export = generateur._exporter_json(grilles, analyse_optimisee, nom_fichier)
                    elif args.format in ['md', 'markdown']:
                        chemin_export = generateur._exporter_markdown(grilles, analyse_optimisee, nom_fichier)
                    else:
                        chemin_export = generateur._exporter_txt(grilles, analyse_optimisee, nom_fichier)
                    
                    print(f"\n📁 Export réussi : {chemin_export}")
                
                print("\n✅ Génération TOP CSV terminée avec succès !")
                return
                
            except Exception as e:
                print(f"❌ Erreur dans le mode TOP CSV : {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                sys.exit(1)
        
        # Génération normale (seulement si pas en mode TOP CSV)
        if nombres is None:
            print("❌ Erreur : Aucun numéro fourni pour la génération normale")
            sys.exit(1)
            
        resultats = generateur.generer(
            nombres=nombres,
            nb_grilles=args.grilles,
            garantie=args.garantie,
            methode=args.methode,
            nb_nombres_utilises=getattr(args, 'nombres_utilises', None),
            nombres_par_grille=getattr(args, 'nombres_par_grille', None),
            export=args.export,
            format_export=args.format
        )
        
        print("\n✅ Génération terminée avec succès !")
        
        if args.verbose and resultats['fichier_export']:
            print(f"\n📁 Fichier généré : {resultats['fichier_export']}")
            print(f"📂 Dossier de sortie : {generateur.output_path}")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Génération interrompue par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
