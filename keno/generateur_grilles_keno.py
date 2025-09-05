import pandas as pd
import random
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary
from datetime import datetime
from pathlib import Path
from itertools import combinations

def charger_top30(path="keno_output/keno_top30_ml.csv"):
    df = pd.read_csv(path)
    # Prend la colonne 'Numero' ou 'numero'
    col = 'Numero' if 'Numero' in df.columns else 'numero'
    return list(df[col].head(30))

def systeme_reducteur_pulp(numeros, taille_grille, nb_grilles):
    """
    Génère des grilles optimisées pour couvrir au mieux les numéros du TOP 30
    """
    # Variables : grille_i contient le numéro_j (binaire)
    prob = LpProblem("KenoSystemeReducteur", LpMinimize)
    N = len(numeros)
    G = nb_grilles

    # x[i][j] = 1 si le numéro j est dans la grille i
    x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary) for j in range(N)] for i in range(G)]

    # Chaque grille contient exactement taille_grille numéros
    for i in range(G):
        prob += lpSum(x[i][j] for j in range(N)) == taille_grille

    # Chaque numéro doit apparaître au moins une fois (maximiser la couverture)
    for j in range(N):
        prob += lpSum(x[i][j] for i in range(G)) >= 1

    # Objectif : minimiser le nombre total de répétitions (équilibrage)
    prob += lpSum(x[i][j] for i in range(G) for j in range(N))

    prob.solve()

    grilles = []
    for i in range(G):
        grille = [numeros[j] for j in range(N) if x[i][j].varValue == 1]
        grilles.append(tuple(sorted(grille)))
    # Suppression des doublons
    grilles = list({grille for grille in grilles})
    # Si moins de grilles uniques que demandé, compléter avec des combinaisons aléatoires optimisées
    while len(grilles) < nb_grilles:
        grille = tuple(sorted(random.sample(numeros, taille_grille)))
        if grille not in grilles:
            grilles.append(grille)
    return [list(grille) for grille in grilles]

def systeme_reducteur_paires(numeros, taille_grille, grilles_univers):
    """
    Système réducteur optimal par couverture de paires (PLNE)
    grilles_univers : liste de toutes les grilles candidates
    """
    prob = LpProblem("KenoSystemeReducteurPaires", LpMinimize)
    N = len(grilles_univers)
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(N)]
    prob += lpSum(x)
    # Générer toutes les paires possibles
    paires = list(combinations(numeros, 2))
    for a, b in paires:
        indices = [i for i, grille in enumerate(grilles_univers) if a in grille and b in grille]
        if indices:
            prob += lpSum([x[i] for i in indices]) >= 1
    prob.solve()
    grilles_reduites = [grilles_univers[i] for i in range(N) if x[i].varValue == 1]
    return grilles_reduites

def generer_univers_grilles(numeros, taille_grille, max_univers=5000):
    """
    Génère un univers de grilles candidates (échantillon aléatoire)
    """
    univers = set()
    while len(univers) < max_univers:
        grille = tuple(sorted(random.sample(numeros, taille_grille)))
        univers.add(grille)
    return [list(g) for g in univers]

def export_grilles(grilles, taille, dossier="keno_output"):
    Path(dossier).mkdir(exist_ok=True)
    # Export CSV (remplace à chaque fois)
    csv_path = f"{dossier}/grilles_keno.csv"
    df = pd.DataFrame({"grille": [grille for grille in grilles]})
    df.to_csv(csv_path, index=False)
    # Export Markdown (remplace à chaque fois)
    md_path = f"{dossier}/grilles_keno.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 🎯 Grilles Keno générées ({len(grilles)} grilles de {taille} numéros)\n\n")
        for i, grille in enumerate(grilles, 1):
            f.write(f"- **Grille {i:02d}** : {' - '.join(str(n) for n in grille)}\n")
    print(f"✅ Export CSV : {csv_path}")
    print(f"✅ Export Markdown : {md_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Générateur de grilles Keno optimisées (système réducteur)")
    parser.add_argument("--grilles", type=int, default=10, help="Nombre de grilles à générer")
    parser.add_argument("--taille", type=int, default=8, choices=range(6,11), help="Nombre de numéros par grille (6 à 10)")
    parser.add_argument("--top30", type=str, default="keno_output/keno_top30_ml.csv", help="Fichier TOP 30 ML")
    parser.add_argument("--mode", type=str, choices=["classique", "paires"], default="classique", help="Mode d'optimisation")
    args = parser.parse_args()

    print(f"🔢 Chargement des numéros du TOP 30 ML depuis {args.top30} ...")
    numeros = charger_top30(args.top30)
    print(f"Numéros utilisés : {numeros}")

    if args.mode == "paires":
        print(f"🧮 Génération de l'univers de grilles candidates...")
        grilles_univers = generer_univers_grilles(numeros, args.taille, max_univers=5000)
        print(f"🧮 Optimisation par couverture de paires...")
        grilles = systeme_reducteur_paires(numeros, args.taille, grilles_univers)
    else:
        print(f"🧮 Génération de {args.grilles} grilles de {args.taille} numéros (optimisation pulp classique)...")
        grilles = systeme_reducteur_pulp(numeros, args.taille, args.grilles)

    for i, grille in enumerate(grilles, 1):
        print(f"Grille {i:02d} : {' - '.join(str(n) for n in grille)}")

    export_grilles(grilles, args.taille)

if __name__ == "__main__":
    main()