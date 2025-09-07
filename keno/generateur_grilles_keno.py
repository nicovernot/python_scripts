"""
Amélioration du générateur de grilles Keno — version complète
- Calculs étendus de scores de paires (lift, P(j|i), MI, gap, recence)
- Pool de candidats pondéré par paires fortes
- Sélection greedy pondérée (set-cover) + recherche locale (1-swap) pour améliorer
- Option ILP (PuLP) pour couverture pondérée des paires
- Contraintes pratiques : dizaines max, parité, consécutifs max, overlap interdiction
- Backtest simple sur l'historique
- Export CSV/Markdown

Usage: python keno_improved_generator.py --help
"""

import pandas as pd
import random
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary, PULP_CBC_CMD
from datetime import datetime
from pathlib import Path
from itertools import combinations, combinations_with_replacement
import numpy as np
from collections import Counter, defaultdict
import argparse
import math

# ----------------------------- Utilities -----------------------------

def charger_top30(path="keno_output/keno_top30_ml.csv"):
    df = pd.read_csv(path)
    col = 'Numero' if 'Numero' in df.columns else ('numero' if 'numero' in df.columns else df.columns[0])
    return list(df[col].head(30).astype(int))

# ----------------------------- Scoring pairs -----------------------------

def calculer_stats_historique(filepath, numeros, boule_prefixes=('b',)):
    """Parcours de l'historique pour calculer freq, cofreq, derniers tirages, etc."""
    df = pd.read_csv(filepath)
    boule_cols = [col for col in df.columns if any(col.startswith(p) for p in boule_prefixes)]
    freq = Counter()
    cofreq = Counter()
    last_seen = defaultdict(lambda: -1)
    draw_index = 0
    for _, row in df.iterrows():
        draw_index += 1
        tirage = [int(row[col]) for col in boule_cols]
        for i in tirage:
            if i in numeros:
                freq[i] += 1
                last_seen[i] = draw_index
        for i, j in combinations(tirage, 2):
            if i in numeros and j in numeros:
                a, b = tuple(sorted((int(i), int(j))))
                cofreq[(a, b)] += 1
    total_draws = draw_index
    return {
        'df': df,
        'boule_cols': boule_cols,
        'freq': freq,
        'cofreq': cofreq,
        'last_seen': last_seen,
        'total_draws': total_draws
    }


def compute_pair_metrics(stats, numeros, eps=1e-9):
    freq = stats['freq']
    cofreq = stats['cofreq']
    total = stats['total_draws']
    pair_metrics = {}
    for i, j in combinations(sorted(numeros), 2):
        a, b = i, j
        f_i = freq[a]
        f_j = freq[b]
        c = cofreq.get((a, b), 0)
        p_i = f_i / total if total else 0
        p_j = f_j / total if total else 0
        p_ij = c / total if total else 0
        lift = p_ij / (p_i * p_j + eps) if p_i and p_j else 0
        p_cond = c / f_i if f_i else 0
        mi = math.log2(p_ij / (p_i * p_j + eps) + eps) if p_ij > 0 else 0
        # recence / gap
        last_i = stats['last_seen'].get(a, -1)
        last_j = stats['last_seen'].get(b, -1)
        recence_score = 0
        if last_i > 0 and last_j > 0:
            # more recent => higher score
            recence_score = 1 / (1 + (total - max(last_i, last_j)))
        # simple structural penalties
        overlap_penalty = 0
        if a // 10 == b // 10: overlap_penalty += 0.2
        if a % 10 == b % 10: overlap_penalty += 0.1
        if a % 2 == b % 2: overlap_penalty += 0.05
        pair_metrics[(a, b)] = {
            'c': c,
            'lift': lift,
            'p_cond': p_cond,
            'mi': mi,
            'p_ij': p_ij,
            'recence': recence_score,
            'overlap_penalty': overlap_penalty
        }
    return pair_metrics


def default_pair_score(metrics, weights=None):
    if weights is None:
        weights = {'lift':1.0, 'p_cond':0.5, 'mi':0.8, 'recence':0.5, 'penalty':0.4}
    s = (
        weights['lift'] * metrics['lift'] +
        weights['p_cond'] * metrics['p_cond'] +
        weights['mi'] * metrics['mi'] +
        weights['recence'] * metrics['recence'] -
        weights['penalty'] * metrics['overlap_penalty']
    )
    return s

# ----------------------------- Candidate pool -----------------------------

def generer_univers_pondere(numeros, taille_grille, pair_metrics, pool_size=2000, prefer_top_pairs=100):
    """Génère un pool de candidats en privilégiant la présence de paires à fort score.
    Méthode: on sélectionne d'abord quelques paires fortes, puis on complète la grille en tirant numéros
    favorisant paires avec score positif.
    """
    # precompute pair scores
    pair_scores = {p: default_pair_score(m) for p, m in pair_metrics.items()}
    top_pairs = [p for p, _ in sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)][:prefer_top_pairs]
    univers = set()
    tries = 0
    while len(univers) < pool_size and tries < pool_size * 10:
        tries += 1
        grille = set()
        # choose 1-2 top pairs to seed
        n_seeds = random.choice([1,1,2])
        seeds = random.sample(top_pairs, min(n_seeds, len(top_pairs))) if top_pairs else []
        for a, b in seeds:
            grille.add(a); grille.add(b)
            if len(grille) >= taille_grille: break
        # fill rest by weighted sampling of numbers that form good pairs with existing
        while len(grille) < taille_grille:
            candidates = [n for n in numeros if n not in grille]
            if not grille:
                choice = random.choice(candidates)
                grille.add(choice)
                continue
            # score each candidate by sum of pair_scores it would create with current grille
            cand_scores = []
            for c in candidates:
                s = 0
                for g in grille:
                    a, b = tuple(sorted((c, g)))
                    s += pair_scores.get((a, b), 0)
                cand_scores.append(max(s, 0) + 1e-6)
            # probabilistic selection biased by score
            total = sum(cand_scores)
            probs = [cs/total for cs in cand_scores]
            choice = random.choices(candidates, weights=probs, k=1)[0]
            grille.add(choice)
        univers.add(tuple(sorted(grille)))
    return [list(g) for g in univers]

# ----------------------------- Greedy cover (pondéré) -----------------------------

def weighted_greedy_cover(numeros, pool, pair_metrics, top_pairs_list, nb_to_select, constraints=None):
    """Sélectionne nb_to_select grilles depuis pool pour couvrir au mieux top_pairs_list (pondérées par score)
    constraints: dict with keys 'max_same_decade', 'min_even', 'max_consecutive', 'max_overlap'
    """
    pair_scores = {p: default_pair_score(pair_metrics[p]) for p in pair_metrics}
    target_pairs = set(top_pairs_list)
    covered = set()
    selected = []

    # precompute pairs in each candidate
    pool_pairs = [set(tuple(sorted((a,b))) for a,b in combinations(grille,2)) for grille in pool]

    for _ in range(nb_to_select):
        best_idx = None
        best_gain = -1
        best_score = None
        for idx, grille in enumerate(pool):
            if idx in selected: continue
            pairs = pool_pairs[idx]
            newly = pairs - covered
            # compute gain = sum of scores of newly covered pairs, penalize overlap with already selected
            gain = sum(pair_scores.get(p,0) for p in newly)
            # apply simple constraints penalty
            penalty = 0
            if constraints:
                # decades
                if 'max_same_decade' in constraints:
                    decades = [n//10 for n in grille]
                    if max([decades.count(d) for d in set(decades)]) > constraints['max_same_decade']:
                        penalty += 1e3
                # parity
                if 'min_even' in constraints or 'max_even' in constraints:
                    evens = sum(1 for n in grille if n%2==0)
                    if 'min_even' in constraints and evens < constraints['min_even']: penalty += 500
                    if 'max_even' in constraints and evens > constraints['max_even']: penalty += 500
                # consecutive
                if 'max_consecutive' in constraints:
                    consec = max_consecutive_run(sorted(grille))
                    if consec > constraints['max_consecutive']: penalty += 200
                # overlap with existing selected
                if 'max_overlap' in constraints and selected:
                    for sidx in selected:
                        overlap = len(set(pool[sidx]) & set(grille))
                        if overlap > constraints['max_overlap']:
                            penalty += 300
            net = gain - penalty
            if net > best_gain:
                best_gain = net
                best_idx = idx
                best_score = (gain, penalty)
        if best_idx is None:
            break
        selected.append(best_idx)
        covered |= pool_pairs[best_idx]
        if len(selected) >= nb_to_select:
            break
    return [pool[i] for i in selected]


def max_consecutive_run(sorted_list):
    max_run = 1
    cur = 1
    for a, b in zip(sorted_list, sorted_list[1:]):
        if b == a + 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max_run

# ----------------------------- Local search (1-swap) -----------------------------

def local_search_swap(selected, pool, pair_metrics, target_pairs, time_budget=1.0):
    """Try 1-swap improvements on selected (list of grilles), swapping one element with a candidate not in the grid.
    target_pairs: set of pairs we want to maximize coverage/score over.
    """
    pair_scores = {p: default_pair_score(pair_metrics[p]) for p in pair_metrics}
    selected = [list(g) for g in selected]
    current_pairs = set()
    for g in selected:
        for a,b in combinations(g,2): current_pairs.add(tuple(sorted((a,b))))
    current_score = sum(pair_scores.get(p,0) for p in (current_pairs & target_pairs))

    improved = True
    iter_count = 0
    while improved and iter_count < 1000:
        improved = False
        iter_count += 1
        for gi, g in enumerate(selected):
            for idx_in, old in enumerate(g):
                for cand in range(1, 81):
                    if cand in g: continue
                    newg = g[:idx_in] + [cand] + g[idx_in+1:]
                    # compute score quickly
                    new_pairs = set(tuple(sorted((a,b))) for a,b in combinations(newg,2))
                    # rebuild total pairs if swapped
                    total_pairs = set()
                    for k,gg in enumerate(selected):
                        if k == gi:
                            total_pairs |= new_pairs
                        else:
                            total_pairs |= set(tuple(sorted((a,b))) for a,b in combinations(gg,2))
                    new_score = sum(pair_scores.get(p,0) for p in (total_pairs & target_pairs))
                    if new_score > current_score + 1e-6:
                        selected[gi] = sorted(newg)
                        current_score = new_score
                        improved = True
                        break
                if improved: break
            if improved: break
    return selected

# ----------------------------- ILP option -----------------------------

def ilp_cover_pairs(pool, pair_metrics, top_pairs_list, nb_to_select):
    N = len(pool)
    pair_scores = {p: default_pair_score(pair_metrics[p]) for p in pair_metrics}
    target = set(top_pairs_list)
    prob = LpProblem('ILP_Cover', LpMinimize)
    x = [LpVariable(f'x_{i}', cat=LpBinary) for i in range(N)]
    # We minimize negative covered score (i.e. maximize covered score)
    pool_pairs = [set(tuple(sorted((a,b))) for a,b in combinations(grille,2)) for grille in pool]
    # auxiliary binary y_p if pair p is covered
    y = {p: LpVariable(f"y_{p[0]}_{p[1]}", cat=LpBinary) for p in target}
    # constraints: y_p <= sum x_i over grids that include p
    for p in target:
        indices = [i for i, pairs in enumerate(pool_pairs) if p in pairs]
        if indices:
            prob += lpSum(x[i] for i in indices) - y[p] >= 0
        else:
            prob += y[p] == 0
    prob += lpSum(x) == nb_to_select
    # objective: minimize negative sum(score_p * y_p) -> i.e. maximize sum(score_p*y_p)
    prob += -lpSum(pair_scores.get(p,0) * y[p] for p in target)
    prob.solve(PULP_CBC_CMD(msg=0))
    sel = [pool[i] for i in range(N) if x[i].varValue == 1]
    return sel

# ----------------------------- Backtest -----------------------------

def backtest_grilles(grilles, historique_path, boule_prefixes=('b',)):
    df = pd.read_csv(historique_path)
    boule_cols = [col for col in df.columns if any(col.startswith(p) for p in boule_prefixes)]
    results = []
    for _, row in df.iterrows():
        tirage = set(int(row[c]) for c in boule_cols)
        for g in grilles:
            hits = len(set(g) & tirage)
            results.append(hits)
    # return basic stats
    arr = np.array(results)
    return {'mean_hits_per_grid_per_draw': arr.mean(), 'pct_at_least_3': np.mean(arr>=3)}

# ----------------------------- Export -----------------------------

def export_grilles(grilles, taille, dossier="keno_output"):
    Path(dossier).mkdir(exist_ok=True)
    csv_path = f"{dossier}/grilles_keno.csv"
    df = pd.DataFrame({"grille": [grille for grille in grilles]})
    df.to_csv(csv_path, index=False)
    md_path = f"{dossier}/grilles_keno.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 🎯 Grilles Keno générées ({len(grilles)} grilles de {taille} numéros)\n\n")
        for i, grille in enumerate(grilles, 1):
            f.write(f"- **Grille {i:02d}** : {' - '.join(str(n) for n in grille)}\n")
    print(f"✅ Export CSV : {csv_path}")
    print(f"✅ Export Markdown : {md_path}")

# ----------------------------- Pipeline / main -----------------------------

def pipeline(args):
    numeros = charger_top30(args.top30)
    print(f"Using numbers: {numeros}")
    stats = calculer_stats_historique(args.historique, numeros)
    pair_metrics = compute_pair_metrics(stats, numeros)
    # choose top pairs by default score
    pair_scores = {p: default_pair_score(m) for p, m in pair_metrics.items()}
    top_pairs_sorted = [p for p,_ in sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)]
    top_pairs = top_pairs_sorted[:args.top_paires]
    print(f"Top {len(top_pairs)} pairs selected")

    print("Generating candidate pool (weighted)...")
    pool = generer_univers_pondere(numeros, args.taille, pair_metrics, pool_size=args.max_univers, prefer_top_pairs=max(10,args.top_paires//2))
    print(f"Candidate pool size: {len(pool)}")

    constraints = {
        'max_same_decade': args.max_same_decade,
        'min_even': args.min_even,
        'max_even': args.max_even,
        'max_consecutive': args.max_consecutive,
        'max_overlap': args.max_overlap_between_grids
    }

    if args.mode == 'greedy':
        print('Selecting with weighted greedy')
        selected = weighted_greedy_cover(numeros, pool, pair_metrics, top_pairs, args.grilles, constraints=constraints)
    elif args.mode == 'ilp':
        print('Selecting with ILP')
        selected = ilp_cover_pairs(pool, pair_metrics, top_pairs, args.grilles)
    else:
        print('Fallback greedy')
        selected = weighted_greedy_cover(numeros, pool, pair_metrics, top_pairs, args.grilles, constraints=constraints)

    print('Applying local search (swap) to improve')
    improved = local_search_swap(selected, pool, pair_metrics, set(top_pairs))

    print('Final grids:')
    for i,g in enumerate(improved,1):
        print(f"Grille {i:02d}: {' - '.join(str(x) for x in sorted(g))}")

    export_grilles(improved, args.taille)
    if args.backtest:
        print('Backtesting...')
        stats_bt = backtest_grilles(improved, args.historique)
        print(stats_bt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grilles', type=int, default=10)
    parser.add_argument('--taille', type=int, default=8, choices=range(6,11))
    parser.add_argument('--top30', type=str, default='keno_output/keno_top30_ml.csv')
    parser.add_argument('--mode', type=str, choices=['greedy','ilp'], default='greedy')
    parser.add_argument('--max-univers', type=int, dest='max_univers', default=1000)
    parser.add_argument('--top-paires', type=int, default=30)
    parser.add_argument('--historique', type=str, default='keno/keno_data/keno_consolidated.csv')
    parser.add_argument('--backtest', action='store_true')
    # constraints
    parser.add_argument('--max-same-decade', type=int, dest='max_same_decade', default=3)
    parser.add_argument('--min-even', type=int, dest='min_even', default=2)
    parser.add_argument('--max-even', type=int, dest='max_even', default=6)
    parser.add_argument('--max-consecutive', type=int, dest='max_consecutive', default=3)
    parser.add_argument('--max-overlap-between-grids', type=int, dest='max_overlap_between_grids', default=6)

    args = parser.parse_args()
    pipeline(args)
