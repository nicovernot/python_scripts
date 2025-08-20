# 📊 Scripts d'Analyse Statistique Keno

Ce dossier contient plusieurs scripts pour analyser les statistiques du Keno et générer des recommandations basées sur les données historiques.

## 🎯 Scripts Disponibles

### 1. 📈 `analyse_stats_keno_complet.py` - Analyse Complète
**Le script le plus complet pour une analyse approfondie**

- ✅ Fréquences détaillées de tous les numéros (1-70)
- ✅ Analyse pair/impair avec tendances
- ✅ Analyse des zones (1-35 vs 36-70)
- ✅ Sommes des tirages et écarts théoriques
- ✅ Tableau complet des retards avec dates
- ✅ Score de priorité pour chaque numéro
- ✅ Régularité et statistiques avancées
- ✅ 6 graphiques de visualisation
- ✅ Rapport JSON et résumé textuel

**Utilisation :**
```bash
python keno/analyse_stats_keno_complet.py
```

**Sortie :**
- `keno_stats_exports/` : Tous les CSV détaillés
- `keno_analyse_plots/` : Graphiques PNG
- `rapport_complet.json` : Rapport complet
- `resume_analyse.txt` : Résumé textuel

---

### 2. ⚡ `analyse_keno_rapide.py` - Analyse Express
**Script rapide pour des recommandations immédiates**

- ✅ Top N numéros prioritaires (configurable)
- ✅ Analyse des 10 derniers tirages
- ✅ Tendances pair/impair et zones
- ✅ Recommandations express
- ✅ Graphique optionnel

**Utilisation :**
```bash
# Analyse rapide avec top 15
python keno/analyse_keno_rapide.py --top 15

# Avec graphiques
python keno/analyse_keno_rapide.py --top 10 --graphiques

# Fichier CSV spécifique
python keno/analyse_keno_rapide.py --csv chemin/vers/fichier.csv --top 20
```

**Options :**
- `--top N` : Nombre de numéros prioritaires (défaut: 15)
- `--graphiques` : Générer les graphiques
- `--csv PATH` : Chemin vers le fichier CSV

---

### 3. 🚀 `stats_keno_express.py` - Générateur Express
**Script tout-en-un sans paramètres pour usage quotidien**

- ✅ Analyse complète en une commande
- ✅ Top 15 prioritaires automatique
- ✅ Conseils stratégiques
- ✅ Répartition des retards
- ✅ Tendances récentes
- ✅ Sauvegarde automatique

**Utilisation :**
```bash
python stats_keno_express.py
```

**Parfait pour :** Usage quotidien, consultations rapides avant un tirage

---

## 📊 Données Générées

### CSV Principaux
| Fichier | Description |
|---------|-------------|
| `frequences_keno.csv` | Fréquences, retards, statuts de tous les numéros |
| `pair_impair_keno.csv` | Analyse de la répartition pair/impair par tirage |
| `zones_keno.csv` | Répartition zones 1-35 vs 36-70 par tirage |
| `sommes_keno.csv` | Sommes des tirages et écarts théoriques |
| `tableau_retards_complet.csv` | Tableau détaillé avec priorités |

### Graphiques
| Fichier | Description |
|---------|-------------|
| `frequences_keno.png` | Histogramme des fréquences |
| `heatmap_retards.png` | Carte de chaleur des retards |
| `pair_impair.png` | Distribution et évolution pair/impair |
| `zones.png` | Distribution et évolution des zones |
| `sommes.png` | Distribution et évolution des sommes |
| `top_retards.png` | Top 20 des numéros en retard |

---

## 🎯 Interprétation des Résultats

### Score de Priorité
- **> 100** : Numéro à privilégier absolument
- **50-100** : Numéro intéressant
- **< 50** : Numéro moins prioritaire

### Statuts
- **TRÈS EN RETARD** : Retard > 10 tirages
- **EN RETARD** : Retard > 7 tirages  
- **CHAUD** : Fréquence > moyenne + 20%
- **FROID** : Fréquence < moyenne - 20%
- **NORMAL** : Dans la moyenne

### Recommandations Équilibre
- **Pairs/Impairs** : Viser 8-12 pairs sur 20
- **Zones** : Viser 8-12 numéros en zone 1-35
- **Somme** : Viser entre 660-760 (moyenne théorique: 710)

---

## 🔧 Intégration avec le Menu CLI

Les scripts sont intégrés dans le menu principal :

- **Option 26** : 📊 Statistiques Keno complètes
- **Option 27** : ⚡ Analyse Keno rapide

```bash
python cli_menu.py
# Puis choisir 26 ou 27
```

---

## 📋 Exemples d'Usage

### Analyse Quotidienne Rapide
```bash
# 1. Analyse express pour des recommandations immédiates
python stats_keno_express.py

# 2. Consulter le résumé généré
cat keno_stats_exports/resume_express.txt
```

### Analyse Approfondie Hebdomadaire
```bash
# 1. Analyse complète avec visualisations
python keno/analyse_stats_keno_complet.py

# 2. Consulter les graphiques
ls keno_analyse_plots/

# 3. Lire le rapport détaillé
cat keno_stats_exports/resume_analyse.txt
```

### Analyse Personnalisée
```bash
# Top 20 avec graphiques
python keno/analyse_keno_rapide.py --top 20 --graphiques
```

---

## 📈 Conseils d'Utilisation

### Fréquence d'Analyse
- **Quotidienne** : `stats_keno_express.py`
- **Hebdomadaire** : `analyse_stats_keno_complet.py`
- **Avant chaque tirage** : `analyse_keno_rapide.py`

### Stratégie Recommandée
1. **Identifier** les numéros prioritaires (score > 50)
2. **Équilibrer** pair/impair et zones
3. **Vérifier** la somme totale
4. **Éviter** les numéros très chauds récemment
5. **Privilégier** les retards significatifs

### Interprétation des Tendances
- **↗️** Tendance haussière : continuer dans cette direction
- **↘️** Tendance baissière : possible retournement
- **➡️** Stable : maintenir l'équilibre

---

## 🚀 Performance

- **Express** : ~5 secondes
- **Rapide** : ~10 secondes  
- **Complet** : ~30 secondes (avec graphiques)

---

## 🔄 Mise à Jour des Données

Avant d'utiliser les scripts, assurez-vous d'avoir des données à jour :

```bash
# Via le menu CLI
python cli_menu.py
# Option 2 : Télécharger les données Keno

# Ou directement
python keno_cli.py extract
```

---

## 📞 Support

En cas de problème :
1. Vérifiez que `pandas`, `matplotlib`, `seaborn` sont installés
2. Assurez-vous que le fichier `keno/keno_data/keno_consolidated.csv` existe
3. Vérifiez les permissions d'écriture dans les dossiers de sortie

---

**🎰 Bonne chance avec vos analyses Keno ! 🍀**
