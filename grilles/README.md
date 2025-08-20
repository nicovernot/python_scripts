# 🎯 Générateur de Grilles Loto - Systèmes Réduits

## 📖 Qu'est-ce qu'un système réduit ?

Un **système réduit** est une méthode mathématique qui permet de jouer un grand nombre de numéros favoris avec un nombre minimal de grilles, tout en garantissant un certain niveau de gain.

### 🎲 Principe de base

Si vous avez **12 numéros favoris** et que vous voulez jouer toutes les combinaisons possibles, il faudrait jouer **792 grilles** différentes ! Un système réduit vous permet de ne jouer que **8-15 grilles** tout en conservant de bonnes chances de gain.

## 🔢 Niveaux de garantie

- **Garantie 2** : Si 5 de vos favoris sortent, vous êtes sûr d'avoir au moins un 2 dans une grille
- **Garantie 3** : Si 5 de vos favoris sortent, vous êtes sûr d'avoir au moins un 3 dans une grille
- **Garantie 4** : Si 5 de vos favoris sortent, vous êtes sûr d'avoir au moins un 4 dans une grille
- **Garantie 5** : Si 5 de vos favoris sortent, vous êtes sûr d'avoir au moins un 5 dans une grille

## 🚀 Utilisation du générateur

### Méthode 1 : Numéros en paramètres
```bash
python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 5
```

### Méthode 2 : Fichier de numéros
```bash
# Créer un fichier mes_nombres.txt avec vos numéros favoris
echo "1,7,12,18,23,29,34,39,45,49" > mes_nombres.txt

python generateur_grilles.py --fichier mes_nombres.txt --grilles 8
```

### Méthode 3 : Avec export
```bash
python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 10 --export --format csv
```

## ⚙️ Options avancées

### Sélection de numéros
- `--nombres-utilises N` : Utilise seulement N numéros parmi les favoris (minimum 7)
  ```bash
  # Utilise 8 numéros parmi 10 favoris
  python generateur_grilles.py --nombres 1,7,12,18,23,29,34,39,45,49 --grilles 5 --nombres-utilises 8
  ```

### Choix de la méthode
- `--methode optimal` : Algorithme de couverture maximale (recommandé)
- `--methode aleatoire` : Génération aléatoire intelligente

### Niveau de garantie
- `--garantie 2` : Garantie minimale (plus de grilles nécessaires)
- `--garantie 3` : Équilibre optimal (défaut)
- `--garantie 4` : Garantie élevée (moins de grilles nécessaires)
- `--garantie 5` : Garantie maximale (très peu de grilles)

### Formats d'export
- `--format csv` : Tableur Excel/LibreOffice
- `--format json` : Données structurées pour applications
- `--format txt` : Fichier texte lisible

## 📊 Exemple de résultat

```
🎯 GÉNÉRATEUR DE GRILLES LOTO - SYSTÈME RÉDUIT
============================================================
📊 Numéros favoris validés : [1, 7, 12, 18, 23, 29, 34, 39, 45, 49]
🎯 Système réduit initialisé :
   Numéros favoris : [1, 7, 12, 18, 23, 29, 34, 39, 45, 49]
   Nombre de favoris : 10
   Garantie : 3

🎲 GRILLES GÉNÉRÉES
========================================
Grille  1 :  1 - 18 - 23 - 34 - 45
Grille  2 :  7 - 12 - 29 - 39 - 49
Grille  3 :  1 -  7 - 12 - 18 - 23
Grille  4 :  1 -  7 - 12 - 29 - 34
Grille  5 :  1 -  7 - 12 - 39 - 45

📊 ANALYSE
==============================
Nombre total de grilles : 5
Grilles uniques : 5
Pourcentage d'unicité : 100.0%
Score de qualité : 100.0/100
Recommandation : 🟢 Excellent système - Qualité optimale

🎯 COUVERTURE THÉORIQUE
===================================
Couverture combinaisons : 2.4%
Probabilité garantie : 30.0%
Efficacité du système : 600.0%
```

## 💡 Conseils d'utilisation

### Choix des numéros favoris
- **8-12 numéros** : Optimal pour 5-10 grilles
- **13-15 numéros** : Bon pour 10-20 grilles  
- **16-20 numéros** : Pour systèmes plus étendus

### Nombre de grilles recommandé
- **Budget serré** : 5-8 grilles avec garantie 3
- **Budget moyen** : 10-15 grilles avec garantie 3-4
- **Gros budget** : 20+ grilles avec garantie 4-5

### Stratégies avancées
1. **Mixer les méthodes** : Générez plusieurs systèmes avec différentes méthodes
2. **Analyser les patterns** : Regardez la répartition des numéros dans les grilles
3. **Combiner avec l'analyse statistique** : Utilisez les données du système principal

## 📈 Avantages du système réduit

✅ **Économique** : Moins de grilles à jouer
✅ **Mathématique** : Couverture optimisée
✅ **Flexible** : Différents niveaux de garantie
✅ **Traçable** : Analyse détaillée de chaque système

## ⚠️ Limitations

- Les systèmes réduits ne garantissent pas de gains
- Plus de numéros favoris = plus de grilles nécessaires
- La garantie ne s'applique que si vos favoris contiennent les 5 bons numéros

## 🎯 Exemples pratiques

### Système débutant (8 favoris, 5 grilles)
```bash
python generateur_grilles.py --nombres 7,14,21,28,35,42,6,13 --grilles 5 --garantie 3
```

### Système intermédiaire (12 favoris, 10 grilles)
```bash
python generateur_grilles.py --nombres 1,5,7,12,15,18,23,29,33,39,45,49 --grilles 10 --garantie 3
```

### Système avancé (15 favoris, 20 grilles)
```bash
python generateur_grilles.py --nombres 1,3,7,9,12,15,18,21,25,29,33,37,41,45,49 --grilles 20 --garantie 4
```

## 📂 Fichiers générés

Les grilles sont exportées dans le dossier `grilles/sorties/` avec :
- **Nom horodaté** : `grilles_systeme_reduit_YYYYMMDD_HHMMSS.ext`
- **Analyse complète** : Statistiques et recommandations
- **Formats multiples** : CSV, JSON, TXT

## 🔗 Intégration avec le système principal

Le générateur de grilles peut être utilisé en complément du système d'analyse principal pour :
- Créer des systèmes personnalisés basés sur vos analyses
- Générer des grilles avec vos numéros porte-bonheur
- Optimiser vos mises avec les systèmes réduits

---

*Pour plus d'informations sur la théorie des systèmes réduits, consultez la littérature mathématique sur les designs combinatoires et les systèmes de couverture.*
