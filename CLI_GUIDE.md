# 🎯 Guide d'Utilisation du Menu CLI

## 🚀 Introduction

Le **Menu CLI Interactif** est l'interface principale du système Loto/Keno. Il offre une navigation intuitive avec codes couleurs et accès rapide à toutes les fonctionnalités.

## ⚡ Lancement

### Méthode Recommandée
```bash
./lancer_menu.sh
```
*Script automatique qui vérifie l'environnement et lance le menu*

### Méthode Alternative
```bash
python cli_menu.py
```
*Lancement direct (nécessite un environnement configuré)*

## 📋 Navigation

### Interface Principale

```
╔═══════════════════════════════════════════════════════════════╗
║                   🎲 SYSTÈME LOTO/KENO 🎰                    ║
║               Menu Interactif d'Analyse Avancée              ║
╚═══════════════════════════════════════════════════════════════╝

📅 Date: 13/08/2025 10:54:53

📊 Statut des Données:
  🎲 Loto:  ✓ Disponible (0.3MB, MAJ: 13/08/2025)
  🎰 Keno:  ✓ Disponible (0.4MB, MAJ: 13/08/2025)
```

### Codes Couleurs

- 🟦 **Bleu** : Téléchargement de données
- 🟩 **Vert** : Tests et maintenance
- 🟨 **Jaune** : Consultation des résultats
- 🟪 **Violet** : En-têtes et titres
- 🟥 **Rouge** : Quitter ou erreurs

## 📥 Section Téléchargement (1-3)

### 1️⃣ Télécharger données Loto
- Met à jour le fichier CSV Loto depuis FDJ
- Durée : ~30 secondes
- Résultat : `loto/loto_data/loto_201911.csv`

### 2️⃣ Télécharger données Keno
- Met à jour le fichier CSV Keno depuis FDJ
- Durée : ~45 secondes
- Résultat : `keno/keno_data/keno_202010.csv`

### 3️⃣ Mettre à jour toutes les données
- Équivalent à 1 + 2 en séquence
- Recommandé une fois par semaine

## 🎲 Section Loto (4-7)

### 4️⃣ 3 grilles Loto (rapide)
- **Durée** : ~15 secondes
- **Stratégie** : Équilibrée par défaut
- **Résultat** : `grilles.csv`

### 5️⃣ 5 grilles Loto (complet)
- **Durée** : ~20 secondes
- **Qualité** : Optimisée pour gains
- **Résultat** : `grilles.csv`

### 6️⃣ Grilles avec visualisations
- **Durée** : ~45 secondes
- **Génère** : Grilles + graphiques + statistiques
- **Dossiers** : `loto_analyse_plots/`, `loto_stats_exports/`

### 7️⃣ Analyse personnalisée
```
🎲 Configuration Loto Personnalisée

Nombre de grilles à générer (1-10) [3]: 5
Stratégies disponibles: equilibre, agressive, conservatrice, ml_focus
Stratégie [equilibre]: agressive
Générer les visualisations ? (o/N) [N]: o
Exporter les statistiques ? (o/N) [N]: o
```

## 🎰 Section Keno (8-10)

### 8️⃣ Analyse Keno (rapide)
- **Durée** : ~10 secondes
- **Résultat** : `keno_output/recommandations_keno.txt`

### 9️⃣ Analyse avec visualisations
- **Durée** : ~30 secondes
- **Génère** : Recommandations + graphiques + statistiques
- **Dossiers** : `keno_analyse_plots/`, `keno_stats_exports/`

### 1️⃣0️⃣ Analyse personnalisée
```
🎰 Configuration Keno Personnalisée

Générer les visualisations ? (o/N) [N]: o
Exporter les statistiques ? (o/N) [N]: o
Analyse approfondie (plus lent) ? (o/N) [N]: o
```

## 🧪 Section Tests (11-14)

### 1️⃣1️⃣ Tests complets
- **Durée** : ~2 minutes
- **Vérifie** : Toutes les fonctionnalités
- **Utilisation** : Diagnostic complet

### 1️⃣2️⃣ Tests essentiels
- **Durée** : ~30 secondes
- **Vérifie** : Fonctionnalités de base
- **Utilisation** : Vérification rapide

### 1️⃣3️⃣ Test de performance
- **Durée** : ~1 minute
- **Vérifie** : Vitesse et mémoire
- **Utilisation** : Optimisation

### 1️⃣4️⃣ Nettoyage
- **Statut** : En développement
- **Futur** : Suppression des fichiers temporaires

## 📊 Section Consultation (15-17)

### 1️⃣5️⃣ Voir grilles Loto
```
📄 Contenu de grilles.csv:
──────────────────────────────────────────────────
  1: Grille,B1,B2,B3,B4,B5,Score_ML,Somme,Equilibre
  2: 1,11,17,28,35,42,87.5,133,2/3
  3: 2,8,11,26,35,42,87.1,122,3/2
```

### 1️⃣6️⃣ Voir recommandations Keno
```
📄 Contenu de recommandations_keno.txt:
──────────────────────────────────────────────────
  1: RECOMMANDATIONS KENO - 13/08/2025
  2: ================================
  3: 
  4: 🎯 NUMÉROS RECOMMANDÉS:
  5: Stratégie principale: [7, 12, 15, 23, 28, 35, 41, 46]
```

### 1️⃣7️⃣ Ouvrir dossier graphiques
```
📊 Dossiers de Graphiques:
🎲 Loto: 4 graphiques dans loto_analyse_plots/
   - frequences_loto.png
   - heatmap_loto.png
   - retards_loto.png
   - sommes_loto.png
🎰 Keno: 4 graphiques dans keno_analyse_plots/
   - frequences_keno.png
   - heatmap_keno.png
   - paires_keno.png
   - retards_keno.png
```

## ⌨️ Raccourcis et Astuces

### Navigation Rapide
- **Entrée** : Accepter la valeur par défaut
- **Ctrl+C** : Quitter à tout moment
- **0** : Retour au menu principal

### Valeurs par Défaut
- Nombre de grilles Loto : **3**
- Stratégie : **equilibre**
- Visualisations : **Non**
- Export statistiques : **Non**

### Optimisation Temps
| Action | Durée | Recommandation |
|--------|-------|----------------|
| Test rapide | 30s | Option 12 |
| Grilles express | 15s | Option 4 |
| Analyse complète | 45s | Option 6 |
| Mise à jour données | 75s | Option 3 (hebdomadaire) |

## 🚨 Gestion des Erreurs

### Erreurs Courantes

#### "Fichier CSV manquant"
```
Solution: Lancer l'option 1 ou 2 pour télécharger les données
```

#### "Module not found"
```
Solution: 
1. Vérifier l'environnement virtuel : source venv/bin/activate
2. Installer les dépendances : pip install -r requirements.txt
```

#### "Permission denied"
```
Solution: chmod +x lancer_menu.sh
```

### Mode Debug
```bash
# Lancement avec logs détaillés
python cli_menu.py --debug

# Vérification de l'environnement
python test_cli.py
```

## 💡 Conseils d'Utilisation

### Workflow Recommandé

1. **Premier lancement** : Option 3 (mise à jour données)
2. **Usage quotidien** : Option 4 ou 8 (génération rapide)
3. **Analyse poussée** : Option 6 ou 9 (avec visualisations)
4. **Maintenance** : Option 12 (tests essentiels)

### Fréquence d'Utilisation

- **Données** : Mise à jour hebdomadaire (lundi)
- **Grilles** : Génération bi-hebdomadaire
- **Tests** : Vérification mensuelle

### Performance

- **Système lent** : Utiliser options 4 et 8
- **Système rapide** : Utiliser options 6 et 9
- **Analyse poussée** : Option 7 et 10 avec paramètres avancés

---

*Guide mis à jour le 13 août 2025*
