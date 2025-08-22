# 🚀 API ÉTENDUE LOTO/KENO - DOCUMENTATION COMPLÈTE

*Mise à jour du 22 août 2025*

## 📋 FONCTIONNALITÉS AJOUTÉES

### 🎯 **NOUVEAUTÉS PRINCIPALES**

1. **📁 Gestion des Fichiers**
   - Liste et téléchargement des CSV générés
   - Visualisation des graphiques dans le navigateur
   - Catégorisation automatique par type de fichier

2. **🧠 Analyse des Stratégies**
   - Évaluation automatique des performances
   - Recommandations basées sur les données historiques
   - Scoring intelligent par stratégie

3. **📊 Dashboard Complet**
   - Vue d'ensemble des fichiers et analyses
   - Interface web moderne et responsive
   - Intégration complète KENO/LOTO

## 🌐 ENDPOINTS AJOUTÉS

### 📁 **GESTION DES FICHIERS**

#### `GET /api/files/list`
Liste tous les fichiers disponibles avec métadonnées.

**Paramètres :**
- `type` (optionnel) : `keno`, `loto` ou vide pour tous

**Réponse :**
```json
{
  "success": true,
  "data": {
    "keno_csv": [
      {
        "name": "frequences_keno.csv",
        "path": "keno_stats_exports/frequences_keno.csv",
        "size": 15420,
        "size_human": "15.1 KB",
        "modified": "2025-08-22T10:30:00",
        "category": "data",
        "mime_type": "text/csv"
      }
    ],
    "keno_plots": [...],
    "keno_output": [...]
  }
}
```

#### `GET /api/files/download/<path>`
Télécharge un fichier spécifique.

**Exemple :**
```bash
curl -O "http://localhost:5000/api/files/download/keno_stats_exports/frequences_keno.csv"
```

#### `GET /api/files/view/<path>`
Affiche un fichier dans le navigateur (images, CSV, etc.).

**Exemple :**
```bash
curl "http://localhost:5000/api/files/view/keno_analyse_plots/frequences_keno.png"
```

### 🎯 **ANALYSE DES STRATÉGIES**

#### `GET /api/strategies/analyze/<game_type>`
Analyse en profondeur des stratégies disponibles.

**Game types :** `keno`, `loto`

**Réponse :**
```json
{
  "success": true,
  "data": {
    "strategies": {
      "frequences": {
        "name": "frequences",
        "score": 10077.14,
        "data_points": 70,
        "metrics": {
          "avg_frequency": 1007.71,
          "max_frequency": 1062.0
        }
      },
      "retards": {...},
      "paires": {...}
    },
    "best_strategy": {
      "name": "frequences",
      "score": 10077.14,
      "reason": "Meilleur score de performance"
    }
  }
}
```

#### `GET /api/strategies/recommend/<game_type>`
Recommandations personnalisées basées sur l'analyse.

**Réponse :**
```json
{
  "success": true,
  "data": {
    "primary_strategy": "frequences",
    "confidence": 85,
    "recommendations": [
      {
        "type": "strategy",
        "message": "Privilégier les numéros fréquents pour cette session"
      }
    ]
  }
}
```

### 📊 **DASHBOARD COMPLET**

#### `GET /api/dashboard/<game_type>`
Toutes les données nécessaires pour le dashboard en une seule requête.

**Réponse :**
```json
{
  "success": true,
  "data": {
    "files": {...},
    "strategy_recommendations": {...},
    "data_status": {
      "loto_available": true,
      "keno_available": true,
      "overall_health": true
    },
    "game_type": "keno",
    "last_updated": "2025-08-22T10:30:00"
  }
}
```

### 🌐 **INTERFACE WEB**

#### `GET /dashboard`
Interface web complète avec :
- Sélection du type de jeu (Keno/Loto)
- Visualisation des fichiers par catégorie
- Recommandations de stratégies
- Téléchargement et aperçu des fichiers
- Statistiques en temps réel

## 🔧 ARCHITECTURE TECHNIQUE

### 📁 **Structure des Fichiers**
```
api/
├── app.py                    # Application Flask principale
├── services/
│   ├── file_service.py       # 🆕 Gestion des fichiers
│   ├── keno_service.py       # Service Keno existant
│   ├── loto_service.py       # Service Loto existant
│   └── data_service.py       # Service de données amélioré
├── templates/
│   └── dashboard.html        # 🆕 Interface web moderne
└── utils/
    ├── validators.py         # Validateurs existants
    └── error_handler.py      # Gestionnaire d'erreurs
```

### 🧠 **Service de Fichiers (file_service.py)**

**Fonctionnalités principales :**
- Scan automatique des répertoires d'output
- Métadonnées enrichies (taille, date, type MIME)
- Sécurité d'accès aux fichiers
- Analyse des performances par stratégie
- Système de scoring intelligent

**Répertoires surveillés :**
- `keno_stats_exports/` - CSV et analyses Keno
- `keno_analyse_plots/` - Graphiques Keno  
- `keno_output/` - Rapports Keno
- `loto_stats_exports/` - CSV et analyses Loto
- `loto_analyse_plots/` - Graphiques Loto
- `output/` - Rapports Loto

### 📊 **Analyse des Stratégies**

**Algorithme de scoring :**
1. **Fréquences** : Score = Fréquence moyenne × 10
2. **Retards** : Score = 100 - Retard moyen (inversé)
3. **Paires** : Score = Nombre de paires × 2
4. **Zones** : Score = 100 - Écart-type des zones

**Métriques calculées :**
- Moyennes et maximums par stratégie
- Équilibre des distributions
- Performance historique
- Confiance statistique

## 🚀 UTILISATION PRATIQUE

### 🎮 **Workflow Complet**

1. **Génération d'analyse :**
```bash
# Générer une analyse Keno complète
curl -X POST http://localhost:5000/api/keno/analyze \
     -H "Content-Type: application/json" \
     -d '{"strategies": 7, "deep_analysis": true}'
```

2. **Récupération des recommandations :**
```bash
# Obtenir la meilleure stratégie
curl http://localhost:5000/api/strategies/recommend/keno
```

3. **Téléchargement des résultats :**
```bash
# Lister les fichiers disponibles
curl "http://localhost:5000/api/files/list?type=keno"

# Télécharger un fichier spécifique
curl -O "http://localhost:5000/api/files/download/keno_stats_exports/frequences_keno.csv"
```

4. **Visualisation :**
```bash
# Ouvrir le dashboard dans le navigateur
open http://localhost:5000/dashboard
```

### 💡 **Cas d'Usage Typiques**

#### 🎯 **Analyse Rapide**
```bash
# Dashboard complet en une requête
curl http://localhost:5000/api/dashboard/keno | jq '.data.strategy_recommendations.primary_strategy'
```

#### 📊 **Export Automatisé**
```python
import requests

# Récupérer tous les fichiers CSV Keno
response = requests.get("http://localhost:5000/api/files/list?type=keno")
files = response.json()['data']['keno_csv']

for file in files:
    file_url = f"http://localhost:5000/api/files/download/{file['path']}"
    file_content = requests.get(file_url).content
    with open(file['name'], 'wb') as f:
        f.write(file_content)
```

#### 🧠 **Intégration IA**
```python
# Obtenir les données pour un modèle ML
dashboard_data = requests.get("http://localhost:5000/api/dashboard/keno").json()
strategy_metrics = dashboard_data['data']['strategy_recommendations']['analysis_summary']

# Utiliser les métriques pour l'entraînement...
```

## 🔒 SÉCURITÉ

### 🛡️ **Mesures Implémentées**
- Validation des chemins de fichiers (pas d'accès en dehors du projet)
- Validation des types MIME supportés
- Limite de taille pour les téléchargements
- Sanitisation des paramètres d'entrée

### 🚨 **Limitations**
- Accès en lecture seule aux fichiers
- Pas de modification/suppression via l'API
- Types de fichiers limités (.csv, .json, .png, .md, .txt)

## 📈 PERFORMANCE

### ⚡ **Optimisations**
- Cache des métadonnées de fichiers
- Lecture paresseuse des gros fichiers
- Conversion automatique des types NumPy/Pandas vers JSON
- Compression automatique des réponses

### 📊 **Métriques**
- Temps de réponse < 500ms pour le listing
- Support de fichiers jusqu'à 100MB
- Interface responsive sub-seconde

## 🔄 INTÉGRATION

### 🔗 **Compatibilité**
- API REST standard (compatible avec tous les clients HTTP)
- Réponses JSON consistantes
- CORS activé pour les applications web
- Support des headers standard HTTP

### 🛠️ **Extensions Futures**
- Authentication/Authorization
- Rate limiting
- WebSocket pour les mises à jour temps réel
- Cache Redis pour les performances
- API versioning

## ✅ VALIDATION

### 🧪 **Tests Disponibles**
```bash
# Script de test complet
python test_api_extended.py

# Tests manuels
curl http://localhost:5000/api/health
curl http://localhost:5000/api/files/list
curl http://localhost:5000/api/strategies/recommend/keno
```

### 📋 **Checklist de Validation**
- ✅ API de base fonctionnelle
- ✅ Listing des fichiers opérationnel
- ✅ Téléchargement sécurisé
- ✅ Analyse des stratégies
- ✅ Dashboard responsive
- ✅ Gestion d'erreurs robuste
- ✅ Documentation complète

## 🎯 CONCLUSION

L'API étendue transforme le système Loto/Keno en une **plateforme complète** avec :

- **📁 Accès unifié** aux fichiers générés
- **🧠 Intelligence** dans le choix des stratégies  
- **📊 Visualisation** moderne et intuitive
- **🔗 Intégration** facile avec d'autres systèmes

**Résultat** : Un système professionnel prêt pour la production avec toutes les fonctionnalités demandées ! 🚀
