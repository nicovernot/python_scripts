# Documentation API Loto/Keno

## Vue d'ensemble

L'API Loto/Keno est une API RESTful qui expose les fonctionnalités d'analyse et de génération de grilles pour les jeux Loto et Keno de la FDJ.

## URL de base

```
http://localhost:5000
```

## Endpoints

### 1. Documentation et Santé

#### GET `/`
Affiche la page de documentation HTML de l'API.

**Réponse :** Page HTML avec documentation interactive

#### GET `/api/health`
Vérifie l'état de santé de l'API et de ses services.

**Réponse :**
```json
{
    "status": "healthy",
    "timestamp": "2025-08-13T19:47:20.894395",
    "services": {
        "loto": true,
        "keno": true,
        "data": true
    },
    "version": "1.0.0"
}
```

#### GET `/api/config`
Retourne la configuration actuelle de l'API.

**Réponse :**
```json
{
    "environment": "development",
    "config_found": true,
    "version": "v2.0",
    "parameters_count": 74
}
```

---

### 2. Données

#### GET `/api/data/status`
Retourne le statut des fichiers de données Loto et Keno.

**Réponse :**
```json
{
    "loto": {
        "exists": true,
        "size_mb": 2.1,
        "last_modified": "2025-08-13T10:30:00",
        "records_count": 2847
    },
    "keno": {
        "exists": true,
        "size_mb": 1.8,
        "last_modified": "2025-08-13T10:30:00",
        "records_count": 3456
    }
}
```

#### POST `/api/data/update`
Lance la mise à jour des données depuis le site FDJ.

**Corps de la requête :**
```json
{
    "sources": ["loto", "keno"],  // optionnel, par défaut les deux
    "force": false                // optionnel, force le téléchargement
}
```

**Réponse :**
```json
{
    "success": true,
    "updated": ["loto", "keno"],
    "execution_time": 15.3,
    "details": {
        "loto": {
            "downloaded": true,
            "size_mb": 2.1,
            "records": 2847
        },
        "keno": {
            "downloaded": true,
            "size_mb": 1.8,
            "records": 3456
        }
    }
}
```

---

### 3. Loto

#### GET `/api/loto/strategies`
Retourne la liste des stratégies de génération disponibles.

**Réponse :**
```json
{
    "strategies": [
        {
            "id": "equilibre",
            "name": "Stratégie Équilibrée",
            "description": "Mélange de fréquences et retards"
        },
        {
            "id": "agressive",
            "name": "Stratégie Agressive",
            "description": "Focus sur les numéros chauds"
        },
        {
            "id": "conservatrice",
            "name": "Stratégie Conservatrice",
            "description": "Numéros les plus stables"
        },
        {
            "id": "ml_focus",
            "name": "Machine Learning",
            "description": "Algorithmes d'apprentissage automatique"
        }
    ]
}
```

#### POST `/api/loto/generate`
Génère des grilles Loto selon une stratégie donnée.

**Corps de la requête :**
```json
{
    "count": 5,                    // nombre de grilles (1-20)
    "strategy": "equilibre",       // stratégie (voir /api/loto/strategies)
    "export_csv": false,           // exporter en CSV (optionnel)
    "export_plots": false          // générer les graphiques (optionnel)
}
```

**Réponse :**
```json
{
    "success": true,
    "grids": [
        [7, 12, 23, 34, 45],      // grille 1
        [3, 18, 27, 39, 41],      // grille 2
        // ...
    ],
    "strategy": "equilibre",
    "count": 5,
    "execution_time": 2.3,
    "confidence_scores": [0.85, 0.78, 0.82, 0.79, 0.88],
    "export_info": {
        "csv_file": "output/grilles_20250813_194720.csv",
        "plots_generated": false
    }
}
```

---

### 4. Keno

#### POST `/api/keno/analyze`
Effectue une analyse Keno et génère des recommandations.

**Corps de la requête :**
```json
{
    "strategies": 7,               // nombre de stratégies (1-10)
    "deep_analysis": false,        // analyse approfondie (optionnel)
    "plots": false,                // générer les graphiques (optionnel)
    "export_stats": false          // exporter les statistiques (optionnel)
}
```

**Réponse :**
```json
{
    "success": true,
    "recommendations": {
        "strategies": [
            {
                "name": "🎯 Stratégie Fréquences",
                "numbers": [12, 23, 34, 45, 56, 67, 78, 89, 90, 11],
                "confidence": "87%",
                "description": "Basée sur les numéros les plus fréquents"
            },
            // ...
        ],
        "summary": {
            "total_strategies": 7,
            "generation_time": "2025-08-13T19:47:20"
        }
    },
    "stats": {
        "draws_analyzed": 3456,
        "analysis_type": "standard",
        "data_period": "Octobre 2020 - Août 2025"
    },
    "execution_time": 5.2
}
```

---

## Codes de statut HTTP

- **200** : Succès
- **400** : Erreur de paramètres
- **404** : Ressource non trouvée
- **500** : Erreur serveur

## Format des erreurs

```json
{
    "success": false,
    "error": "Description de l'erreur",
    "status_code": 400,
    "timestamp": "2025-08-13T19:47:20.894395"
}
```

## Exemples d'utilisation

### Générer 3 grilles Loto avec stratégie équilibrée

```bash
curl -X POST http://localhost:5000/api/loto/generate \
  -H "Content-Type: application/json" \
  -d '{
    "count": 3,
    "strategy": "equilibre",
    "export_csv": true
  }'
```

### Analyser Keno avec analyse approfondie

```bash
curl -X POST http://localhost:5000/api/keno/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": 5,
    "deep_analysis": true,
    "plots": true
  }'
```

### Vérifier la santé de l'API

```bash
curl http://localhost:5000/api/health
```

### Mettre à jour les données

```bash
curl -X POST http://localhost:5000/api/data/update \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["loto", "keno"],
    "force": false
  }'
```

## Installation et lancement

### 1. Installation des dépendances

```bash
pip install flask flask-cors duckdb pandas requests
```

### 2. Lancement de l'API

```bash
# Méthode 1 : Script direct
python3 api/app.py

# Méthode 2 : Script de lancement
./lancer_api.sh

# Méthode 3 : Flask CLI
export FLASK_APP=api/app.py
flask run --host=0.0.0.0 --port=5000
```

### 3. Configuration

L'API utilise le fichier `.env` pour sa configuration. Les paramètres principaux :

```bash
# Serveur Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false

# Chemins des données
LOTO_CSV_PATH=loto/loto_data/nouveau_loto.csv
KENO_CSV_PATH=keno/keno_data/keno_202010.csv

# Répertoires de sortie
LOTO_OUTPUT_PATH=output
KENO_OUTPUT_PATH=keno_output
```

## Support et développement

- **Tests** : `python3 test_api.py`
- **Logs** : Les logs sont affichés dans la console lors du développement
- **Performance** : L'API est optimisée pour des réponses < 30 secondes
- **Threading** : Support des requêtes simultanées

## Limitations

- Génération Loto : 1-20 grilles maximum par requête
- Analyse Keno : Timeout de 10 minutes pour l'analyse approfondie
- Données : Mise à jour depuis FDJ nécessite une connexion internet
- Fichiers : Les exports CSV/graphiques sont stockés localement

## Changelog

- **v1.0.0** (13/08/2025) : Version initiale avec endpoints complets
