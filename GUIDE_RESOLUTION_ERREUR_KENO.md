# 🎯 Guide de Résolution - Erreur "Erreur analyse keno: Erreur analyse Keno:"

## 🚨 Problème Identifié

L'erreur "Erreur analyse keno: Erreur analyse Keno:" se produit quand le **serveur Flask n'est pas démarré** ou s'arrête de manière inattendue.

## ✅ Solution Simple

### 1. Démarrer le serveur API

**Dans un terminal, exécutez :**

```bash
cd /home/nvernot/projets/loto_keno
source venv/bin/activate
python api/app.py
```

**Vous devriez voir :**
```
🚀 Démarrage de l'API Loto/Keno...
🌐 API disponible sur: http://0.0.0.0:5000
📚 Documentation: http://0.0.0.0:5000/
🔧 Mode debug: Désactivé
 * Running on http://127.0.0.1:5000
```

### 2. Accéder au Dashboard

**Ouvrez votre navigateur sur :**
- http://localhost:5000

### 3. Utiliser l'analyse Keno

1. Cliquez sur **"Analyser Keno"**
2. L'analyse devrait maintenant fonctionner sans erreur

## 🔧 Script de Démarrage Automatique

**Utilisez le script fourni :**

```bash
cd /home/nvernot/projets/loto_keno
./start_server.sh
```

## 🚨 Messages d'Erreur Améliorés

Le dashboard affiche maintenant des messages d'erreur plus précis :

- ❌ **"Impossible de se connecter au serveur"** → Le serveur n'est pas démarré
- ❌ **"Erreur HTTP 500"** → Erreur interne du serveur
- ❌ **"Erreur lors de l'analyse"** → Problème dans les données ou la logique

## 🎯 Vérification du Statut

**Pour vérifier que tout fonctionne :**

```bash
# Test rapide de l'API
curl -X POST http://localhost:5000/api/analysis/run/keno \
  -H "Content-Type: application/json" \
  -d '{"options": {"plots": true, "export_stats": true}}'
```

**Réponse attendue :**
```json
{
  "success": true,
  "status": "completed",
  "message": "Analyse KENO terminée avec 7 stratégies",
  "details": "Analyse KENO terminée avec 7 stratégies"
}
```

## 📚 Points Importants

1. **Le serveur doit rester actif** - Ne fermez pas le terminal où vous avez démarré l'API
2. **Port 5000** - Le dashboard est configuré pour utiliser le port 5000
3. **Environnement virtuel** - Toujours activer `venv` avant de démarrer le serveur

## 🎉 Résultat

Avec ces modifications, vous ne devriez plus voir l'erreur "Erreur analyse keno: Erreur analyse Keno:" et obtiendrez des messages d'erreur clairs pour diagnostiquer rapidement tout problème.
