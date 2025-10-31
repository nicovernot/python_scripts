import os
import pandas as pd
import subprocess
import requests
import json

# === CONFIGURATION ===
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "loto_data/loto_201911.csv")
MODEL_NAME = "phi3:latest"   # Nom tel qu'apparu dans `ollama list` (ou "phi3")
NB_LAST_DRAWS = 30
OLLAMA_HTTP = "http://localhost:11434/api/generate"  # endpoint Ollama local

# === Chargement CSV ===
print(f"Chargement des données depuis : {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Le fichier CSV n'existe pas : {CSV_PATH}")
df = pd.read_csv(CSV_PATH, sep=';', dtype=str)
cols_boules = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'numero_chance']
for c in cols_boules:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
df = df.sort_values('date_de_tirage')
last_draws = df.tail(NB_LAST_DRAWS)
all_boules = pd.concat([last_draws[c] for c in cols_boules[:-1]])
freqs = all_boules.value_counts().sort_index()
top_freq = freqs.sort_values(ascending=False).head(5).to_dict()
low_freq = freqs.sort_values(ascending=True).head(5).to_dict()
mean_ball = all_boules.mean()

# Réduire les données pour éviter un prompt trop long
recent_summary = last_draws[['date_de_tirage','boule_1','boule_2','boule_3','boule_4','boule_5','numero_chance']].tail(10)

prompt = f"""Analyse statistique du Loto français.

Statistiques sur {NB_LAST_DRAWS} tirages récents :
- Numéros les plus fréquents : {top_freq}
- Numéros les moins fréquents : {low_freq}
- Moyenne des boules : {mean_ball:.1f}

Derniers tirages (extrait) :
{recent_summary.to_string(index=False)}

Mission :
1. Identifie 3 numéros "chauds" et 3 "froids"
2. Propose 5 numéros pour le prochain tirage
3. Suggère 1 numéro chance
4. Justification en 2 phrases maximum

Réponds de façon concise."""

print("🧠 Prompt préparé (longueur de prompt):", len(prompt))

# === Vérification préalable d'Ollama ===
def check_ollama_status():
    try:
        # Vérifier si Ollama répond
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"📋 Modèles disponibles : {model_names}")
            
            # Vérifier si notre modèle existe
            if any(MODEL_NAME in name for name in model_names):
                print(f"✅ Modèle '{MODEL_NAME}' trouvé")
                return True
            else:
                print(f"❌ Modèle '{MODEL_NAME}' non trouvé. Modèles disponibles : {model_names}")
                return False
        else:
            print(f"❌ Ollama ne répond pas (status: {resp.status_code})")
            return False
    except Exception as e:
        print(f"❌ Impossible de contacter Ollama : {e}")
        return False

if not check_ollama_status():
    print("\n🔧 Pour résoudre le problème :")
    print("1. Démarrer Ollama : 'ollama serve'")
    print(f"2. Installer le modèle : 'ollama pull {MODEL_NAME}'")
    print("3. Vérifier les modèles : 'ollama list'")
    exit(1)

# === Tentative 1 : appel HTTP direct à Ollama (recommandé) ===
def call_ollama_http(prompt_text):
    try:
        payload = {
            "model": MODEL_NAME, 
            "prompt": prompt_text,
            "stream": False,  # Important : désactiver le streaming pour avoir une réponse complète
            "options": {
                "temperature": 0.7,  # Réduire la créativité pour plus de vitesse
                "top_p": 0.9,
                "num_predict": 400   # Limiter la longueur de la réponse
            }
        }
        resp = requests.post(OLLAMA_HTTP, json=payload, timeout=180)  # Augmenter le timeout à 3 minutes
        # debug
        print("HTTP status:", resp.status_code)
        if resp.status_code != 200:
            print("HTTP response text:", resp.text[:1000])
            return None
        data = resp.json()
        # selon version Ollama la clé peut être "response" ou "output"
        return data.get("response") or data.get("output") or json.dumps(data, indent=2)
    except Exception as e:
        print("Erreur appel HTTP Ollama:", str(e))
        return None

print("👉 Essai via API HTTP d'Ollama...")
http_out = call_ollama_http(prompt)
if http_out:
    print("\n🎯 Rapport (via HTTP Ollama) :\n")
    print(http_out)
    exit(0)
else:
    print("Échec de l'appel HTTP — on tente le fallback CLI...")

# === Fallback : appel CLI ollama ===
def call_ollama_cli(prompt_text):
    try:
        # Méthode 1 : Passer le prompt via stdin (plus fiable)
        proc = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt_text,
            capture_output=True, text=True, timeout=180
        )
    except FileNotFoundError:
        print("Commande 'ollama' introuvable. Vérifie ton PATH ou installe Ollama.")
        return None
    except subprocess.TimeoutExpired:
        print("La commande ollama a expiré.")
        return None

    print("CLI exit code:", proc.returncode)
    if proc.stderr:
        print("CLI stderr (début):\n", proc.stderr[:1000])
    if proc.returncode != 0:
        print("La commande ollama a retourné un code d'erreur.")
        return None
    return proc.stdout

cli_out = call_ollama_cli(prompt)
if cli_out:
    print("\n🎯 Rapport (via CLI Ollama) :\n")
    print(cli_out)
else:
    print("\n❌ Échec : ni l'appel HTTP ni la CLI n'ont renvoyé de sortie exploitable.")
    print("→ Vérifie que le démon Ollama tourne, que le modèle existe (ollama list), et que 'ollama' est dans le PATH.")
    print("Commandes utiles :")
    print("  ollama status")
    print("  ollama list")
