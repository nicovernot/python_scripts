#!/usr/bin/env python3
"""
Script de prédiction Loto utilisant le modèle LoRA entraîné
"""

import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from datetime import datetime

def load_model_and_tokenizer(model_path="./loto_lora_model"):
    """
    Charge le modèle LoRA et le tokenizer
    """
    print(f"🔄 Chargement du modèle depuis : {model_path}")
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Le modèle {model_path} n'existe pas")
        print("💡 Conseil : Lancez d'abord train_lora.py pour entraîner le modèle")
        return None, None
    
    try:
        # Charger le modèle de base
        print("📦 Chargement du modèle de base...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",  # Même modèle que l'entraînement
            torch_dtype=torch.float32,
            device_map="cpu"  # Forcer CPU si pas de GPU
        )
        
        # Charger l'adaptateur LoRA
        print("🔧 Chargement de l'adaptateur LoRA...")
        lora_model = PeftModel.from_pretrained(base_model, model_path)
        
        # Charger le tokenizer
        print("📝 Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ajouter le pad token si nécessaire
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Modèle et tokenizer chargés avec succès")
        return lora_model, tokenizer
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None, None

def load_recent_draws(csv_path, n_draws=5):
    """
    Charge les n derniers tirages du CSV
    """
    try:
        df = pd.read_csv(csv_path, sep=';', dtype=str)
        
        # Convertir les colonnes numériques
        cols_boules = ['boule_1','boule_2','boule_3','boule_4','boule_5','numero_chance']
        for c in cols_boules:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        # Convertir et trier par date
        df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
        df = df.sort_values('date_de_tirage', ascending=False)
        
        # Prendre les n derniers tirages
        recent_draws = []
        for _, row in df.head(n_draws).iterrows():
            draw = [int(row[col]) for col in cols_boules[:-1]]  # Les 5 boules
            chance = int(row['numero_chance'])  # Le numéro chance
            recent_draws.append({
                'date': row['date_de_tirage'].strftime('%d/%m/%Y'),
                'boules': draw,
                'chance': chance
            })
            
        return recent_draws
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des tirages : {e}")
        return []

def create_prediction_prompt(recent_draws):
    """
    Crée le prompt pour la prédiction basé sur les tirages récents
    """
    # Formater les tirages récents
    draws_text = []
    for draw in recent_draws:
        boules_str = ','.join(map(str, draw['boules']))
        draws_text.append(f"[{boules_str}] (chance: {draw['chance']}) - {draw['date']}")
    
    prompt = f"""Analyse ces {len(recent_draws)} derniers tirages du Loto Français :
{chr(10).join(draws_text)}

Basé sur ces données historiques, prédit les 5 numéros et le numéro chance pour le prochain tirage :"""
    
    return prompt

def generate_prediction(model, tokenizer, prompt, max_tokens=100):
    """
    Génère une prédiction avec le modèle
    """
    print("🎯 Génération de la prédiction...")
    
    try:
        # Tokeniser le prompt
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Vérifier le device du modèle
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Générer avec des paramètres adaptés
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Décoder la réponse
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la partie générée (après le prompt)
        prediction = generated_text[len(prompt):].strip()
        
        return prediction
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération : {e}")
        return None

def extract_numbers_from_prediction(prediction_text):
    """
    Extrait les numéros de la prédiction générée
    """
    import re
    
    # Chercher des patterns de numéros
    # Pattern pour 5 numéros entre 1 et 49
    boules_pattern = r'(\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})[,\s]*(\d{1,2})'
    # Pattern pour le numéro chance entre 1 et 10
    chance_pattern = r'chance[:\s]*(\d{1,2})'
    
    boules_match = re.search(boules_pattern, prediction_text)
    chance_match = re.search(chance_pattern, prediction_text.lower())
    
    boules = []
    chance = None
    
    if boules_match:
        boules = [int(x) for x in boules_match.groups()]
        # Vérifier que les numéros sont dans la bonne plage
        boules = [b for b in boules if 1 <= b <= 49]
    
    if chance_match:
        chance = int(chance_match.group(1))
        if not (1 <= chance <= 10):
            chance = None
    
    return boules, chance

def main():
    """
    Fonction principale
    """
    print("🎲 === PRÉDICTEUR LOTO IA === 🎲")
    print("=" * 40)
    
    # Chemins des fichiers
    model_path = "/home/nico/projets/python_scripts/loto/phi3_lora_loto"
    csv_path = "/home/nico/projets/python_scripts/loto/loto_data/loto_201911.csv"
    
    # Charger le modèle
    model, tokenizer = load_model_and_tokenizer(model_path)
    if model is None:
        return
    
    # Charger les tirages récents
    print(f"📊 Chargement des tirages récents depuis : {csv_path}")
    recent_draws = load_recent_draws(csv_path, n_draws=5)
    
    if not recent_draws:
        print("❌ Impossible de charger les tirages récents")
        return
    
    print(f"✅ {len(recent_draws)} tirages récents chargés")
    
    # Afficher les tirages récents
    print("\n📈 Tirages récents :")
    for i, draw in enumerate(recent_draws, 1):
        boules_str = ' - '.join(map(str, draw['boules']))
        print(f"  {i}. {boules_str} (chance: {draw['chance']}) [{draw['date']}]")
    
    # Créer le prompt
    prompt = create_prediction_prompt(recent_draws)
    print(f"\n🤖 Prompt généré :")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # Générer la prédiction
    prediction = generate_prediction(model, tokenizer, prompt)
    
    if prediction:
        print(f"\n🎯 Prédiction générée :")
        print(prediction)
        
        # Extraire les numéros
        boules, chance = extract_numbers_from_prediction(prediction)
        
        if boules and len(boules) >= 5:
            print(f"\n🎲 Prédiction extraite :")
            print(f"   Boules : {' - '.join(map(str, boules[:5]))}")
            if chance:
                print(f"   Chance : {chance}")
            else:
                print(f"   Chance : (non détecté)")
        else:
            print(f"\n⚠️  Impossible d'extraire des numéros valides de la prédiction")
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/nico/projets/python_scripts/loto_output/prediction_{timestamp}.json"
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialiser les variables si pas définies
        if 'boules' not in locals():
            boules = []
        if 'chance' not in locals():
            chance = None
            
        result = {
            "timestamp": timestamp,
            "recent_draws": recent_draws,
            "prompt": prompt,
            "raw_prediction": prediction,
            "extracted_numbers": {
                "boules": boules[:5] if boules and len(boules) >= 5 else [],
                "chance": chance
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés dans : {output_file}")
        
    except Exception as e:
        print(f"⚠️  Erreur lors de la sauvegarde : {e}")
    
    print("\n✅ Prédiction terminée !")

if __name__ == "__main__":
    main()