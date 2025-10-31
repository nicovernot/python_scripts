#!/usr/bin/env python3
"""
Entraînement LoRA amélioré pour le Loto avec plus d'exemples et meilleurs paramètres
"""

import os
import pandas as pd
import torch
import json
import multiprocessing
from datetime import datetime

# === CONFIGURATION MULTI-THREADING ===
cpu_count = multiprocessing.cpu_count()
optimal_threads = min(cpu_count, 8)
torch.set_num_threads(optimal_threads)
torch.set_num_interop_threads(optimal_threads)

os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(f"🖥️  CPU détectés : {cpu_count} cœurs")
print(f"⚡ Configuration multi-threading : {optimal_threads} threads")

# === MODÈLE AMÉLIORÉ ===
model_name = "microsoft/phi-3-mini-4k-instruct"  # Meilleur modèle pour les prédictions

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import psutil

def check_resources():
    """Vérifier les ressources disponibles avec seuils plus permissifs"""
    ram_gb = psutil.virtual_memory().total / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"RAM disponible : {available_gb:.1f} GB sur {ram_gb:.1f} GB total")
    
    # Seuils plus permissifs pour plus d'exemples
    if available_gb < 3.0:
        print("⚠️  RAM très limitée. Mode conservateur.")
        return "conservative"
    elif available_gb < 5.0:
        print("✅ RAM suffisante. Mode normal.")
        return "normal"  
    else:
        print("🚀 RAM excellente. Mode optimisé.")
        return "optimized"

def create_better_examples(df, max_examples):
    """
    Créer des exemples d'entraînement plus riches et variés
    """
    examples = []
    
    for i in range(min(len(df) - 5, max_examples)):
        # Prendre 5 tirages consécutifs pour l'analyse
        historical_draws = []
        for j in range(5):
            if i + j < len(df):
                row = df.iloc[i + j]
                draw = [int(row[f'boule_{k}']) for k in range(1, 6)]
                chance = int(row['numero_chance'])
                date = row['date_de_tirage'].strftime('%d/%m/%Y')
                historical_draws.append({
                    'boules': draw,
                    'chance': chance, 
                    'date': date
                })
        
        if len(historical_draws) >= 5:
            # Le tirage suivant (cible à prédire)
            if i + 5 < len(df):
                target_row = df.iloc[i + 5]
                target_draw = [int(target_row[f'boule_{k}']) for k in range(1, 6)]
                target_chance = int(target_row['numero_chance'])
                
                # Créer différents types de prompts
                prompt_templates = [
                    # Template 1: Analyse historique
                    f"""Analyse ces 5 derniers tirages du Loto Français :
{chr(10).join([f'Tirage {j+1}: {h["boules"]} (chance: {h["chance"]}) - {h["date"]}' for j, h in enumerate(historical_draws)])}

Prédit les 5 numéros et le numéro chance pour le prochain tirage :""",
                    
                    # Template 2: Format plus direct
                    f"""Historique des tirages :
{chr(10).join([f'{h["boules"]}' for h in historical_draws])}

Prochain tirage prédit :""",
                    
                    # Template 3: Analyse de fréquence
                    f"""Basé sur ces tirages récents : {[h['boules'] for h in historical_draws[-3:]]}
Numéros les plus probables :"""
                ]
                
                # Réponse cible formatée
                completion = f"Numéros: {target_draw} | Chance: {target_chance}"
                
                # Ajouter plusieurs variations
                for template in prompt_templates:
                    examples.append({
                        "prompt": template,
                        "completion": completion
                    })
                    
                    # Limiter le nombre total d'exemples
                    if len(examples) >= max_examples:
                        return examples
    
    return examples

def main():
    """Entraînement amélioré"""
    print("🎲 === ENTRAÎNEMENT LORA AMÉLIORÉ === 🎲")
    print("=" * 50)
    
    # Vérifier les ressources
    resource_status = check_resources()
    
    # Configuration selon les ressources (plus généreux)
    if resource_status == "conservative":
        max_examples = 100  # Au lieu de 20
        num_workers = 2
        max_length = 256    # Au lieu de 128
        max_steps = 50      # Au lieu de 10
        lora_r = 8          # Au lieu de 2
        lora_alpha = 16     # Au lieu de 4
        print(f"Mode conservateur : {max_examples} exemples, {max_steps} steps")
    elif resource_status == "normal":
        max_examples = 300
        num_workers = cpu_count // 2
        max_length = 512
        max_steps = 100
        lora_r = 16
        lora_alpha = 32
        print(f"Mode normal : {max_examples} exemples, {max_steps} steps")
    else:  # optimized
        max_examples = 500
        num_workers = cpu_count - 1
        max_length = 512
        max_steps = 200
        lora_r = 32
        lora_alpha = 64
        print(f"Mode optimisé : {max_examples} exemples, {max_steps} steps")
    
    # Chargement des données
    CSV_PATH = "/home/nico/projets/python_scripts/loto/loto_data/loto_201911.csv"
    print(f"Chargement du CSV : {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH, sep=';', dtype=str)
    cols_boules = ['boule_1','boule_2','boule_3','boule_4','boule_5','numero_chance']
    for c in cols_boules:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('date_de_tirage')
    
    # Créer des exemples améliorés
    print("Génération des exemples d'entraînement améliorés...")
    examples = create_better_examples(df, max_examples)
    print(f"Dataset créé : {len(examples)} exemples")
    
    if len(examples) < 10:
        print("❌ Pas assez d'exemples générés")
        return
    
    # Convertir en Dataset
    dataset = Dataset.from_list(examples)
    
    # Chargement du tokenizer
    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_fn(examples):
        full_texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            full_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{completion}<|end|>"
            full_texts.append(full_text)
        
        tokenized = tokenizer(
            full_texts, 
            truncation=True, 
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Tokenisation
    print(f"🔄 Tokenisation avec {num_workers} processus...")
    tokenized_dataset = dataset.map(
        tokenize_fn, 
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Chargement du modèle
    print("Chargement du modèle avec optimisations...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print(f"✅ Modèle {model_name} chargé")
    except Exception as e:
        print(f"❌ Erreur avec {model_name}, fallback vers DialoGPT-small : {e}")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        print("✅ Modèle DialoGPT-small chargé")
    
    # Configuration LoRA améliorée
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,                    # Rang plus élevé
        lora_alpha=lora_alpha,       # Alpha plus élevé
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
        if "phi" in model_name.lower() else ["c_attn", "c_proj"]  # Plus de modules ciblés
    )
    
    model = get_peft_model(model, lora_config)
    print(f"✅ Modèle LoRA configuré avec r={lora_r}, alpha={lora_alpha}")
    
    # Configuration d'entraînement
    output_dir = f"/home/nico/projets/python_scripts/loto/loto_lora_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  # Accumulation pour batch virtuel plus grand
        learning_rate=2e-4,             # Learning rate plus élevé
        logging_steps=10,
        save_steps=max_steps // 2,
        warmup_steps=max_steps // 10,
        fp16=False,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    print("🚀 Démarrage de l'entraînement amélioré...")
    trainer.train()
    
    # Sauvegarde
    print("💾 Sauvegarde du modèle amélioré...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Métadonnées
    metadata = {
        "model_name": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "max_steps": max_steps,
        "examples_count": len(examples),
        "max_length": max_length,
        "resource_mode": resource_status,
        "improvement_version": "v2_enhanced"
    }
    
    with open(f"{output_dir}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Entraînement terminé ! Modèle sauvé dans : {output_dir}")
    print(f"📊 Statistiques : {len(examples)} exemples, {max_steps} steps, LoRA r={lora_r}")

if __name__ == "__main__":
    main()