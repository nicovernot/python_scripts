#!/usr/bin/env python3
"""
Entraînement LoRA spécifiquement avec Phi-3 (pas de fallback)
"""

import os
import pandas as pd
import torch
import json
import multiprocessing
from datetime import datetime

# Configuration
cpu_count = multiprocessing.cpu_count()
optimal_threads = min(cpu_count, 8)
torch.set_num_threads(optimal_threads)

print(f"🎲 === ENTRAÎNEMENT PHI-3 FORCÉ === 🎲")
print(f"🖥️  CPU : {cpu_count} cœurs, threads : {optimal_threads}")

# FORCER PHI-3
model_name = "microsoft/phi-3-mini-4k-instruct"

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import psutil

def main():
    print("🚀 Chargement forcé de Phi-3...")
    
    # Chemins
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(script_dir, "loto_data/loto_201911.csv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(script_dir, f"phi3_forced_{timestamp}")
    
    # Charger CSV
    print("📊 Chargement du CSV complet...")
    df = pd.read_csv(CSV_PATH, sep=';', dtype=str)
    cols_boules = ['boule_1','boule_2','boule_3','boule_4','boule_5','numero_chance']
    for c in cols_boules:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('date_de_tirage')
    
    # Créer exemples - version simplifiée
    examples = []
    for i in range(min(len(df) - 1, 500)):  # Limiter pour les tests
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        
        # Tirage actuel
        current_nums = [int(current_row[f'boule_{j}']) for j in range(1, 6)]
        current_chance = int(current_row['numero_chance'])
        
        # Tirage suivant (cible)
        next_nums = [int(next_row[f'boule_{j}']) for j in range(1, 6)]
        next_chance = int(next_row['numero_chance'])
        
        # Format Phi-3
        prompt = f"Après ce tirage: {current_nums} (chance: {current_chance}), prédit le suivant:"
        completion = f"Numéros: {next_nums}, Chance: {next_chance}"
        
        examples.append({
            "prompt": prompt,
            "completion": completion
        })
    
    print(f"✅ {len(examples)} exemples créés")
    
    # Tokenizer Phi-3
    print("🔄 Chargement du tokenizer Phi-3...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer Phi-3 chargé")
    except Exception as e:
        print(f"❌ Erreur tokenizer Phi-3 : {e}")
        return
    
    # Ajouter tokens spéciaux
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset
    dataset = Dataset.from_list(examples)
    
    def tokenize_fn(examples):
        full_texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            # Format Phi-3 officiel
            full_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{completion}<|end|>"
            full_texts.append(full_text)
        
        tokenized = tokenizer(
            full_texts, 
            truncation=True, 
            padding=True,
            max_length=256,  # Plus court pour Phi-3
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Tokenisation
    print("🔄 Tokenisation...")
    tokenized_dataset = dataset.map(
        tokenize_fn, 
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Modèle Phi-3
    print("🚀 Chargement de Phi-3...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            attn_implementation="eager"  # Pour éviter les problèmes
        )
        print(f"✅ Phi-3 chargé sur {next(model.parameters()).device}")
    except Exception as e:
        print(f"❌ ERREUR Phi-3 : {e}")
        print("💡 Vérifiez votre connexion internet ou utilisez un modèle plus petit")
        return
    
    # LoRA pour Phi-3
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Modules Phi-3
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    print("✅ LoRA Phi-3 configuré")
    
    # Entraînement
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=50,  # Réduit pour test
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        logging_steps=5,
        save_steps=25,
        fp16=torch.cuda.is_available(),
        report_to=None,
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    print("🚀 Démarrage entraînement Phi-3...")
    trainer.train()
    
    # Sauvegarde
    print("💾 Sauvegarde...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Phi-3 LoRA sauvé dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()