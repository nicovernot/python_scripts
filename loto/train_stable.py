#!/usr/bin/env python3
"""
Script d'entraînement LoRA optimisé et stable
Utilise des modèles légers mais efficaces
"""

import os
import pandas as pd
import torch
import json
import multiprocessing
from datetime import datetime

# Configuration CPU optimisée
cpu_count = multiprocessing.cpu_count()
optimal_threads = min(cpu_count, 4)  # Limité pour éviter les plantages
torch.set_num_threads(optimal_threads)

print(f"🖥️  CPU : {cpu_count} cœurs (utilisation: {optimal_threads})")

# Modèles légers et stables
LIGHTWEIGHT_MODELS = [
    "distilgpt2",           # Très léger, rapide
    "gpt2",                 # Léger, efficace  
    "microsoft/DialoGPT-small"  # Fallback
]

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import psutil

def select_best_model():
    """Sélectionne le meilleur modèle disponible selon les ressources"""
    ram_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"RAM disponible : {ram_gb:.1f} GB")
    
    if ram_gb >= 4.0:
        return "gpt2"  # Bon compromis
    elif ram_gb >= 2.0:
        return "distilgpt2"  # Plus léger
    else:
        return "microsoft/DialoGPT-small"  # Le plus léger

def create_smart_examples(df, max_examples=500):
    """Crée des exemples d'entraînement optimisés"""
    examples = []
    
    # Limiter selon la RAM
    ram_gb = psutil.virtual_memory().available / (1024**3)
    if ram_gb < 3.0:
        max_examples = 200
    elif ram_gb < 4.0:
        max_examples = 350
    
    print(f"Génération de {max_examples} exemples maximum")
    
    for i in range(min(len(df) - 1, max_examples)):
        if i + 1 < len(df):
            # Tirage courant
            current_row = df.iloc[i]
            current_draw = [int(current_row[f'boule_{k}']) for k in range(1, 6)]
            current_chance = int(current_row['numero_chance'])
            
            # Tirage suivant (cible)
            next_row = df.iloc[i + 1]
            next_draw = [int(next_row[f'boule_{k}']) for k in range(1, 6)]
            next_chance = int(next_row['numero_chance'])
            
            # Plusieurs formats de prompt
            prompts = [
                f"Tirage précédent: {current_draw}. Prochain tirage:",
                f"Analyse: {current_draw} (chance: {current_chance}). Prédit:",
                f"Séquence: {current_draw}. Suivant:"
            ]
            
            response = f"{next_draw} chance:{next_chance}"
            
            for prompt in prompts[:2]:  # Limiter à 2 variations
                examples.append({
                    "prompt": prompt,
                    "completion": response
                })
                
                if len(examples) >= max_examples:
                    return examples
    
    return examples

def main():
    """Entraînement optimisé et stable"""
    print("🎲 === ENTRAÎNEMENT LOTO STABLE === 🎲")
    print("=" * 40)
    
    # Sélectionner le modèle optimal
    model_name = select_best_model()
    print(f"🤖 Modèle sélectionné : {model_name}")
    
    # Chargement des données
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "loto_data/loto_201911.csv")
    
    print(f"📊 Chargement : {csv_path}")
    df = pd.read_csv(csv_path, sep=';', dtype=str)
    
    # Conversion des colonnes
    cols_boules = ['boule_1','boule_2','boule_3','boule_4','boule_5','numero_chance']
    for c in cols_boules:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('date_de_tirage')
    
    # Créer les exemples
    examples = create_smart_examples(df)
    print(f"✅ {len(examples)} exemples créés")
    
    if len(examples) < 10:
        print("❌ Pas assez d'exemples")
        return
    
    # Dataset
    dataset = Dataset.from_list(examples)
    
    # Tokenizer
    print(f"📝 Chargement tokenizer {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Erreur tokenizer : {e}")
        return
    
    def tokenize_fn(batch):
        """Tokenisation optimisée"""
        texts = []
        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            # Format simple et efficace
            text = f"{prompt} {completion}{tokenizer.eos_token}"
            texts.append(text)
        
        result = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,  # Court pour éviter les problèmes mémoire
            return_tensors="pt"
        )
        result["labels"] = result["input_ids"].clone()
        return result
    
    # Tokenisation
    print("🔄 Tokenisation...")
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Modèle
    print(f"🔧 Chargement modèle {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print("✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur modèle : {e}")
        return
    
    # Configuration LoRA conservative
    lora_config = LoraConfig(
        r=4,  # Petit rang
        lora_alpha=8,  # Alpha modéré
        target_modules=["c_attn"] if "gpt" in model_name else ["lm_head"],
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    
    try:
        model = get_peft_model(model, lora_config)
        print("✅ LoRA configuré")
    except Exception as e:
        print(f"❌ Erreur LoRA : {e}")
        return
    
    # Dossier de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, f"loto_stable_{timestamp}")
    
    # Configuration d'entraînement conservative
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=50,  # Peu d'étapes pour éviter les plantages
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,  # Learning rate plus faible
        logging_steps=10,
        save_steps=25,
        warmup_steps=5,
        fp16=False,
        dataloader_num_workers=0,  # Désactiver le multiprocessing
        remove_unused_columns=False,
        report_to=None,
        save_safetensors=False  # Utiliser le format pickle
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    print("🚀 Entraînement...")
    try:
        trainer.train()
        print("✅ Entraînement terminé")
        
        # Sauvegarde
        print("💾 Sauvegarde...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Métadonnées
        metadata = {
            "model_name": model_name,
            "examples_count": len(examples),
            "max_steps": 50,
            "lora_r": 4,
            "status": "stable_training_complete"
        }
        
        with open(f"{output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Modèle sauvé : {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"❌ Erreur entraînement : {e}")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"🎯 Succès ! Modèle disponible dans : {result}")
    else:
        print("❌ Échec de l'entraînement")