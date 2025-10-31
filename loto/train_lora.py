import os
import pandas as pd
import torch
import json
import multiprocessing
from datetime import datetime

# === CONFIGURATION MULTI-THREADING ===
# Détection automatique du nombre de cœurs
cpu_count = multiprocessing.cpu_count()
print(f"🖥️  CPU détectés : {cpu_count} cœurs")

# Configuration optimale pour PyTorch
optimal_threads = min(cpu_count, 8)  # Limiter à 8 threads max pour éviter la sur-utilisation
torch.set_num_threads(optimal_threads)
torch.set_num_interop_threads(optimal_threads)

# Variables d'environnement pour optimiser les performances
os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(optimal_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)

# Configuration pour les tokenizers (parallélisme)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Désactiver TensorFlow pour éviter les conflits
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

print(f"⚡ Configuration multi-threading : {optimal_threads} threads")

# === CONFIGURATION DU MODÈLE ===
model_name = "microsoft/phi-3-mini-4k-instruct"  # Modèle Phi-3 pour de meilleures prédictions

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset

# Import transformers avec gestion d'erreur
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Import conditionnel pour éviter les conflits
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as e:
        print(f"Attention: Impossible d'importer Trainer: {e}")
        print("Tentative d'import alternatif...")
        from transformers.trainer import Trainer
        from transformers.training_args import TrainingArguments
        
except ImportError as e:
    print(f"Erreur d'import transformers: {e}")
    exit(1)

# Import PEFT avec gestion d'erreur
try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError as e:
    print(f"PEFT non disponible: {e}")
    print("Installation requise: pip install peft")
    exit(1)

# Import datasets
try:
    from datasets import Dataset
except ImportError as e:
    print(f"Datasets non disponible: {e}")
    print("Installation requise: pip install datasets")
    exit(1)

# === CONFIGURATION SÉCURISÉE ===
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "loto_data/loto_201911.csv")
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"  # modèle open-source compatible LoRA
# Nouveau dossier pour le modèle Phi-3 avec dataset complet
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(script_dir, f"phi3_lora_full_{timestamp}")
NB_LAST_DRAWS = 5  # nombre de tirages pour générer un prompt

# Vérification des ressources système
import psutil
import threading
import time
# GPUtil n'est pas toujours disponible, on utilise torch directement pour le GPU

def check_system_resources():
    # Vérifier la RAM disponible
    ram = psutil.virtual_memory()
    ram_gb = ram.available / (1024**3)
    print(f"RAM disponible : {ram_gb:.1f} GB")
    
    # Vérifier le GPU si disponible
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU détectés : {gpu_count}")
        for i in range(gpu_count):
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {gpu_mem:.1f} GB")
    else:
        print("Pas de GPU CUDA détecté - utilisation CPU uniquement")
    
    # Recommandations (seuils ajustés pour être plus permissif)
    if ram_gb < 4:
        print("⚠️  ATTENTION: RAM très insuffisante (<4GB). Arrêt recommandé.")
        return False
    elif ram_gb < 8:
        print("⚠️  RAM limitée mais utilisable. Mode ultra-conservateur.")
        return "ultra_conservative"
    elif ram_gb < 16:
        print("⚠️  RAM correcte. Utilisation de paramètres conservateurs.")
        return "conservative"
    else:
        print("✅ RAM suffisante pour l'entraînement")
        return True

# Vérifier avant de continuer
resource_status = check_system_resources()

# === 1. Charger le CSV ===
print(f"Chargement du CSV : {CSV_PATH}")
df = pd.read_csv(CSV_PATH, sep=';', dtype=str)
cols_boules = ['boule_1','boule_2','boule_3','boule_4','boule_5','numero_chance']
for c in cols_boules:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['date_de_tirage'] = pd.to_datetime(df['date_de_tirage'], format='%d/%m/%Y', errors='coerce')
df = df.sort_values('date_de_tirage')

# === 2. Créer le dataset prompt/réponse (version limitée) ===
print("Génération des exemples prompt/réponse...")

# Utiliser tout le CSV comme demandé
print("🎯 Configuration : Utilisation de TOUT le dataset CSV")
max_examples = len(df)  # TOUT le dataset !
num_workers = min(cpu_count - 1, 4)  # Optimisé mais stable

if resource_status == False:
    print("❌ Ressources insuffisantes. Arrêt du script.")
    exit(1)
elif resource_status == "ultra_conservative":
    print(f"Mode ultra-conservateur mais TOUT le dataset : {max_examples} exemples")
elif resource_status == "conservative":
    print(f"Mode conservateur mais TOUT le dataset : {max_examples} exemples")
else:
    print(f"Mode normal avec TOUT le dataset : {max_examples} exemples")

examples = []
for i in range(NB_LAST_DRAWS, min(len(df), NB_LAST_DRAWS + max_examples)):
    tirages = df.iloc[i-NB_LAST_DRAWS:i][cols_boules].values.tolist()
    prompt = f"Analyse ces {NB_LAST_DRAWS} derniers tirages du Loto français : {tirages}. Donne les 5 numéros les plus probables et 1 numéro chance."
    # Réponse fictive : numéros les plus fréquents dans ces tirages (exemple simple)
    flat = [b for t in tirages for b in t[:-1]]  # boules sans numéro chance
    counts = pd.Series(flat).value_counts()
    top5 = counts.head(5).index.tolist()
    chance = tirages[-1][-1]  # dernier numéro chance
    completion = f"Numéros probables : {top5}, Numéro chance : {chance}"
    examples.append({"prompt": prompt, "completion": completion})

print(f"Dataset créé : {len(examples)} exemples")

# Convertir en Dataset HuggingFace
dataset = Dataset.from_list(examples)

# === 3. Tokenizer & Tokenization ===
print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ajouter un token de padding si absent
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configurer la longueur maximale (optimisé pour Phi-3)
if resource_status == "ultra_conservative":
    max_length = 256  # Plus long pour Phi-3
elif resource_status == "conservative":
    max_length = 512  # Optimal pour Phi-3
else:
    max_length = 1024  # Max pour Phi-3

def tokenize_fn(examples):
    # Format optimisé pour Phi-3
    full_texts = []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        # Format Phi-3 avec tokens spéciaux
        full_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{completion}<|end|>"
        full_texts.append(full_text)
    
    # Tokeniser avec la longueur adaptée aux ressources
    tokenized = tokenizer(
        full_texts, 
        truncation=True, 
        padding=True,
        max_length=max_length,  # Utiliser la variable définie plus haut
        return_tensors=None
    )
    
    # Pour l'entraînement causal, les labels sont identiques aux input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # Ajouter la longueur pour le grouping
    tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
    
    return tokenized

# Tokenisation avec parallélisation
print(f"🔄 Tokenisation avec {num_workers} processus...")
tokenized_dataset = dataset.map(
    tokenize_fn, 
    batched=True,
    num_proc=num_workers,  # Utiliser le multi-processing
    remove_columns=dataset.column_names,  # Supprimer les colonnes originales
    desc="Tokenisation"
)

# === 4. Charger le modèle et configurer LoRA (version sécurisée) ===
print("Chargement du modèle avec gestion mémoire...")

# Configuration selon les ressources
if resource_status == "ultra_conservative":
    # Mode ultra-conservateur pour RAM < 8GB
    device_map = "cpu"
    torch_dtype = torch.float32
    max_length = 128  # Très court
    batch_size = 1
    grad_accum = 1
    model_name = "microsoft/DialoGPT-small"  # Modèle plus petit
    print("Mode ultra-conservateur : CPU, DialoGPT-small, batch_size=1")
elif resource_status == "conservative":
    # Mode conservateur
    device_map = "cpu"  # Forcer sur CPU
    torch_dtype = torch.float32  # Précision normale
    max_length = 256  # Séquences plus courtes
    batch_size = 1
    grad_accum = 2
    model_name = MODEL_NAME  # Phi-3 original
    print("Mode conservateur : CPU uniquement, batch_size=1")
else:
    # Mode normal mais sécurisé
    device_map = "auto"
    torch_dtype = torch.float16
    max_length = 512
    batch_size = 1  # Toujours conservateur pour éviter les plantages
    grad_accum = 4
    model_name = MODEL_NAME
    print("Mode normal : GPU/CPU auto, batch_size=1")

# Nettoyer le cache GPU si disponible
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cache GPU nettoyé")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # Utiliser le model_name adapté aux ressources
        device_map=device_map, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,  # Optimisation mémoire
        trust_remote_code=True
    )
    print(f"✅ Modèle {model_name} chargé sur {device_map}")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    print("Tentative avec le modèle le plus petit...")
    model_name = "microsoft/DialoGPT-small"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="cpu", 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    device_map = "cpu"
    torch_dtype = torch.float32
    batch_size = 1
    grad_accum = 1
    max_length = 64

# Configuration des modules cibles pour Phi-3
def get_phi3_target_modules(model_name):
    """Retourne les modules cibles optimaux selon le modèle"""
    if "phi-3" in model_name.lower():
        # Modules spécifiques à Phi-3
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "dialogpt" in model_name.lower():
        # Modules pour DialoGPT
        return ["c_attn", "c_proj"]
    else:
        # Modules génériques
        return ["q_proj", "v_proj"]

# Sélectionner les modules cibles selon le modèle
target_modules = get_phi3_target_modules(model_name)
print(f"Modules cibles sélectionnés pour {model_name}: {target_modules}")

# Fonction pour détecter les modules disponibles (fallback)
def find_available_modules(model):
    available = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = name.split('.')[-1]
            available.add(module_name)
    return list(available)

# Vérifier que les modules existent
all_modules = find_available_modules(model)
print(f"Modules Linear disponibles : {all_modules}")

# Filtrer les modules cibles qui existent réellement
existing_targets = [m for m in target_modules if m in all_modules]
if not existing_targets:
    # Fallback vers modules génériques
    existing_targets = [m for m in all_modules if any(x in m.lower() for x in ['proj', 'attn', 'head'])][:3]

final_targets = existing_targets or ["lm_head"]  # Dernier recours
print(f"Modules cibles finaux sélectionnés : {final_targets}")

# Configuration LoRA optimisée pour Phi-3 et training complet
print("🔧 Configuration LoRA pour training complet avec Phi-3")
if resource_status == "ultra_conservative":
    lora_r = 8   # Plus élevé pour meilleur apprentissage
    lora_alpha = 16
    max_steps = 100  # Plus d'étapes
elif resource_status == "conservative": 
    lora_r = 16  # Bon équilibre
    lora_alpha = 32
    max_steps = 200
else:
    lora_r = 32  # Maximum pour les meilleures performances
    lora_alpha = 64
    max_steps = 300

lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=final_targets,  # Utiliser les modules finaux
    lora_dropout=0.1,  # Dropout plus élevé pour éviter overfitting
    task_type="CAUSAL_LM",
    bias="none"
)

print(f"LoRA Config: r={lora_r}, alpha={lora_alpha}, modules={final_targets}, steps={max_steps}")

try:
    model = get_peft_model(model, lora_config)
    print("✅ Modèle LoRA configuré")
except Exception as e:
    print(f"❌ Erreur configuration LoRA: {e}")
    exit(1)

# === 5. Entraînement (paramètres sécurisés) ===
print("Configuration du Trainer...")

# Paramètres adaptatifs - utilisation de max_steps défini plus haut
print(f"Configuration training: {max_steps} étapes avec {num_workers} workers")

# Configuration des DataLoaders pour la parallélisation
dataloader_config = {
    "num_workers": num_workers,
    "persistent_workers": True if num_workers > 0 else False,
    "prefetch_factor": 2 if num_workers > 0 else None,
    "pin_memory": False,  # Désactivé pour CPU
}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,
    max_steps=max_steps,
    logging_steps=10,
    save_steps=50,
    fp16=(torch_dtype == torch.float16),  # FP16 seulement si supporté
    save_total_limit=2,
    dataloader_pin_memory=dataloader_config["pin_memory"],
    dataloader_num_workers=dataloader_config["num_workers"],
    remove_unused_columns=False,
    report_to=None,  # Désactiver wandb/tensorboard
    logging_dir=None,
    # Optimisations supplémentaires
    group_by_length=True,  # Grouper par longueur pour efficacité
    length_column_name="length",
    disable_tqdm=False,  # Garder la barre de progression
)

print(f"⚙️  Paramètres d'entraînement:")
print(f"  - Batch size: {batch_size}")
print(f"  - Gradient accumulation: {grad_accum}")
print(f"  - Max steps: {max_steps}")
print(f"  - FP16: {training_args.fp16}")
print(f"  - Device: {device_map}")
print(f"  - DataLoader workers: {num_workers}")
print(f"  - PyTorch threads: {optimal_threads}")

# Fonction de monitoring des performances
def monitor_performance():
    import time
    import threading
    
    def monitor():
        while not monitor.stop_event.is_set():
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"📊 VRAM: {gpu_mem:.2f} GB", end=" | ")
            
            ram = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            print(f"RAM: {ram.percent:.1f}% | CPU: {cpu:.1f}%")
            
            time.sleep(10)  # Monitoring toutes les 10 secondes
    
    monitor.stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    
    return monitor

# Démarrer le monitoring
perf_monitor = monitor_performance()
print("📈 Monitoring des performances démarré...")

try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )
    
    print("🚀 Début de l'entraînement LoRA...")
    print("   (Si le processus plante, réduisez batch_size ou utilisez CPU)")
    
    # Monitoring mémoire pendant l'entraînement
    if torch.cuda.is_available():
        print(f"VRAM avant entraînement: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    trainer.train()
    
    print("✅ Entraînement terminé avec succès!")
    
except Exception as e:
    print(f"❌ Erreur pendant l'entraînement: {e}")
    print("💡 Suggestions:")
    print("   - Réduire batch_size à 1")
    print("   - Utiliser device_map='cpu'")
    print("   - Fermer d'autres applications pour libérer la RAM")
    
    # Nettoyer la mémoire
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    exit(1)

# === 6. Sauvegarde ===
try:
    print(f"💾 Sauvegarde de l'adaptateur LoRA dans : {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Sauvegarder les métadonnées de l'entraînement
    metadata = {
        "model_name": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": final_targets,  # Utiliser final_targets au lieu de attention_modules
        "max_steps": max_steps,
        "batch_size": batch_size,
        "examples_count": len(examples),
        "device": device_map,
        "dtype": str(torch_dtype),
        "resource_mode": resource_status
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Fine-tuning LoRA terminé avec succès !")
    print(f"📁 Fichiers sauvegardés dans: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde: {e}")

# Nettoyage final
perf_monitor.stop_event.set()  # Arrêter le monitoring
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("🧹 Cache GPU nettoyé")

print(f"🎯 Entraînement terminé avec {optimal_threads} threads sur {cpu_count} cœurs")
