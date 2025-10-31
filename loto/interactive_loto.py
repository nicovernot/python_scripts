#!/usr/bin/env python3
"""
Interface interactive pour le modèle LoRA Loto
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

class LoraLotoPredictor:
    """
    Classe pour gérer les prédictions avec le modèle LoRA
    """
    
    def __init__(self, model_path="./phi3_lora_loto"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load_model(self):
        """
        Charge le modèle et le tokenizer
        """
        try:
            print("🔄 Chargement du modèle...")
            
            # Vérifier l'existence du modèle
            if not os.path.exists(self.model_path):
                print(f"❌ Modèle non trouvé : {self.model_path}")
                return False
            
            # Charger le modèle de base
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            # Charger l'adaptateur LoRA
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.loaded = True
            print("✅ Modèle chargé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")
            return False
    
    def predict(self, prompt, max_tokens=100, temperature=0.8):
        """
        Génère une prédiction à partir d'un prompt
        """
        if not self.loaded:
            print("❌ Modèle non chargé")
            return None
        
        try:
            # Tokeniser
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Générer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Décoder
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction : {e}")
            return None

def interactive_mode():
    """
    Mode interactif pour tester le modèle
    """
    print("🎲 === INTERFACE INTERACTIVE LOTO IA === 🎲")
    print("=" * 50)
    
    # Initialiser le prédicteur
    predictor = LoraLotoPredictor("/home/nico/projets/python_scripts/loto/phi3_lora_loto")
    
    if not predictor.load_model():
        print("💡 Conseil : Entraînez d'abord le modèle avec 'python train_lora.py'")
        return
    
    print("\n📖 Commandes disponibles :")
    print("  - Tapez votre prompt de prédiction")
    print("  - 'exemples' pour voir des exemples de prompts")
    print("  - 'quit' ou 'exit' pour quitter")
    print("  - 'help' pour afficher cette aide")
    print("=" * 50)
    
    examples = [
        "Analyse ces tirages récents : [1,12,23,34,45], [2,13,24,35,46], [3,14,25,36,47]",
        "Prédit les 5 numéros pour le prochain tirage du loto",
        "Basé sur l'historique, quels numéros ont le plus de chances ?",
        "Donne-moi une combinaison gagnante pour demain",
        "Analyse la fréquence et prédit : [5,15,25,35,45], [6,16,26,36,46]"
    ]
    
    while True:
        try:
            print("\n🎯 Entrez votre prompt (ou 'help' pour l'aide) :")
            user_input = input("> ").strip()
            
            if not user_input:
                continue
            
            # Commandes spéciales
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir !")
                break
            
            elif user_input.lower() in ['help', 'aide']:
                print("\n📖 Aide :")
                print("  - Entrez un prompt décrivant ce que vous voulez prédire")
                print("  - Exemple : 'Analyse ces tirages et prédit le suivant : [1,2,3,4,5]'")
                print("  - Le modèle va générer une réponse basée sur son entraînement")
                continue
            
            elif user_input.lower() == 'exemples':
                print("\n💡 Exemples de prompts :")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example}")
                continue
            
            # Génération de prédiction
            print(f"\n🤖 Génération en cours...")
            prediction = predictor.predict(user_input)
            
            if prediction:
                print(f"\n🎲 Prédiction :")
                print(f"📝 {prediction}")
            else:
                print("❌ Erreur lors de la génération")
                
        except KeyboardInterrupt:
            print("\n\n👋 Arrêt par l'utilisateur")
            break
        
        except Exception as e:
            print(f"❌ Erreur : {e}")

def quick_test():
    """
    Test rapide avec des exemples prédéfinis
    """
    print("🧪 Test rapide du modèle")
    print("=" * 30)
    
    predictor = LoraLotoPredictor("/home/nico/projets/python_scripts/loto/phi3_lora_loto")
    
    if not predictor.load_model():
        return
    
    test_prompt = "Analyse ces 3 derniers tirages et prédit le prochain : [7,14,21,28,35], [3,9,15,27,42], [5,11,22,33,44]"
    
    print(f"\n🎯 Prompt de test :")
    print(f"📝 {test_prompt}")
    
    prediction = predictor.predict(test_prompt)
    
    if prediction:
        print(f"\n🎲 Résultat :")
        print(f"📝 {prediction}")
    else:
        print("❌ Échec du test")

def main():
    """
    Fonction principale avec menu
    """
    print("🎲 === TESTEUR MODÈLE LORA LOTO === 🎲")
    print("=" * 40)
    print("1. Mode interactif")
    print("2. Test rapide")
    print("3. Quitter")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\nChoisissez une option (1-3) : ").strip()
            
            if choice == '1':
                interactive_mode()
                break
            elif choice == '2':
                quick_test()
                break
            elif choice == '3':
                print("👋 Au revoir !")
                break
            else:
                print("❌ Option invalide. Choisissez 1, 2 ou 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break

if __name__ == "__main__":
    main()