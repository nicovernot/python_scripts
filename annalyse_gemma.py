import pandas as pd
import ollama

# --- Configuration ---
MODEL_ID = "gemma2:9b"
CSV_FILE_PATH = "/home/nico/projets/python_scripts/loto/loto_data/loto_201911.csv"
USER_QUESTION = "En te basant sur les statistiques fournies, quelles sont les tendances générales des tirages ? Présente les numéros chauds, les numéros froids et l'analyse des fréquences dans un rapport clair."

def analyze_loto_data_with_ollama(file_path: str, question: str):
    """
    Pré-calcule les statistiques du Loto et demande à Ollama de les interpréter.
    """
    print(f"\nAnalyse du fichier '{file_path}'...")

    try:
        df = pd.read_csv(file_path, sep=';')  # Utiliser point-virgule comme séparateur
        
        # --- Étape 1 : Pré-traitement des données avec Pandas ---
        print("Calcul des statistiques (fréquences, numéros chauds/froids)...")

        # Adapter cette liste si les noms de vos colonnes sont différents
        ball_columns = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']
        
        # Vérifier que les colonnes existent
        if not all(col in df.columns for col in ball_columns):
            print(f"Colonnes disponibles : {list(df.columns)}")
            return f"Erreur : Le fichier CSV doit contenir les colonnes suivantes : {', '.join(ball_columns)}"

        # Rassembler tous les numéros tirés dans une seule liste
        all_numbers = pd.concat([df[col] for col in ball_columns])
        
        # Calculer la fréquence de chaque numéro
        frequency = all_numbers.value_counts().sort_index()
        
        # Identifier les numéros les plus et les moins fréquents
        hot_numbers = frequency.nlargest(5)
        cold_numbers = frequency.nsmallest(5)
        
        total_draws = len(df)

    except FileNotFoundError:
        return f"Erreur : Le fichier '{file_path}' n'a pas été trouvé."
    except Exception as e:
        return f"Erreur lors du traitement du fichier CSV : {e}"

    # --- Étape 2 : Construire un prompt avec les statistiques calculées ---
    prompt = f"""
    **Rôle :** Tu es un expert en analyse statistique spécialisé dans les jeux de loterie.

    **Tâche :** Rédige un rapport d'analyse clair et concis en te basant exclusivement sur les statistiques suivantes, qui ont été calculées à partir d'un fichier de résultats de tirages. Ne mentionne pas que les données sont potentiellement incomplètes, analyse simplement ce qui est fourni.

    **Statistiques Pré-calculées :**
    - Nombre total de tirages analysés : {total_draws}
    
    - Top 5 des numéros les plus fréquents (Numéros Chauds) :
    {hot_numbers.to_string()}

    - Top 5 des numéros les moins fréquents (Numéros Froids) :
    {cold_numbers.to_string()}

    - Fréquence complète de chaque numéro (Numéro -> Nombre de sorties) :
    {frequency.to_string()}

    **Rapport demandé par l'utilisateur :**
    {question}

    **Rapport d'Analyse :**
    """

    # --- Étape 3 : Envoyer la requête à Ollama ---
    print("Envoi des statistiques à l'agent local Ollama pour interprétation...")
    try:
        response = ollama.chat(
            model=MODEL_ID,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
        
    except Exception as e:
        return f"Erreur de communication avec l'API d'Ollama. Assurez-vous qu'Ollama est bien en cours d'exécution. Détails : {e}"

if __name__ == "__main__":
    # Lance l'analyse
    analysis_result = analyze_loto_data_with_ollama(CSV_FILE_PATH, USER_QUESTION)
    
    # Affiche le résultat
    print("\n--- Début de l'analyse de l'agent ---")
    print(analysis_result)
    print("--- Fin de l'analyse ---")