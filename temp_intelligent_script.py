
import sys
sys.path.append('.')
from keno_intelligent_generator import KenoIntelligentGenerator

# Simulation du choix automatique
original_input = input
def mock_input(prompt):
    if "profil" in prompt.lower():
        return "2"
    return original_input(prompt)

__builtins__['input'] = mock_input

# Exécution
generator = KenoIntelligentGenerator()
if generator.load_and_analyze():
    generator.calculate_intelligent_top30()
    generator.analyze_optimal_parameters()
    generator.extract_top_pairs()
    grids, metadata = generator.generate_system_grids("moyen")
    files = generator.export_system_grids(grids, metadata)
    print("✅ Génération terminée avec succès!")
else:
    print("❌ Échec de la génération")
