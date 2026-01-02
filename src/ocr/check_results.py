import cv2
import os
import random
import numpy as np

def compare_random_result(raw_dir, processed_dir):
    """Choisit une image au hasard et affiche le Avant/Après proprement."""
    
    # 1. Lister les fichiers dans le dossier brut (raw)
    if not os.path.exists(raw_dir):
        print(f"Erreur : Le dossier {raw_dir} n'existe pas.")
        return
        
    files = [f for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not files:
        print("Aucune image trouvée dans le dossier raw.")
        return
        
    random_file = random.choice(files)
    
    # 2. Charger les deux images
    path_raw = os.path.join(raw_dir, random_file)
    path_processed = os.path.join(processed_dir, random_file)
    
    original = cv2.imread(path_raw)
    cleaned = cv2.imread(path_processed)
    
    if cleaned is None:
        print(f"L'image nettoyée pour {random_file} n'a pas été trouvée dans {processed_dir}.")
        print("Avez-vous bien lancé le script cleaner.py avant ?")
        return

    # 3. REDIMENSIONNEMENT INTELLIGENT (La modif)
    # On définit une largeur cible pour chaque image (500 pixels)
    target_width = 500
    h, w = original.shape[:2]
    ratio = target_width / float(w)
    target_height = int(h * ratio)

    # On redimensionne les deux images à la même taille pour la comparaison
    original_res = cv2.resize(original, (target_width, target_height))
    cleaned_res = cv2.resize(cleaned, (target_width, target_height))

    # 4. Création de la vue côte à côte
    # On accole les deux images horizontalement
    comparison = np.hstack((original_res, cleaned_res))
    
    # 5. Affichage
    window_name = f"Comparaison SROIE - Gauche: Original | Droite: Nettoyé ({random_file})"
    cv2.imshow(window_name, comparison)
    
    print(f"--- Affichage de : {random_file} ---")
    print("Instructions :")
    print("- Appuyez sur n'importe quelle touche pour FERMER la fenêtre.")
    print("- Ne cliquez pas sur la loupe de la visionneuse pour éviter le bug du zoom.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # On pointe vers le dossier train par défaut
    compare_random_result("data/raw/archive/SROIE2019/train/img", "data/processed/train/img")