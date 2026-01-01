import cv2
import numpy as np
import os
from pathlib import Path

def deskew(image):
    """Redresse l'image si elle est penchée."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    # Si on n'a pas de coords (image vide/unie), on retourne l'image telle quelle
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    
    # Ajustement de l'angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def clean_one_image(image_path, save_path):
    """Nettoie une seule image (Niveaux de gris + Redressement + Seuillage)."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] impossible de lire l'image: {image_path}")
        return
    
    # 1. Redressement
    img = deskew(img)
    
    # 2. Niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Binarisation (Seuillage d'Otsu pour un contraste optimal)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    ok = cv2.imwrite(str(save_path), binary)
    if not ok:
        print(f"[ERROR] impossible d'écrire l'image nettoyée: {save_path}")

def process_all_images(input_folder, output_folder):
    """Parcourt le dossier et nettoie tout en respectant la structure."""
    input_p = Path(input_folder)
    output_p = Path(output_folder)

    # Vérifier que le dossier d'entrée existe
    if not input_p.exists():
        print(f"[ERROR] Le dossier d'entrée n'existe pas: {input_folder}")
        return

    # Créer le dossier de sortie s'il n'existe pas
    output_p.mkdir(parents=True, exist_ok=True)

    try:
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    except Exception as e:
        print(f"[ERROR] Impossible de lister le dossier {input_folder}: {e}")
        return

    if len(files) == 0:
        print(f"--- Aucun fichier image trouvé dans {input_folder} ---")
        return

    print(f"--- Début du nettoyage de {len(files)} images dans : {input_folder} ---")

    for filename in files:
        input_path = input_p / filename
        output_path = output_p / filename
        clean_one_image(input_path, output_path)
        print(f"Nettoyé : {filename}")

if __name__ == "__main__":
    # Liste des dossiers à nettoyer
    folders_to_process = [
        ("data/raw/archive/SROIE2019/train/img", "data/processed/train/img"),
        ("data/raw/archive/SROIE2019/test/img", "data/processed/test/img")
    ]
    
    for input_dir, output_dir in folders_to_process:
        print(f"\n--- Dossier actuel : {input_dir} ---")
        process_all_images(input_dir, output_dir)
        
    print("\n--- Tout le dataset (Train et Test) est propre ! ---")