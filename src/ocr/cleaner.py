import cv2
import os
from pathlib import Path

def clean_one_image_safe(image_path, save_path, debug_path):
    """Nettoie l'image SANS rotation pour éviter les coupures."""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    # 1. Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Binarisation (Otsu) - Rend le texte bien noir et le fond bien blanc
    # C'est l'étape la plus utile pour Tesseract et LayoutLM
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Sauvegarde (Même dimensions que l'original)
    cv2.imwrite(str(save_path), binary)
    
    # 4. Debug (Optionnel)
    # Pour ta comparaison, on crée juste un montage simple
    h, w = img.shape[:2]
    target_w = 500
    ratio = target_w / w
    res_orig = cv2.resize(img, (target_w, int(h * ratio)))
    res_bin = cv2.resize(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), (target_w, int(h * ratio)))
    comp = cv2.hconcat([res_orig, res_bin])
    cv2.imwrite(str(debug_path), comp)

def process_all_images(input_folder, output_folder):
    input_p = Path(input_folder)
    output_p = Path(output_folder)
    debug_p = output_p / "comparaisons"

    output_p.mkdir(parents=True, exist_ok=True)
    debug_p.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Nettoyage sécurisé de {len(files)} images...")

    for filename in files:
        clean_one_image_safe(input_p / filename, output_p / filename, debug_p / f"check_{filename}")
        print(f"OK : {filename}")

if __name__ == "__main__":
    folders = [
        ("data/raw/archive/SROIE2019/train/img", "data/processed/train/img"),
        ("data/raw/archive/SROIE2019/test/img", "data/processed/test/img")
    ]
    for in_dir, out_dir in folders:
        process_all_images(in_dir, out_dir)
    print("\n--- Nettoyage terminé sans aucune perte de données ! ---")