# extract_true_crops.py
# Extrait les crops de TOUS les PDFs d'un dossier avec les VRAIS labels du JSON.
#
# Contrairement à run_pipeline.py qui sauvegarde les crops triés par prédiction,
# ce script sauvegarde les crops triés par VRAI label (depuis l'ID dans le JSON).
#
# Utilisation :
#   python3 extract_true_crops.py \
#       --input  ../scans \
#       --json   ../atomic-boxes.json \
#       --output ../true_crops

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from pdf_converter   import pdf_to_images
from json_loader     import load_atomic_boxes, get_page_template, \
                            get_all_page_numbers, get_label_from_id, convert_box_to_px
from marker_detector import detect_qr_markers
from transform       import compute_global_transform, warp_to_canonical, estimate_page_size_mm
from extractor       import extract_box_with_margin, preprocess_digit_crop


def extract_all_crops(
    input_dir:  str,
    json_path:  str,
    output_dir: str,
    dpi:        int   = 300,
    margin:     float = 0.15,
):
    """
    Parcourt tous les PDFs du dossier input_dir et extrait les crops
    avec les vrais labels depuis le JSON.

    Structure de sortie :
        output_dir/
            0/   ← tous les crops de chiffre 0 de tous les PDFs
            1/
            ...
            9/
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    # Créer les dossiers de classes
    for i in range(10):
        (output_path / str(i)).mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERREUR] Aucun PDF trouvé dans : {input_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  EXTRACTION CROPS — {len(pdf_files)} PDFs")
    print(f"{'='*60}\n")

    data       = load_atomic_boxes(json_path)
    page_nums  = get_all_page_numbers(data)

    total_saved  = 0
    total_failed = 0

    for pdf_idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{pdf_idx}/{len(pdf_files)}] {pdf_path.name}")

        try:
            page_images = pdf_to_images(str(pdf_path), dpi=dpi)
        except Exception as e:
            print(f"  [ERREUR] Conversion PDF : {e}")
            continue

        pdf_saved = 0

        for page_num in page_nums:
            img_idx = page_num - 1
            if img_idx >= len(page_images):
                continue

            image = page_images[img_idx]
            template_markers, boxes = get_page_template(data, page_num)

            if len(template_markers) != 4:
                continue

            # Détection marqueurs
            detected = detect_qr_markers(image)
            if detected is None:
                print(f"  [SKIP] Page {page_num} : marqueurs non détectés")
                continue

            # Homographie + warp
            try:
                page_size_mm = estimate_page_size_mm(template_markers)
                H      = compute_global_transform(detected, template_markers, dpi=dpi)
                warped = warp_to_canonical(image, H, page_size_mm=page_size_mm, dpi=dpi)
            except Exception as e:
                print(f"  [SKIP] Page {page_num} : {e}")
                continue

            # Extraction de chaque case
            for box in boxes:
                box_id = box["id"]

                # Récupérer le VRAI label depuis l'ID
                label = get_label_from_id(box_id)
                if label is None:
                    continue

                # Extraire et préprocesser
                box_px    = convert_box_to_px(box, dpi=dpi)
                crop      = extract_box_with_margin(warped, box_px, margin_ratio=margin)
                digit_img = preprocess_digit_crop(crop)

                if digit_img is None:
                    total_failed += 1
                    continue

                # Sauvegarder dans le bon dossier de classe (vrai label)
                img_uint8 = (digit_img * 255).astype(np.uint8)
                fname     = f"{pdf_path.stem}_p{page_num}_{box_id}.png"
                save_path = output_path / str(label) / fname
                cv2.imwrite(str(save_path), img_uint8)

                total_saved += 1
                pdf_saved   += 1

        print(f"  → {pdf_saved} crops extraits")

    # Résumé par classe
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ EXTRACTION")
    print(f"{'='*60}")
    print(f"  Total crops sauvegardés : {total_saved}")
    print(f"  Échecs                  : {total_failed}")
    print(f"\n  Par classe :")
    for i in range(10):
        count = len(list((output_path / str(i)).glob("*.png")))
        print(f"    {i}/ → {count} images")
    print(f"\n  Dossier : {output_path.resolve()}")
    print(f"{'='*60}\n")
    print(f"  Maintenant lance :")
    print(f"  python3 retrain.py --mnist ../dataset --crops {output_dir} "
          f"--model ../hog_svm/models/svm_allpdfs.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrait les crops de tous les PDFs avec les vrais labels du JSON"
    )
    parser.add_argument("--input",  required=True, help="Dossier contenant les PDFs")
    parser.add_argument("--json",   required=True, help="atomic-boxes.json")
    parser.add_argument("--output", required=True, help="Dossier de sortie des crops")
    parser.add_argument("--dpi",    type=int,   default=300)
    parser.add_argument("--margin", type=float, default=0.15)

    args = parser.parse_args()

    extract_all_crops(
        input_dir  = args.input,
        json_path  = args.json,
        output_dir = args.output,
        dpi        = args.dpi,
        margin     = args.margin,
    )
