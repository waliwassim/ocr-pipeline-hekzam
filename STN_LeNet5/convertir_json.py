"""
Convertisseur JSON — Projet Hekzam
====================================
Convertit la sortie de detection_coins.py vers le format attendu par pipeline_ocr.py.

Format entrée (detection_coins.py) :
    {
      "1": {"tl": [x, y], "tr": [...], "br": [...], "bl": [...], "type": "grid"},
      "2": {...}
    }

Format sortie (pipeline_ocr.py) :
    [
      {"id": "c1", "page": 1, "coins": {"tl": [...], "tr": [...], "bl": [...], "br": [...]}},
      {"id": "c2", "page": 1, "coins": {...}}
    ]

Usage :
    python convertir_json.py --input results/page_001.json --output cases.json
    python convertir_json.py --input results/page_003.json --output cases.json --page 3
"""

import json
import os
import argparse
import re


def extraire_numero_page(nom_fichier: str) -> int:
    """
    Extrait le numéro de page depuis le nom du fichier.
    Ex: page_001.json → 1,  page_03.json → 3
    Si non trouvé, retourne 1 par défaut.
    """
    nom = os.path.basename(nom_fichier)
    match = re.search(r'(\d+)', nom)
    return int(match.group(1)) if match else 1


def convertir(chemin_input: str, chemin_output: str, page: int = None):
    """
    Convertit le JSON de detection_coins.py vers le format de pipeline_ocr.py.
    """
    with open(chemin_input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Numéro de page : priorité à l'argument CLI, sinon extrait du nom de fichier
    numero_page = page if page is not None else extraire_numero_page(chemin_input)

    liste = []
    for key, val in data.items():
        liste.append({
            "id": f"c{key}",
            "page": numero_page,
            "coins": {
                "tl": val.get("tl"),
                "tr": val.get("tr"),
                "bl": val.get("bl"),
                "br": val.get("br")
            }
        })

    with open(chemin_output, 'w', encoding='utf-8') as f:
        json.dump(liste, f, indent=2, ensure_ascii=False)

    print(f"✓ Converti : {len(liste)} cases  (page {numero_page})  → {chemin_output}")
    return liste


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertit le JSON de detection_coins.py vers le format pipeline_ocr.py"
    )
    parser.add_argument("--input",  required=True, help="JSON produit par detection_coins.py")
    parser.add_argument("--output", required=True, help="JSON converti pour pipeline_ocr.py")
    parser.add_argument("--page",   type=int, default=None,
                        help="Numéro de page (extrait automatiquement du nom de fichier si absent)")
    args = parser.parse_args()

    convertir(args.input, args.output, args.page)
