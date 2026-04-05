# json_loader.py
# Chargement et parsing du fichier atomic-boxes.json.
#
# Format réel du JSON :
#   - dict à plat (pas de clé "pages")
#   - chaque entrée a un champ "page" (int, commence à 1)
#   - marqueurs QR : clés contenant "marker barcode" + "tl/tr/bl/br"
#   - cases normales : clés format "id-X-Y-LABEL"
#   - coordonnées en MILLIMÈTRES (x, y, width, height)

import json
from pathlib import Path


def load_atomic_boxes(json_path: str) -> dict:
    """
    Charge le fichier atomic-boxes.json.

    Args:
        json_path : chemin vers le fichier JSON

    Returns:
        dict brut du JSON
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier JSON introuvable : {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[JSON] Chargé : {json_path}  ({len(data)} entrées au total)")
    return data


def get_page_template(data: dict, page_num: int) -> tuple[list, list]:
    """
    Extrait pour une page donnée les marqueurs QR et les cases normales.

    Règles de détection :
      - Marqueur QR  : la clé contient "marker barcode"
      - Case normale : la clé commence par "id-"

    Les coordonnées restent en millimètres — la conversion en pixels
    se fait dans transform.py selon le DPI du scan.

    Args:
        data     : dict brut du JSON
        page_num : numéro de page (commence à 1, comme dans le JSON)

    Returns:
        (markers, boxes) :
          markers : liste de dicts, chacun avec les champs du JSON + "id" + "corner"
          boxes   : liste de dicts, chacun avec les champs du JSON + "id"
    """
    markers = []
    boxes   = []

    for key, elem in data.items():
        # Filtrer par numéro de page
        if elem.get("page") != page_num:
            continue

        entry = dict(elem)   # copie pour ne pas modifier l'original
        entry["id"] = key    # on injecte la clé comme champ "id"

        key_lower = key.lower()

        if "marker barcode" in key_lower:
            # Déterminer le coin : tl / tr / bl / br
            for corner in ("tl", "tr", "bl", "br"):
                if corner in key_lower:
                    entry["corner"] = corner
                    break
            else:
                entry["corner"] = "unknown"
            markers.append(entry)

        elif key_lower.startswith("id-"):
            boxes.append(entry)

    # Trier les marqueurs dans l'ordre canonique : tl, tr, br, bl
    corner_order = {"tl": 0, "tr": 1, "br": 2, "bl": 3, "unknown": 4}
    markers.sort(key=lambda m: corner_order.get(m.get("corner", "unknown"), 4))

    print(f"[JSON] Page {page_num} → {len(markers)} marqueurs, {len(boxes)} cases")

    if len(markers) != 4:
        print(f"[AVERTISSEMENT] {len(markers)} marqueur(s) trouvé(s) au lieu de 4.")

    return markers, boxes


def get_all_page_numbers(data: dict) -> list[int]:
    """
    Retourne la liste triée de tous les numéros de pages présents dans le JSON.

    Exemple : [1, 2, 3]
    """
    pages = sorted(set(elem["page"] for elem in data.values() if "page" in elem))
    print(f"[JSON] Pages trouvées : {pages}")
    return pages


def get_label_from_id(element_id: str) -> int | None:
    """
    Extrait le label depuis l'ID d'une case.

    Convention : "id-X-Y-LABEL" → le dernier segment est le label.
    Exemples :
        "id-0-0-0"  → 0
        "id-1-2-4"  → 4

    Returns:
        label (int) ou None si non parsable
    """
    parts = element_id.split("-")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None


def mm_to_px(mm: float, dpi: int) -> float:
    """
    Convertit des millimètres en pixels selon le DPI du scan.

    Formule : px = mm * (dpi / 25.4)

    Args:
        mm  : valeur en millimètres
        dpi : résolution du scan (typiquement 150, 200, 300)

    Returns:
        valeur en pixels (float)
    """
    return mm * (dpi / 25.4)


def convert_box_to_px(box: dict, dpi: int) -> dict:
    """
    Retourne une copie du dict de case avec les coordonnées converties en pixels.

    Args:
        box : dict avec x, y, width, height en mm
        dpi : résolution du scan

    Returns:
        dict avec x, y, width, height en pixels (float)
    """
    converted = dict(box)
    for field in ("x", "y", "width", "height"):
        if field in converted:
            converted[field] = mm_to_px(converted[field], dpi)
    return converted
