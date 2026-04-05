# pdf_converter.py
# Conversion d'un PDF multi-pages en images (une image par page).
#
# Dépendance : pdf2image  →  pip install pdf2image
# Système    : poppler   →  sudo apt install poppler-utils  (Linux)
#                         →  brew install poppler            (Mac)
#                         →  https://github.com/oschwartz10612/poppler-windows (Windows)

from pathlib import Path
import numpy as np
import cv2


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[np.ndarray]:
    """
    Convertit un PDF multi-pages en liste d'images OpenCV (BGR).

    Chaque image correspond à une page du PDF, dans l'ordre.
    La résolution DPI doit correspondre à celle utilisée pour les coordonnées
    dans atomic-boxes.json (qui sont en mm → converties via mm_to_px(mm, dpi)).

    Args:
        pdf_path : chemin vers le fichier PDF scanné
        dpi      : résolution de conversion (défaut : 300 dpi)
                   → plus élevé = meilleure qualité mais plus lent et lourd

    Returns:
        liste d'images BGR numpy, une par page
        images[0] = page 1, images[1] = page 2, etc.

    Raises:
        FileNotFoundError si le PDF n'existe pas
        ImportError si pdf2image n'est pas installé
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image requis : pip install pdf2image\n"
            "Et poppler : sudo apt install poppler-utils (Linux) "
            "ou brew install poppler (Mac)"
        )

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    print(f"[PDF] Conversion de : {path.name}  (DPI={dpi})")

    # convert_from_path retourne une liste de PIL Images
    pil_images = convert_from_path(str(path), dpi=dpi)

    print(f"[PDF] {len(pil_images)} page(s) détectée(s)")

    # Conversion PIL → OpenCV (BGR numpy array)
    cv_images = []
    for i, pil_img in enumerate(pil_images):
        # PIL est en RGB, OpenCV attend BGR
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv_images.append(bgr)
        print(f"  Page {i+1} : {bgr.shape[1]}×{bgr.shape[0]} px")

    return cv_images


def save_page_image(image: np.ndarray, output_dir: str, page_num: int) -> str:
    """
    Sauvegarde l'image d'une page dans un dossier temporaire.
    Utile pour le débogage ou la reprise sur erreur.

    Args:
        image      : image BGR de la page
        output_dir : dossier de sortie
        page_num   : numéro de page (commence à 1)

    Returns:
        chemin du fichier sauvegardé
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"page_{page_num:03d}.png"
    cv2.imwrite(str(path), image)
    print(f"[PDF] Page {page_num} sauvegardée → {path}")
    return str(path)
