# extractor.py
# Extraction des cases depuis l'image canonique + preprocessing 28×28.

import cv2
import numpy as np

# Taille cible pour le CNN / HOG+SVM
DIGIT_SIZE       = 28
MIN_CONTENT_AREA = 10
DEFAULT_MARGIN   = 0.15


def extract_box_with_margin(
    warped_image: np.ndarray,
    box_px: dict,
    margin_ratio: float = DEFAULT_MARGIN
) -> np.ndarray | None:
    """
    Extrait une case depuis l'image recalée en ajoutant une marge de tolérance.

    Les coordonnées de box_px doivent déjà être en PIXELS
    (après conversion mm→px via json_loader.convert_box_to_px).

    Args:
        warped_image : image recalée dans l'espace canonique (BGR)
        box_px       : dict avec x, y, width, height en pixels
        margin_ratio : fraction de marge autour de la case (0.15 = 15%)

    Returns:
        crop (numpy array) ou None si hors limites
    """
    h_img, w_img = warped_image.shape[:2]

    x  = int(box_px["x"])
    y  = int(box_px["y"])
    bw = int(box_px["width"])
    bh = int(box_px["height"])

    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w_img, x + bw + mx)
    y2 = min(h_img, y + bh + my)

    if x2 <= x1 or y2 <= y1:
        return None

    return warped_image[y1:y2, x1:x2]


def preprocess_digit_crop(crop: np.ndarray) -> np.ndarray | None:
    """
    Prépare un crop de case pour le classifieur (HOG+SVM ou CNN) :
      1. Niveaux de gris
      2. Débruitage médian
      3. Binarisation Otsu
      4. Inversion si fond noir
      5. Détection bounding box du contenu manuscrit
      6. Recentrage dans une image 28×28 normalisée [0,1]

    Returns:
        image float32 (28, 28) normalisée, ou None si crop invalide
    """
    if crop is None or crop.size == 0:
        return None

    # Niveaux de gris
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # Débruitage
    denoised = cv2.medianBlur(gray, 3)

    # Binarisation Otsu
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fond = blanc (255), chiffre = noir (0)
    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)

    # Trouver le contenu manuscrit (pixels noirs sur fond blanc)
    content = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(content, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > MIN_CONTENT_AREA]

    if not valid:
        # Case vide → image noire
        return np.zeros((DIGIT_SIZE, DIGIT_SIZE), dtype=np.float32)

    all_pts = np.vstack(valid)
    rx, ry, rw, rh = cv2.boundingRect(all_pts)
    content_crop = content[ry:ry+rh, rx:rx+rw]

    return _center_in_canvas(content_crop, DIGIT_SIZE)


def _center_in_canvas(content: np.ndarray, size: int) -> np.ndarray:
    """Redimensionne et centre le contenu dans un canvas size×size."""
    h, w = content.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.float32)

    target = size - 8
    ratio  = target / max(h, w)
    new_w  = max(1, int(w * ratio))
    new_h  = max(1, int(h * ratio))

    resized  = cv2.resize(content, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas   = np.zeros((size, size), dtype=np.uint8)
    off_x    = (size - new_w) // 2
    off_y    = (size - new_h) // 2
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized

    return canvas.astype(np.float32) / 255.0
