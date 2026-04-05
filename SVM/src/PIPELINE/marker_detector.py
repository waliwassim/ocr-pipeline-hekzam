# marker_detector.py
# Détection des 4 marqueurs QR/barcode dans l'image scannée.
#
# Stratégie (3 niveaux de robustesse) :
#   1. cv2.QRCodeDetector  → détecteur natif OpenCV
#   2. pyzbar              → bibliothèque dédiée QR/barcode (plus robuste)
#   3. Fallback coins      → cherche le plus grand carré sombre dans chaque zone de coin

import cv2
import numpy as np


def detect_qr_markers(image: np.ndarray) -> list[dict] | None:
    """
    Détecte les 4 marqueurs QR/barcode dans l'image scannée.

    Args:
        image : image BGR (numpy array)

    Returns:
        Liste de 4 dicts triés [tl, tr, br, bl] :
            {"corner": "tl", "center": (cx_px, cy_px)}
        ou None si la détection échoue sur toutes les méthodes.
    """
    # --- Méthode 1 : QRCodeDetector OpenCV
    result = _detect_with_opencv_qr(image)
    if result:
        print("[Marqueurs] ✓ Détection via cv2.QRCodeDetector")
        return result

    # --- Méthode 2 : pyzbar
    result = _detect_with_pyzbar(image)
    if result:
        print("[Marqueurs] ✓ Détection via pyzbar")
        return result

    # --- Méthode 3 : fallback zones de coin
    print("[Marqueurs] QR non détecté → fallback zones de coin")
    result = _detect_with_corner_zones(image)
    if result:
        print("[Marqueurs] ✓ Détection via fallback coins")
        return result

    print("[ERREUR] Impossible de détecter les 4 marqueurs.")
    return None


# ---------------------------------------------------------------------------
# Méthode 1 : cv2.QRCodeDetector
# ---------------------------------------------------------------------------

def _detect_with_opencv_qr(image: np.ndarray) -> list[dict] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    detector = cv2.QRCodeDetector()
    try:
        retval, decoded_info, points, _ = detector.detectMulti(gray)
    except Exception:
        return None

    if not retval or points is None or len(points) < 4:
        return None

    centers = []
    for qr_pts in points[:4]:
        pts = qr_pts.reshape(-1, 2)
        centers.append((float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))))

    return _sort_to_corners(centers, image.shape)


# ---------------------------------------------------------------------------
# Méthode 2 : pyzbar
# ---------------------------------------------------------------------------

def _detect_with_pyzbar(image: np.ndarray) -> list[dict] | None:
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
    except ImportError:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    barcodes = pyzbar_decode(gray)

    if len(barcodes) < 4:
        return None

    centers = []
    for bc in barcodes[:4]:
        pts = np.array(bc.polygon, dtype=np.float32)
        centers.append((float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))))

    return _sort_to_corners(centers, image.shape)


# ---------------------------------------------------------------------------
# Méthode 3 : fallback — cherche le plus grand carré sombre dans chaque coin
# ---------------------------------------------------------------------------

def _detect_with_corner_zones(image: np.ndarray) -> list[dict] | None:
    """
    Cherche le plus grand carré sombre dans chaque zone de coin de l'image.

    Clé de robustesse : on cherche uniquement dans les 15% depuis chaque bord.
    Les QR codes sont toujours dans les coins → pas de confusion avec les cases
    du formulaire qui sont dans la zone centrale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape

    # Zone de recherche : 15% depuis chaque bord
    CORNER_RATIO = 0.15
    zw = int(w * CORNER_RATIO)
    zh = int(h * CORNER_RATIO)

    # (coin, x_debut, y_debut, x_fin, y_fin)
    corner_zones = [
        ("tl", 0,      0,      zw,   zh),
        ("tr", w - zw, 0,      w,    zh),
        ("br", w - zw, h - zh, w,    h),
        ("bl", 0,      h - zh, zw,   h),
    ]

    result = []

    for corner, x1, y1, x2, y2 in corner_zones:
        roi     = gray[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_area      = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Surface minimale : 0.5% de la zone de coin
            if area < (zw * zh) * 0.005:
                continue

            bx, by, bw_c, bh_c = cv2.boundingRect(cnt)
            ratio = bw_c / bh_c if bh_c > 0 else 0
            if not (0.4 < ratio < 2.5):
                continue

            if area > best_area:
                best_area = area
                # Remettre les coordonnées dans le repère de l'image complète
                cx = x1 + bx + bw_c / 2
                cy = y1 + by + bh_c / 2
                best_candidate = {"corner": corner, "center": (cx, cy), "area": area}

        if best_candidate is None:
            print(f"[Fallback] Aucun marqueur dans la zone {corner} ({x1},{y1})→({x2},{y2})")
            return None

        print(f"[Fallback] {corner.upper()} → ({best_candidate['center'][0]:.0f}, "
              f"{best_candidate['center'][1]:.0f})  aire={best_area:.0f}px²")
        result.append(best_candidate)

    return result


# ---------------------------------------------------------------------------
# Utilitaire : tri géométrique des centres en tl / tr / br / bl
# ---------------------------------------------------------------------------

def _sort_to_corners(centers: list[tuple], image_shape: tuple) -> list[dict]:
    pts        = np.array(centers, dtype=np.float32)
    sorted_y   = pts[np.argsort(pts[:, 1])]
    top_two    = sorted_y[:2]
    bottom_two = sorted_y[2:]

    tl = top_two[np.argmin(top_two[:, 0])]
    tr = top_two[np.argmax(top_two[:, 0])]
    bl = bottom_two[np.argmin(bottom_two[:, 0])]
    br = bottom_two[np.argmax(bottom_two[:, 0])]

    return [
        {"corner": "tl", "center": (float(tl[0]), float(tl[1]))},
        {"corner": "tr", "center": (float(tr[0]), float(tr[1]))},
        {"corner": "br", "center": (float(br[0]), float(br[1]))},
        {"corner": "bl", "center": (float(bl[0]), float(bl[1]))},
    ]
