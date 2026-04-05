# transform.py
# Calcul de l'homographie et warp vers l'espace canonique.
# Les coordonnées JSON sont en mm → converties en px selon le DPI.

import cv2
import numpy as np
from json_loader import mm_to_px


def compute_global_transform(
    detected_markers: list[dict],
    template_markers: list[dict],
    dpi: int
) -> np.ndarray:
    """
    Calcule la matrice d'homographie entre :
      - les positions détectées des marqueurs dans l'image scannée (en pixels)
      - les positions théoriques du JSON (en mm → converties en pixels via DPI)

    L'homographie encode rotation + translation + échelle + légère perspective.

    Args:
        detected_markers : 4 dicts {"corner": str, "center": (cx_px, cy_px)}
        template_markers : 4 dicts du JSON avec x, y, width, height en mm + "corner"
        dpi              : résolution du scan

    Returns:
        H : matrice 3×3 float64 (homographie)
    """
    # Construire les tableaux de points appariés par coin
    src_list = []   # positions dans l'image scannée (pixels)
    dst_list = []   # positions théoriques (pixels depuis mm)

    # Index des marqueurs template par coin
    tmpl_by_corner = {m["corner"]: m for m in template_markers}
    det_by_corner  = {m["corner"]: m for m in detected_markers}

    for corner in ("tl", "tr", "br", "bl"):
        if corner not in tmpl_by_corner or corner not in det_by_corner:
            raise ValueError(f"Coin '{corner}' manquant dans les marqueurs.")

        t = tmpl_by_corner[corner]
        d = det_by_corner[corner]

        # Centre théorique en pixels (depuis mm)
        dst_cx = mm_to_px(t["x"] + t["width"]  / 2, dpi)
        dst_cy = mm_to_px(t["y"] + t["height"] / 2, dpi)

        src_list.append(d["center"])
        dst_list.append((dst_cx, dst_cy))

    src_pts = np.array(src_list, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(dst_list, dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

    if H is None:
        raise RuntimeError("Calcul d'homographie échoué.")

    inliers = int(np.sum(mask)) if mask is not None else 0
    print(f"[Transform] Homographie calculée — inliers RANSAC : {inliers}/4")

    return H


def warp_to_canonical(
    image: np.ndarray,
    H: np.ndarray,
    page_size_mm: tuple[float, float],
    dpi: int
) -> np.ndarray:
    """
    Applique l'homographie pour recaler l'image scannée dans l'espace canonique.

    La taille de sortie est calculée depuis les dimensions réelles de la page
    (en mm) et le DPI, pour correspondre exactement aux coordonnées du JSON.

    Args:
        image        : image scannée BGR
        H            : matrice d'homographie 3×3
        page_size_mm : (largeur_mm, hauteur_mm) du formulaire théorique
                       ex : (210.0, 297.0) pour A4
        dpi          : résolution du scan

    Returns:
        warped : image recalée (même résolution que page_size_mm × dpi)
    """
    out_w = int(mm_to_px(page_size_mm[0], dpi))
    out_h = int(mm_to_px(page_size_mm[1], dpi))

    warped = cv2.warpPerspective(
        image, H, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    print(f"[Transform] Warp OK → {out_w}×{out_h} px  ({page_size_mm[0]}×{page_size_mm[1]} mm @ {dpi} dpi)")
    return warped


def estimate_page_size_mm(template_markers: list[dict]) -> tuple[float, float]:
    """
    Estime la taille de la page en mm depuis les positions des marqueurs du JSON.

    Utilise le coin bas-droit du marqueur br comme coin bas-droit de la page,
    et ajoute la même marge que le côté tl (symétrie supposée).

    Args:
        template_markers : liste des 4 marqueurs du JSON (avec x, y, width, height en mm)

    Returns:
        (largeur_mm, hauteur_mm)
    """
    by_corner = {m["corner"]: m for m in template_markers}

    tl = by_corner.get("tl")
    br = by_corner.get("br")

    if tl is None or br is None:
        # Fallback : A4
        print("[Transform] Impossible d'estimer la taille de page → fallback A4 (210×297mm)")
        return (210.0, 297.0)

    # Marge gauche/haut = position du coin tl
    margin_x = tl["x"]
    margin_y = tl["y"]

    # Coin bas-droit du marqueur br
    br_right  = br["x"] + br["width"]
    br_bottom = br["y"] + br["height"]

    # Page = jusqu'au bord br + même marge que le côté tl
    width_mm  = br_right  + margin_x
    height_mm = br_bottom + margin_y

    print(f"[Transform] Taille de page estimée : {width_mm:.1f}×{height_mm:.1f} mm")
    return (width_mm, height_mm)
