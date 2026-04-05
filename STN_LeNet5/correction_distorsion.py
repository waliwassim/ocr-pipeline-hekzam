"""
Correction de distorsion — Projet Hekzam
=========================================
Auteur : [Collègue A]

Rôle dans la chaîne :
    [CE SCRIPT] ──PNG corrigés──► [Collègue détection coins] ──JSON──► [Pipeline OCR]

Ce script prend le PDF scanné en entrée et produit des images PNG corrigées
(grille supprimée, distorsion redressée) prêtes pour la détection des coins.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Étapes de traitement par page :
  1. Binarisation adaptative (seuillage de Gauss)
  2. Détection et suppression de la grille (Hough Lines)
  3. Redressement de la distorsion (Homographie via QR codes)
  4. Sauvegarde de l'image corrigée en PNG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage :
    python correction_distorsion.py --pdf scan.pdf --json atomic-boxes.json
    python correction_distorsion.py --pdf scan.pdf --json atomic-boxes.json --debug
"""

import os
import json
import argparse
import cv2
import numpy as np

try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

PDF_DPI        = 200
OUTPUT_DIR     = "./scans_corriges"
DEBUG_DIR      = "./debug_correction"

# Poppler pour Windows — modifie ce chemin si nécessaire
POPPLER_PATH_WINDOWS = r"C:\Users\hp\Downloads\Release-24.02.0-0\poppler\Library\bin"


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — BINARISATION ADAPTATIVE
# ─────────────────────────────────────────────────────────────────────────────

def binariser(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convertit l'image en niveaux de gris puis applique un seuillage
    adaptatif de Gauss (chiffres blancs sur fond noir).
    Le seuillage adaptatif est préféré à Otsu ici car il gère mieux
    les variations locales de luminosité sur un scan A4 entier.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binaire = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 1
    )
    return binaire


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — SUPPRESSION DE LA GRILLE (HOUGH LINES)
# ─────────────────────────────────────────────────────────────────────────────

def supprimer_grille(binaire: np.ndarray, debug: bool = False,
                     prefixe_debug: str = "") -> tuple:
    """
    Détecte et supprime les lignes horizontales et verticales de la grille
    imprimée du formulaire.

    Stratégie :
      A. Isoler les lignes par morphologie (kernels directionnels longs)
      B. Affiner avec Hough probabiliste pour ne garder que les vraies lignes
      C. Fusionner H + V → masque de grille
      D. Soustraire le masque de la binarisation → image propre

    Retourne :
        (image_propre_inv, masque_grille)
        image_propre_inv : image binarisée SANS la grille (chiffres blancs / fond noir)
    """
    h, w = binaire.shape

    # Marges de sécurité : on ignore les coins (QR codes) pour ne pas
    # les confondre avec la grille
    safe_margin = int(w * 0.09)
    binary_for_grid = binaire.copy()
    binary_for_grid[0:safe_margin,        0:safe_margin       ] = 0
    binary_for_grid[0:safe_margin,        w - safe_margin:w   ] = 0
    binary_for_grid[h - safe_margin:h,    w - safe_margin:w   ] = 0
    binary_for_grid[h - safe_margin:h,    0:safe_margin       ] = 0

    # ── A. Morphologie directionnelle ──
    # Kernel horizontal long (54px) → ne laisse passer que les traits horizontaux
    h_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (54, 1))
    h_mask_brut = cv2.morphologyEx(binary_for_grid, cv2.MORPH_OPEN, h_kernel)

    # Kernel vertical long → ne laisse passer que les traits verticaux
    v_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 54))
    v_mask_brut = cv2.morphologyEx(binary_for_grid, cv2.MORPH_OPEN, v_kernel)

    # ── B. Hough probabiliste pour affiner ──
    h_mask_propre = np.zeros_like(binaire)
    v_mask_propre = np.zeros_like(binaire)

    lines_h = cv2.HoughLinesP(
        h_mask_brut, 1, np.pi / 180,
        threshold=40, minLineLength=80, maxLineGap=80
    )
    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            cv2.line(h_mask_propre, (x1, y1), (x2, y2), 255, 2)

    lines_v = cv2.HoughLinesP(
        v_mask_brut, 1, np.pi / 180,
        threshold=40, minLineLength=80, maxLineGap=80
    )
    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            cv2.line(v_mask_propre, (x1, y1), (x2, y2), 255, 2)

    # ── C. Fusion H + V → masque de grille ──
    fusion_brute = cv2.bitwise_or(h_mask_propre, v_mask_propre)
    kernel_fuse  = np.ones((3, 3), np.uint8)
    grid_mask    = cv2.dilate(fusion_brute, kernel_fuse, iterations=1)

    # ── D. Suppression ──
    # image_propre_inv : chiffres blancs sur fond noir, SANS la grille
    image_propre_inv = cv2.subtract(binaire, grid_mask)

    if debug and prefixe_debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(f"{prefixe_debug}_grille_h.png",    h_mask_propre)
        cv2.imwrite(f"{prefixe_debug}_grille_v.png",    v_mask_propre)
        cv2.imwrite(f"{prefixe_debug}_grille_mask.png", grid_mask)
        cv2.imwrite(f"{prefixe_debug}_propre_inv.png",  image_propre_inv)

    return image_propre_inv, grid_mask


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — REDRESSEMENT PAR HOMOGRAPHIE (QR CODES)
# ─────────────────────────────────────────────────────────────────────────────

def detecter_qr_codes(image_propre_inv: np.ndarray) -> dict:
    """
    Détecte les 4 QR codes dans les coins de la page en cherchant
    le plus grand contour dans chaque zone d'angle.

    Retourne un dict {clé_qr: [cx, cy]} ou {} si détection échouée.
    """
    h, w = image_propre_inv.shape

    # On retravaille sur l'image inversée (fond blanc / chiffres noirs)
    # pour la recherche de contours des QR codes
    binary_recherche = cv2.bitwise_not(image_propre_inv)
    marge_qr = int(w * 0.15)

    cles_qr = [
        "marker barcode tl page1",
        "marker barcode tr page1",
        "marker barcode br page1",
        "marker barcode bl page1"
    ]

    zones_recherche = {
        cles_qr[0]: (0,             0,             marge_qr,     marge_qr    ),
        cles_qr[1]: (w - marge_qr,  0,             w,            marge_qr    ),
        cles_qr[2]: (w - marge_qr,  h - marge_qr,  w,            h           ),
        cles_qr[3]: (0,             h - marge_qr,  marge_qr,     h           ),
    }

    centres_scan = {}
    kernel_qr = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    for nom, (x1, y1, x2, y2) in zones_recherche.items():
        zone        = binary_recherche[y1:y2, x1:x2]
        zone_fermee = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, kernel_qr)
        contours, _ = cv2.findContours(
            zone_fermee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            gros = max(contours, key=cv2.contourArea)
            xr, yr, wr, hr = cv2.boundingRect(gros)
            cx = xr + wr / 2.0 + x1
            cy = yr + hr / 2.0 + y1
            centres_scan[nom] = [cx, cy]

    return centres_scan


def calculer_homographie(centres_scan: dict, data_json: dict,
                         img_w: int, img_h: int) -> np.ndarray | None:
    """
    Calcule la matrice d'homographie pixels_scan → pixels_redresse.

    Les positions théoriques des QR codes (en mm dans atomic-boxes.json)
    sont converties en pixels via les dimensions réelles de l'image,
    en supposant une page A4 (210×297 mm).
    Ainsi le warp redresse la distorsion SANS changer l'espace de coordonnées.
    """
    cles_qr = [
        "marker barcode tl page1",
        "marker barcode tr page1",
        "marker barcode br page1",
        "marker barcode bl page1"
    ]

    if len(centres_scan) != 4:
        return None

    # Échelle mm → pixels basée sur la taille réelle de l'image
    scale_x = img_w / 210.0   # A4 = 210mm de large
    scale_y = img_h / 297.0   # A4 = 297mm de haut

    centres_theoriques = {}
    for cle in cles_qr:
        if cle not in data_json:
            return None
        box = data_json[cle]
        cx_mm = box['x'] + box['width']  / 2.0
        cy_mm = box['y'] + box['height'] / 2.0
        # Conversion mm → pixels
        centres_theoriques[cle] = [cx_mm * scale_x, cy_mm * scale_y]

    # src = positions pixels détectées dans le scan (avec distorsion)
    # dst = positions pixels théoriques (sans distorsion)
    src_points = np.array([centres_scan[k]        for k in cles_qr], dtype="float32")
    dst_points = np.array([centres_theoriques[k]  for k in cles_qr], dtype="float32")

    # H : pixels_scan → pixels_redresses (même espace, distorsion corrigée)
    H, _ = cv2.findHomography(src_points, dst_points)
    return H


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — CORRECTION ET SAUVEGARDE
# ─────────────────────────────────────────────────────────────────────────────

def corriger_page(image_bgr: np.ndarray, data_json: dict,
                  numero_page: int, debug: bool = False) -> np.ndarray | None:
    """
    Applique le pipeline complet de correction sur une page :
      1. Binarisation (utilisée uniquement pour détecter la grille et les QR codes)
      2. Suppression de la grille
      3. Détection QR codes + homographie
      4. Retourne l'image GRISE originale redressée (pas binarisée)
         → ainsi detection_coins.py peut faire son propre seuillage

    Retourne None si la correction échoue (QR codes non détectés).
    """
    pfx = os.path.join(DEBUG_DIR, f"page{numero_page}")

    # ── Image grise originale (conservée pour la sortie finale) ──
    gray_original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── 1. Binarisation (uniquement pour détection grille + QR) ──
    binaire = binariser(image_bgr)
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(f"{pfx}_1_binaire.png", binaire)

    # ── 2. Suppression grille ──
    image_propre_inv, _ = supprimer_grille(binaire, debug=debug, prefixe_debug=pfx)

    # ── 3. QR codes + homographie ──
    # On détecte les QR sur l'image binarisée (meilleure détection)
    centres_scan = detecter_qr_codes(image_propre_inv)
    if len(centres_scan) != 4:
        print(f"  [WARN] Page {numero_page} : seulement {len(centres_scan)}/4 QR codes détectés — page ignorée.")
        return None

    h_img, w_img = gray_original.shape
    H = calculer_homographie(centres_scan, data_json, w_img, h_img)
    if H is None:
        print(f"  [WARN] Page {numero_page} : homographie impossible.")
        return None

    # ── 4. Redressement de l'image GRISE originale ──
    # On applique le warp sur le niveau de gris original (pas binarisé)
    # pour que detection_coins.py puisse faire son propre traitement
    image_corrigee = cv2.warpPerspective(
        gray_original, H,
        (w_img, h_img),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255   # fond blanc
    )

    if debug:
        cv2.imwrite(f"{pfx}_4_corrigee.png", image_corrigee)

    return image_corrigee


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def executer_correction(chemin_pdf: str, chemin_json: str,
                        dossier_sortie: str = OUTPUT_DIR,
                        debug: bool = False):
    """
    Pipeline complet :
      - Rasterise le PDF page par page
      - Corrige chaque page (suppression grille + homographie)
      - Sauvegarde les PNG corrigés dans dossier_sortie/
    """
    if not PDF_AVAILABLE:
        raise ImportError(
            "pdf2image requis : pip install pdf2image\n"
            "Linux : sudo apt install poppler-utils"
        )

    print(f"\n[Correction] PDF  : {chemin_pdf}")
    print(f"[Correction] JSON : {chemin_json}")
    print(f"[Correction] DPI  : {PDF_DPI}\n")

    with open(chemin_json, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    os.makedirs(dossier_sortie, exist_ok=True)
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    import platform
    if platform.system() == "Windows":
        import os as _os
        _os.environ["PATH"] += _os.pathsep + POPPLER_PATH_WINDOWS
        pages = convert_from_path(chemin_pdf, dpi=PDF_DPI, poppler_path=POPPLER_PATH_WINDOWS)
    else:
        pages = convert_from_path(chemin_pdf, dpi=PDF_DPI)
    print(f"  {len(pages)} page(s) détectée(s)\n")

    resultats = []
    for i, page_pil in enumerate(pages, start=1):
        import numpy as np
        page_bgr = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)
        print(f"  Traitement page {i}...")

        image_corrigee = corriger_page(page_bgr, data_json, i, debug=debug)

        if image_corrigee is not None:
            chemin_sortie = os.path.join(dossier_sortie, f"page_{i:03d}.png")
            cv2.imwrite(chemin_sortie, image_corrigee)
            print(f"  ✓ Page {i} → {chemin_sortie}")
            resultats.append({"page": i, "fichier": chemin_sortie, "statut": "ok"})
        else:
            print(f"  ✗ Page {i} → correction échouée")
            resultats.append({"page": i, "fichier": None, "statut": "echec"})

    print(f"\n{'─'*50}")
    ok    = sum(1 for r in resultats if r["statut"] == "ok")
    echec = sum(1 for r in resultats if r["statut"] == "echec")
    print(f"  ✓ {ok} page(s) corrigée(s)   ✗ {echec} échec(s)")
    print(f"  Dossier de sortie : {dossier_sortie}/")
    print(f"{'─'*50}\n")

    return resultats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correction de distorsion — Projet Hekzam\n"
                    "Supprime la grille et redresse les pages avant détection des coins."
    )
    parser.add_argument("--pdf",    required=True, help="Fichier PDF scanné")
    parser.add_argument("--json",   required=True, help="atomic-boxes.json (coordonnées QR codes)")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help=f"Dossier de sortie (défaut: {OUTPUT_DIR})")
    parser.add_argument("--debug",  action="store_true",
                        help="Sauvegarde les étapes intermédiaires dans ./debug_correction/")
    args = parser.parse_args()

    executer_correction(
        chemin_pdf    = args.pdf,
        chemin_json   = args.json,
        dossier_sortie= args.output,
        debug         = args.debug
    )
