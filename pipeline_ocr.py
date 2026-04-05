"""
Pipeline OCR — Projet Hekzam  (section 5.a de l'état de l'art)
================================================================
Auteur : W. Wassim

Rôle dans la chaîne :
    [Collègue 1 : Détection coins] ──JSON──► [CE SCRIPT] ──PNG──► [Collègue 2 : STN-LeNet]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT JSON D'ENTRÉE (à transmettre au collègue 1) :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  {
    "id":   "c1",
    "page": 1,
    "coins": {
      "tl": [120, 340],   ← top-left     (x, y) ou null si non détecté
      "tr": [180, 342],   ← top-right
      "bl": [119, 400],   ← bottom-left
      "br": [181, 401]    ← bottom-right
    }
  },
  {
    "id":   "c2",
    "page": 1,
    "coins": {
      "tl": [200, 340],
      "tr": null,         ← coin non détecté (contour dégradé)
      "bl": [199, 400],
      "br": [261, 399]
    }
  }
]

Nombre de coins détectés → stratégie adoptée :
  4 coins  → perspective warp exact  (redressement de la distorsion ADF)
  3 coins  → 4e coin reconstruit par géométrie (parallélogramme)
  2 coins  → bounding box estimée depuis les 2 coins + taille médiane
  0-1 coin → interpolation depuis les cases voisines détectées
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage :
    python pipeline_ocr.py --pdf scan.pdf --json cases.json
    python pipeline_ocr.py --pdf scan.pdf --json cases.json --debug
    python pipeline_ocr.py --demo
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = "./cases_hekzam"
DEBUG_DIR    = "./debug_pipeline"
RESULTS_FILE = "./results.json"

PDF_DPI      = 200
TARGET_SIZE  = 28

EMPTY_DENSITY_THRESHOLD = 0.01   # 1 % → case considérée vide
BORDER_CROP_RATIO       = 0.06   # rognage du cadre imprimé (6 %)

try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION DES COINS MANQUANTS
# ─────────────────────────────────────────────────────────────────────────────

def reconstruire_coins(coins: dict, toutes_cases: list, case_cible: dict) -> dict | None:
    """
    Reçoit un dict {"tl", "tr", "bl", "br"} où certaines valeurs peuvent être None.
    Retourne un dict complet des 4 coins, ou None si reconstruction impossible.

    Stratégie selon le nombre de coins présents :
      4 → retourné tel quel
      3 → le 4e est déduit par la règle du parallélogramme :
             coin_manquant = coin_opposé_diag + vecteur_adjacent1 + vecteur_adjacent2
      2 → bounding box estimée depuis les 2 coins + taille médiane des voisins
      1 → idem mais avec 1 seul coin d'ancrage
      0 → interpolation depuis les cases voisines (grille)
    """
    presents = {k: v for k, v in coins.items() if v is not None}
    nb = len(presents)

    if nb == 4:
        return coins

    if nb == 3:
        return _reconstruire_4e_coin(coins)

    if nb >= 1:
        return _estimer_depuis_taille_mediane(presents, toutes_cases)

    # 0 coin détecté → interpolation depuis les voisins
    return _interpoler_depuis_voisins(toutes_cases, case_cible)


def _reconstruire_4e_coin(coins: dict) -> dict | None:
    """
    Avec 3 coins connus, le 4e se calcule par la propriété du parallélogramme :
        tl + br = tr + bl  (les diagonales se coupent en leur milieu)
    Donc : coin_manquant = somme_des_deux_autres_coins - coin_opposé_connu
    """
    noms  = ["tl", "tr", "bl", "br"]
    # Paires opposées (diagonales)
    opposees = [("tl", "br"), ("tr", "bl")]

    manquant = next(k for k, v in coins.items() if v is None)
    result = dict(coins)

    # Trouver la diagonale qui contient le manquant
    for (a, b) in opposees:
        if manquant in (a, b):
            autre = b if manquant == a else a
            # Les deux autres coins (les deux de l'autre diagonale)
            reste = [k for k in noms if k not in (a, b)]
            if coins[reste[0]] is not None and coins[reste[1]] is not None and coins[autre] is not None:
                # coin_manquant = (reste[0] + reste[1]) - coin_autre
                p1 = np.array(coins[reste[0]], dtype=float)
                p2 = np.array(coins[reste[1]], dtype=float)
                p3 = np.array(coins[autre],    dtype=float)
                reconstruit = p1 + p2 - p3
                result[manquant] = reconstruit.astype(int).tolist()
                return result

    return None


def _estimer_depuis_taille_mediane(presents: dict, toutes_cases: list) -> dict | None:
    """
    Avec 1 ou 2 coins, estime les 4 coins en utilisant la taille médiane
    des cases voisines bien détectées (4 coins présents).
    """
    # Taille médiane des cases complètes
    cases_completes = [
        c for c in toutes_cases
        if c.get("coins") and sum(1 for v in c["coins"].values() if v is not None) == 4
    ]
    if not cases_completes:
        return None

    largeurs = []
    hauteurs = []
    for c in cases_completes:
        co = c["coins"]
        w = ((np.array(co["tr"]) - np.array(co["tl"]))[0] +
             (np.array(co["br"]) - np.array(co["bl"]))[0]) / 2
        h = ((np.array(co["bl"]) - np.array(co["tl"]))[1] +
             (np.array(co["br"]) - np.array(co["tr"]))[1]) / 2
        largeurs.append(abs(w))
        hauteurs.append(abs(h))

    w_med = int(np.median(largeurs))
    h_med = int(np.median(hauteurs))

    # Ancre : utiliser le premier coin disponible comme tl de référence
    ordre_preference = ["tl", "bl", "tr", "br"]
    decalages = {
        "tl": (0,     0    ),
        "tr": (-w_med, 0   ),
        "bl": (0,     -h_med),
        "br": (-w_med, -h_med)
    }

    ancre_nom = next((k for k in ordre_preference if k in presents), None)
    if ancre_nom is None:
        return None

    ancre = np.array(presents[ancre_nom])
    dx, dy = decalages[ancre_nom]
    tl = ancre + np.array([dx, dy])

    return {
        "tl": tl.tolist(),
        "tr": (tl + np.array([w_med, 0])).tolist(),
        "bl": (tl + np.array([0, h_med])).tolist(),
        "br": (tl + np.array([w_med, h_med])).tolist()
    }


def _interpoler_depuis_voisins(toutes_cases: list, case_cible: dict) -> dict | None:
    """
    0 coin détecté : estime la position à partir des cases voisines (grille).
    Utilise l'espacement médian entre cases pour extrapoler.
    """
    cases_completes = [
        c for c in toutes_cases
        if c.get("coins") and sum(1 for v in c["coins"].values() if v is not None) == 4
        and c is not case_cible
    ]
    if len(cases_completes) < 2:
        return None

    # Centre de chaque case complète
    centres = []
    for c in cases_completes:
        co = c["coins"]
        pts = np.array([co["tl"], co["tr"], co["bl"], co["br"]])
        centres.append((c, pts.mean(axis=0)))

    # Taille médiane
    result = _estimer_depuis_taille_mediane({}, toutes_cases)
    if result is None:
        return None
    w_med = int(np.array(result["tr"])[0] - np.array(result["tl"])[0])
    h_med = int(np.array(result["bl"])[1] - np.array(result["tl"])[1])

    # Espacement médian horizontal
    triees = sorted(centres, key=lambda x: (x[1][1], x[1][0]))
    espacements = []
    for i in range(1, len(triees)):
        dy = abs(triees[i][1][1] - triees[i-1][1][1])
        if dy < h_med * 0.5:
            dx = triees[i][1][0] - triees[i-1][1][0]
            if dx > 0:
                espacements.append(dx)

    if not espacements:
        return None

    pas_x = int(np.median(espacements))
    idx_cible  = toutes_cases.index(case_cible)

    # Voisin le plus proche par index
    voisin, centre_voisin = min(
        centres, key=lambda x: abs(toutes_cases.index(x[0]) - idx_cible)
    )
    delta = idx_cible - toutes_cases.index(voisin)
    tl = np.array([
        centre_voisin[0] + delta * pas_x - w_med / 2,
        centre_voisin[1] - h_med / 2
    ]).astype(int)

    return {
        "tl": tl.tolist(),
        "tr": (tl + np.array([w_med, 0])).tolist(),
        "bl": (tl + np.array([0, h_med])).tolist(),
        "br": (tl + np.array([w_med, h_med])).tolist()
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION ROI DEPUIS LES 4 COINS
# ─────────────────────────────────────────────────────────────────────────────

def extraire_roi_depuis_coins(page_gris: np.ndarray, coins: dict,
                               nb_coins_originaux: int) -> np.ndarray:
    """
    Extrait la région de la case depuis les 4 coins.

    Si tous les 4 coins étaient détectés → perspective warp :
        redresse la distorsion trapézoïdale introduite par l'ADF.
    Sinon → crop simple de la bounding box
        (la distorsion résiduelle sera gérée par le STN-LeNet).
    """
    tl = np.array(coins["tl"], dtype=np.float32)
    tr = np.array(coins["tr"], dtype=np.float32)
    bl = np.array(coins["bl"], dtype=np.float32)
    br = np.array(coins["br"], dtype=np.float32)

    # Taille cible du warp
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    w = max(w, 10)
    h = max(h, 10)

    if nb_coins_originaux == 4:
        # Perspective warp exact
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M   = cv2.getPerspectiveTransform(src, dst)
        roi = cv2.warpPerspective(page_gris, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    else:
        # Crop simple de la bounding box
        H, W = page_gris.shape
        x1 = max(0, int(min(tl[0], bl[0])))
        y1 = max(0, int(min(tl[1], tr[1])))
        x2 = min(W, int(max(tr[0], br[0])))
        y2 = min(H, int(max(bl[1], br[1])))
        roi = page_gris[y1:y2, x1:x2].copy()

    if roi.size == 0:
        raise ValueError("ROI vide après extraction.")

    # Rognage du cadre imprimé
    marge_y = max(1, int(roi.shape[0] * BORDER_CROP_RATIO))
    marge_x = max(1, int(roi.shape[1] * BORDER_CROP_RATIO))
    roi_rogne = roi[marge_y:-marge_y, marge_x:-marge_x]
    return roi_rogne if roi_rogne.size > 0 else roi


# ─────────────────────────────────────────────────────────────────────────────
# PRÉTRAITEMENT, SEGMENTATION, NORMALISATION
# (identiques à la v1 — LeCun et al. [4])
# ─────────────────────────────────────────────────────────────────────────────

def pretraiter(roi: np.ndarray) -> np.ndarray:
    """Binarisation Otsu + inversion + nettoyage morphologique."""
    _, b = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.bitwise_not(b)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return b


def est_vide(binaire: np.ndarray) -> bool:
    if binaire.size == 0:
        return True
    return (np.count_nonzero(binaire) / binaire.size) < EMPTY_DENSITY_THRESHOLD


def segmenter_chiffre(binaire: np.ndarray) -> np.ndarray:
    """
    Isole le contour le plus pertinent pour le chiffre
    en évitant de garder la bordure complète de la case.
    """
    contours, _ = cv2.findContours(binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return binaire

    H, W = binaire.shape
    aire_image = H * W
    valid_contours = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area_box = w * h
        area_cnt = cv2.contourArea(c)

        # Rejeter les tout petits bruits
        if area_cnt < 8:
            continue

        # Rejeter les contours trop grands (souvent la bordure de la case)
        if area_box > aire_image * 0.80:
            continue

        # Rejeter les traits très fins horizontaux/verticaux
        if w > 0 and h > 0:
            ratio = max(w / h, h / w)
            if ratio > 12:
                continue

        valid_contours.append(c)

    if not valid_contours:
        return binaire

    # Garder le contour le plus "utile"
    c = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    m = 2
    x1 = max(0, x - m)
    y1 = max(0, y - m)
    x2 = min(W, x + w + m)
    y2 = min(H, y + h + m)

    return binaire[y1:y2, x1:x2]


def normaliser_mnist(chiffre: np.ndarray) -> np.ndarray:
    """Produit une image 28×28 centrée par centre de masse (standard MNIST)."""
    if chiffre.size == 0 or chiffre.max() == 0:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    h, w  = chiffre.shape
    ratio = 20.0 / max(h, w)
    redim = cv2.resize(chiffre,
                       (max(1, int(w*ratio)), max(1, int(h*ratio))),
                       interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    oh = (TARGET_SIZE - redim.shape[0]) // 2
    ow = (TARGET_SIZE - redim.shape[1]) // 2
    canvas[oh:oh+redim.shape[0], ow:ow+redim.shape[1]] = redim
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        M = np.float32([[1, 0, TARGET_SIZE//2 - int(m["m10"]/m["m00"])],
                        [0, 1, TARGET_SIZE//2 - int(m["m01"]/m["m00"])]])
        canvas = cv2.warpAffine(canvas, M, (TARGET_SIZE, TARGET_SIZE),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION — traitement d'une case
# ─────────────────────────────────────────────────────────────────────────────

def traiter_case(page_gris: np.ndarray, case: dict,
                 toutes_cases: list, debug: bool = False) -> dict:
    """
    Applique le pipeline complet sur une case.

    Retourne :
        {
          "id"              : identifiant,
          "statut"          : "ok" | "vide" | "approximatif" | "non_lisible",
          "coins_utilises"  : nb de coins originalement détectés,
          "fichier"         : chemin PNG produit,
          "erreur"          : message si non_lisible
        }
    """
    case_id = case.get("id", "?")
    res = {"id": case_id, "statut": "non_lisible",
           "coins_utilises": 0, "fichier": None, "erreur": None}
    pfx = os.path.join(DEBUG_DIR, f"case_{case_id}")

    try:
        coins_bruts = case.get("coins", {})
        nb_coins_originaux = sum(1 for v in coins_bruts.values() if v is not None)
        res["coins_utilises"] = nb_coins_originaux

        # ── Reconstruction des coins manquants ──
        coins = reconstruire_coins(coins_bruts, toutes_cases, case)
        if coins is None:
            res["erreur"] = f"Reconstruction impossible ({nb_coins_originaux} coin(s) détecté(s), pas assez de voisins)."
            return res

        if nb_coins_originaux < 4:
            print(f"  ↻  {case_id} : {nb_coins_originaux} coin(s) → reconstruction {'géom.' if nb_coins_originaux==3 else 'taille med.' if nb_coins_originaux>=1 else 'interp. grille'}")

        # ── 1. Extraction ROI ──
        roi = extraire_roi_depuis_coins(page_gris, coins, nb_coins_originaux)
        if debug: _dbg(roi, f"{pfx}_1_roi.png")

        # ── 2. Prétraitement ──
        bin_img = pretraiter(roi)
        if debug: _dbg(bin_img, f"{pfx}_2_binaire.png")

        # ── 3. Vide ? ──
        if est_vide(bin_img):
            chemin = _save(np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8), case_id)
            res.update(statut="vide", fichier=chemin)
            return res

        # ── 4. Segmentation ──
        chiffre = segmenter_chiffre(bin_img)
        if debug: _dbg(chiffre, f"{pfx}_3_segment.png")

        # ── 5. Normalisation ──
        finale = normaliser_mnist(chiffre)
        if debug: _dbg(finale, f"{pfx}_4_normalise.png")

        chemin = _save(finale, case_id)
        statut = "ok" if nb_coins_originaux == 4 else "approximatif"
        res.update(statut=statut, fichier=chemin)

    except Exception as e:
        res["erreur"] = str(e)
        print(f"  ✗  {case_id} : {e}")

    return res


def _save(img: np.ndarray, case_id: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chemin = os.path.join(OUTPUT_DIR, f"case_{case_id}.png")
    Image.fromarray(img).save(chemin)
    return chemin


def _dbg(img: np.ndarray, chemin: str):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    Image.fromarray(img).save(chemin)


# ─────────────────────────────────────────────────────────────────────────────
# RASTERISATION PDF
# ─────────────────────────────────────────────────────────────────────────────

def rasteriser_pdf(chemin_pdf: str) -> dict:
    if not PDF_AVAILABLE:
        raise ImportError(
            "pdf2image requis : pip install pdf2image\n"
            "Linux : sudo apt install poppler-utils\n"
            "macOS : brew install poppler"
        )
    print(f"Rasterisation du PDF à {PDF_DPI} DPI...")
    pages = {}
    import platform
    if platform.system() == "Windows":
        import os as _os
        _os.environ["PATH"] += _os.pathsep + POPPLER_PATH_WINDOWS
        _pages = convert_from_path(chemin_pdf, dpi=PDF_DPI, poppler_path=POPPLER_PATH_WINDOWS)
    else:
        _pages = convert_from_path(chemin_pdf, dpi=PDF_DPI)
    for i, page in enumerate(_pages, start=1):
        pages[i] = np.array(page.convert("L"))
        print(f"  Page {i} : {pages[i].shape[1]}×{pages[i].shape[0]} px")
    return pages


def lire_images_png(dossier_images: str) -> dict:
    """
    Charge les images PNG corrigées depuis le dossier produit par
    correction_distorsion.py, en remplacement de rasteriser_pdf().

    Les fichiers doivent être nommés page_001.png, page_002.png, etc.
    Retourne un dict {numero_page: image_numpy_gris}.
    """
    import re
    pages = {}
    fichiers = sorted([
        f for f in os.listdir(dossier_images)
        if f.lower().endswith(".png")
    ])
    if not fichiers:
        raise FileNotFoundError(f"Aucun PNG trouvé dans : {dossier_images}")

    print(f"Chargement des images corrigées depuis : {dossier_images}")
    for f in fichiers:
        match = re.search(r'(\d+)', f)
        num = int(match.group(1)) if match else len(pages) + 1
        chemin = os.path.join(dossier_images, f)
        img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            pages[num] = img
            print(f"  Page {num} : {img.shape[1]}×{img.shape[0]} px  ({f})")
        else:
            print(f"  [WARN] Impossible de lire : {f}")
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def calculer_homographie_mm_vers_px(page_gris: np.ndarray):
    h_img, w_img = page_gris.shape[:2]

    # Centres théoriques des QR codes en mm
    pts_mm = np.array([
        [16.25, 16.25],    # TL
        [193.75, 16.25],   # TR
        [16.25, 280.75],   # BL
        [193.75, 280.75]   # BR
    ], dtype="float32")

    # Détection des QR codes dans l'image
    _, thresh = cv2.threshold(page_gris, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers_px = []
    margin = 0.15

    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if (cy < h_img * margin or cy > h_img * (1 - margin)) and \
               (cx < w_img * margin or cx > w_img * (1 - margin)):
                if cv2.contourArea(cnt) > 500:
                    centers_px.append([cx, cy])

    if len(centers_px) < 4:
        raise ValueError(f"Seulement {len(centers_px)} QR codes détectés sur 4.")

    centers_px = np.array(centers_px, dtype="float32")

    # Tri TL, TR, BL, BR
    s = centers_px.sum(axis=1)
    diff = np.diff(centers_px, axis=1)

    pts_px = np.zeros((4, 2), dtype="float32")
    pts_px[0] = centers_px[np.argmin(s)]     # TL
    pts_px[1] = centers_px[np.argmin(diff)]  # TR
    pts_px[2] = centers_px[np.argmax(diff)]  # BL
    pts_px[3] = centers_px[np.argmax(s)]     # BR

    # Homographie mm -> px
    matrix_mm_to_px, _ = cv2.findHomography(pts_mm, pts_px)
    return matrix_mm_to_px
def transformer_point_mm_vers_px(pt_mm, H):
    if pt_mm is None:
        return None
    p = np.array([[[pt_mm[0], pt_mm[1]]]], dtype=np.float32)
    p2 = cv2.perspectiveTransform(p, H)
    x, y = p2[0, 0]
    return [int(round(x)), int(round(y))]
def executer_pipeline(chemin_json: str, chemin_pdf: str = None,
                      dossier_images: str = None, debug: bool = False):
    """
    chemin_json    : JSON des coins (cases.json)
    chemin_pdf     : PDF scanné (utilisé si dossier_images absent)
    dossier_images : dossier PNG de correction_distorsion.py (prioritaire sur le PDF)
    """
    print(f"\n[OCR Pipeline] JSON : {chemin_json}")

    with open(chemin_json, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Normalisation des champs optionnels
    for i, c in enumerate(cases):
        c.setdefault("id", f"case_{i:03d}")
        c.setdefault("page", 1)
        c.setdefault("coins", {"tl": None, "tr": None, "bl": None, "br": None})

    # ── Chargement des pages ──
    if dossier_images:
        print(f"[OCR Pipeline] Images : {dossier_images} (PNG corrigés)\n")
        pages = lire_images_png(dossier_images)
    elif chemin_pdf:
        print(f"[OCR Pipeline] PDF  : {chemin_pdf}\n")
        pages = rasteriser_pdf(chemin_pdf)
    else:
        raise ValueError("Il faut fournir --images ou --pdf")

    # Calcul de l'homographie sur la première page
    H_mm_to_px = calculer_homographie_mm_vers_px(pages[1])
    print("[OCR Pipeline] Homographie mm -> px calculée depuis les QR codes.\n")

    # Conversion des coins JSON (mm) -> pixels
    for case in cases:
        for k, pt in case["coins"].items():
            if pt is not None:
                case["coins"][k] = transformer_point_mm_vers_px(pt, H_mm_to_px)

    resultats = []
    print(f"Traitement de {len(cases)} cases...\n")
    sym = {"ok": "✓", "vide": "○", "approximatif": "≈", "non_lisible": "✗"}

    for case in cases:
        pg = pages.get(case.get("page", 1))
        if pg is None:
            r = {
                "id": case["id"],
                "statut": "non_lisible",
                "coins_utilises": 0,
                "fichier": None,
                "erreur": f"Page {case.get('page', 1)} introuvable"
            }
            resultats.append(r)
            continue

        r = traiter_case(pg, case, cases, debug=debug)
        resultats.append(r)

        print(f"  {sym.get(r['statut'], '?')}  {case['id']:15s}  "
              f"coins={r['coins_utilises']}/4  → {r['statut']}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)

    stats = {
        s: sum(1 for r in resultats if r["statut"] == s)
        for s in ["ok", "vide", "approximatif", "non_lisible"]
    }

    print(f"\n{'─'*50}")
    print(f"  ✓ ok={stats['ok']}  ≈ approx={stats['approximatif']}"
          f"  ○ vides={stats['vide']}  ✗ non_lisibles={stats['non_lisible']}")
    print(f"  Dossier : {OUTPUT_DIR}/   Résumé : {RESULTS_FILE}")
    print(f"{'─'*50}\n")

    return resultats
# ─────────────────────────────────────────────────────────────────────────────
# MODE DÉMO
# ─────────────────────────────────────────────────────────────────────────────

def executer_demo():
    """Teste le pipeline sur une page synthétique avec différents scénarios de coins."""
    print("\n[DEMO] Test avec 5 scénarios de coins...\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    page = np.ones((500, 700), dtype=np.uint8) * 235

    # Génère une case avec un chiffre dessiné et légère distorsion
    def gen_case(chiffre, x0, y0, taille=60, distorsion=0):
        arr = np.ones((taille, taille), dtype=np.uint8) * 240
        # Cadre
        cv2.rectangle(arr, (2, 2), (taille-3, taille-3), 80, 1)
        # Chiffre grossier (croix ou forme simple selon chiffre)
        mid = taille // 2
        if chiffre % 2 == 0:
            cv2.line(arr, (mid-12, mid), (mid+12, mid), 30, 3)
            cv2.line(arr, (mid, mid-12), (mid, mid+12), 30, 3)
        else:
            cv2.circle(arr, (mid, mid), 12, 30, 2)
        # Bruit
        bruit = np.random.normal(0, 10, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + bruit, 0, 255).astype(np.uint8)
        page[y0:y0+taille, x0:x0+taille] = arr
        # Coins avec légère distorsion simulant l'ADF
        d = distorsion
        return {
            "tl": [x0 + d,        y0 + d       ],
            "tr": [x0 + taille,   y0           ],
            "bl": [x0,            y0 + taille  ],
            "br": [x0 + taille-d, y0 + taille+d]
        }

    np.random.seed(42)
    scenarios = [
        # (id, chiffre, x0, y0, coins_a_supprimer)
        ("4coins",   3, 50,  50,  []),
        ("3coins",   7, 150, 50,  ["tr"]),
        ("2coins",   1, 250, 50,  ["tr", "br"]),
        ("1coin",    5, 350, 50,  ["tr", "bl", "br"]),
        ("0coin",    9, 450, 50,  ["tl", "tr", "bl", "br"]),
    ]

    cases = []
    for case_id, chiffre, x0, y0, a_supprimer in scenarios:
        coins = gen_case(chiffre, x0, y0, distorsion=3)
        for k in a_supprimer:
            coins[k] = None
        nb = sum(1 for v in coins.values() if v is not None)
        cases.append({
            "id": case_id, "page": 1,
            "coins": coins,
            "_chiffre_attendu": chiffre,
            "_nb_coins_originaux": nb
        })

    resultats = []
    sym = {"ok": "✓", "vide": "○", "approximatif": "≈", "non_lisible": "✗"}
    for case in cases:
        r = traiter_case(page, case, cases, debug=True)
        resultats.append(r)
        nb = case["_nb_coins_originaux"]
        print(f"  {sym.get(r['statut'],'?')}  {case['id']:12s}  "
              f"{nb} coin(s)  chiffre={case['_chiffre_attendu']}  → {r['statut']}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)

    print(f"\n  Images    → {OUTPUT_DIR}/")
    print(f"  Debug     → {DEBUG_DIR}/")
    print(f"  Résumé    → {RESULTS_FILE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global OUTPUT_DIR, DEBUG_DIR, RESULTS_FILE
    parser = argparse.ArgumentParser(
        description="Pipeline OCR Hekzam — gestion des coins partiels"
    )
    parser.add_argument("--images", help="Dossier PNG corrigés (sortie de correction_distorsion.py) — PRIORITAIRE sur --pdf")
    parser.add_argument("--pdf",    help="Fichier PDF scanné (utilisé si --images absent)")
    parser.add_argument("--json",   help="JSON des coordonnées des cases")
    parser.add_argument("--debug",  action="store_true",
                        help="Sauvegarde les étapes dans ./debug_pipeline/")
    parser.add_argument("--demo",   action="store_true",
                        help="Test sur données synthétiques")
    args = parser.parse_args()

    if args.demo:
        executer_demo()
    elif args.json and (args.images or args.pdf):

        # Nom de base pour les dossiers de sortie
        if args.images:
            # ex: scans_corriges/2e-r-0 → "2e-r-0"
            pdf_name = os.path.basename(os.path.normpath(args.images))
        else:
            pdf_name = os.path.splitext(os.path.basename(args.pdf))[0]

        OUTPUT_DIR   = os.path.join("results", pdf_name, "cases_hekzam")
        DEBUG_DIR    = os.path.join("results", pdf_name, "debug_pipeline")
        RESULTS_FILE = os.path.join("results", pdf_name, "results.json")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(DEBUG_DIR,  exist_ok=True)

        print(f"\n[INFO] Dossier de sortie : {OUTPUT_DIR}")
        print(f"[INFO] Dossier debug     : {DEBUG_DIR}")
        print(f"[INFO] Fichier résumé    : {RESULTS_FILE}\n")

        executer_pipeline(
            chemin_json    = args.json,
            chemin_pdf     = args.pdf,
            dossier_images = args.images,
            debug          = args.debug
        )
    else:
        parser.print_help()
        print("\n  Exemple avec images corrigées (recommandé) :")
        print("    python pipeline_ocr.py --images scans_corriges/2e-r-0/ --json cases.json")
        print("  Exemple avec PDF :")
        print("    python pipeline_ocr.py --pdf scan.pdf --json cases.json")
        print("  Démo    : python pipeline_ocr.py --demo\n")

if __name__ == "__main__":
    main()  