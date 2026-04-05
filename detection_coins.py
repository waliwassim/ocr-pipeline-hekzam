"""
Détection des coins et cases — Projet Hekzam
=============================================
Auteur : [Collègue B]

Rôle dans la chaîne :
    [correction_distorsion.py] ──PNG corrigés──► [CE SCRIPT] ──JSON──► [pipeline_ocr.py]

Prend en entrée le dossier de pages corrigées produit par correction_distorsion.py
et génère un fichier JSON de coins par page.

Usage :
    python detection_coins.py --input scans_corriges/ --output results/
    python detection_coins.py --input scans_corriges/ --output results/ --debug
"""

import cv2
import numpy as np
import json
import os
import argparse

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR  = "./scans_corriges"
OUTPUT_DIR = "./results"


def sort_points_logically(points, y_threshold=20):
    """Regroupe les points par lignes (Y) puis les trie par colonnes (X)."""
    if not points:
        return []
    points.sort(key=lambda p: p[1])
    sorted_list = []
    current_line = [points[0]]
    for i in range(1, len(points)):
        if abs(points[i][1] - current_line[-1][1]) <= y_threshold:
            current_line.append(points[i])
        else:
            current_line.sort(key=lambda p: p[0])
            sorted_list.extend(current_line)
            current_line = [points[i]]
    current_line.sort(key=lambda p: p[0])
    sorted_list.extend(current_line)
    return sorted_list


def is_valid_cell(tl, br, img_w, img_h):
    """Vérifie que la case est un rectangle valide."""
    w = br[0] - tl[0]
    h = br[1] - tl[1]
    if w < img_w * 0.015 or h < img_w * 0.015:
        return False
    if w > img_w * 0.20 or h > img_w * 0.20:
        return False
    ratio = w / h if h > 0 else 0
    return 0.4 <= ratio <= 2.5


def is_cell_with_writing(cell_img, padding=5, min_pixels=80):
    """Détecte si la zone contient un chiffre manuscrit."""
    h, w = cell_img.shape
    if h <= 2 * padding or w <= 2 * padding:
        return False
    roi = cell_img[padding:h - padding, padding:w - padding]
    ink_pixels = cv2.countNonZero(roi)
    area = (h - 2 * padding) * (w - 2 * padding)
    ink_ratio = ink_pixels / area if area > 0 else 0
    return ink_pixels > min_pixels and ink_ratio > 0.02


def is_duplicate(tl_mm, existing_cells, tol=3.0):
    """Vérifie si une case (en mm) est déjà dans la liste."""
    return any(
        abs(c["tl"][0] - tl_mm[0]) < tol and abs(c["tl"][1] - tl_mm[1]) < tol
        for c in existing_cells.values()
    )


def is_handwritten(blob_roi_gray, min_stroke_thickness=1.8):
    """
    Distingue un chiffre manuscrit d'un chiffre imprimé via la transformée de distance.
    Les traits manuscrits sont plus épais que les polices imprimées.
    """
    if blob_roi_gray is None or blob_roi_gray.size == 0:
        return False
    _, bw = cv2.threshold(blob_roi_gray,237 , 255,
                          cv2.THRESH_BINARY_INV)
    if cv2.countNonZero(bw) < 30:
        return False
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    nonzero_vals = dist[dist > 0]
    if len(nonzero_vals) == 0:
        return False
    median_thickness = float(np.median(nonzero_vals)) * 2.0
    return median_thickness >= min_stroke_thickness


def group_grid_cells_into_rows(grid_cells_px, y_tol=15):
    """
    Regroupe les cases grille par rangées horizontales.

    Chaque rangée = liste de cases ayant le même Y (à y_tol près).
    Retourne : liste de rangées, chaque rangée = liste de (tl_px, br_px)
    triées par X croissant.
    """
    if not grid_cells_px:
        return []

    # Trier par Y du centre de chaque case
    sorted_cells = sorted(grid_cells_px, key=lambda c: (c[0][1] + c[1][1]) / 2)

    rows = []
    current_row = [sorted_cells[0]]
    current_y = (sorted_cells[0][0][1] + sorted_cells[0][1][1]) / 2

    for cell in sorted_cells[1:]:
        cy = (cell[0][1] + cell[1][1]) / 2
        if abs(cy - current_y) <= y_tol:
            current_row.append(cell)
        else:
            # Trier la rangée par X
            current_row.sort(key=lambda c: c[0][0])
            rows.append(current_row)
            current_row = [cell]
            current_y = cy

    current_row.sort(key=lambda c: c[0][0])
    rows.append(current_row)
    return rows


def detect_free_digits_by_row_scan(writing_only, gray, img_w, img_h,
                                   grid_cells_px, grid_mask, existing_cells,
                                   px_to_mm_fn):
    """
    MÉTHODE B — Ligne de référence horizontale.

    Pour chaque rangée de cases grille :
      1. Calculer la ligne horizontale Y_ref = haut moyen des cases de la rangée
      2. Calculer la hauteur H_ref et largeur W_ref moyennes des cases
      3. Scanner la bande [Y_ref - H_ref*2 .. Y_ref] sur toute la largeur
      4. Trouver les blobs d'écriture manuscrite dans cette bande
         qui NE SONT PAS déjà dans une case grille
      5. Pour chaque blob trouvé : créer une case virtuelle
         de dimensions (W_ref × H_ref) centrée sur le blob

    Retourne : liste de (tl_px, br_px, tl_mm, tr_mm, br_mm, bl_mm)
    """
    # Masque des zones déjà couvertes par les cases grille (dilaté)
    covered_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for (tl_px, br_px) in grid_cells_px:
        cv2.rectangle(covered_mask,
                      (int(tl_px[0]), int(tl_px[1])),
                      (int(br_px[0]), int(br_px[1])), 255, -1)
    covered_mask = cv2.dilate(covered_mask, np.ones((10, 10), np.uint8), iterations=1)

    # Masque zones QR codes (4 coins)
    qr_x = int(img_w * 0.13)
    qr_y = int(img_h * 0.09)

    # Regrouper les cases grille par rangées
    rows = group_grid_cells_into_rows(grid_cells_px, y_tol=15)

    # Légère dilatation pour fusionner les traits d'un même chiffre
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    writing_dilated = cv2.dilate(writing_only, kernel, iterations=1)

    new_cells = []

    for row in rows:
        if not row:
            continue

        # --- Dimensions de référence de cette rangée ---
        cell_heights = [br[1] - tl[1] for (tl, br) in row]
        cell_widths  = [br[0] - tl[0] for (tl, br) in row]
        H_ref = int(np.median(cell_heights))
        W_ref = int(np.median(cell_widths))

        # Y de référence = haut moyen des cases de la rangée
        Y_tops = [tl[1] for (tl, br) in row]
        Y_ref  = int(np.median(Y_tops))

        # X couverture de la rangée (min X gauche → max X droit de toute la page)
        # On scanne sur toute la largeur utile de la page
        x_scan_start = int(img_w * 0.02)
        x_scan_end   = int(img_w * 0.98)

        # Bande de scan : de Y_ref - H_ref*1.5 jusqu'à Y_ref + H_ref
        # (on prend un peu plus haut pour attraper les chiffres qui dépassent)
        y_scan_top = max(0, Y_ref - int(H_ref * 1.5))
        y_scan_bot = min(img_h, Y_ref + H_ref)

        # Extraire la bande d'écriture
        band = writing_dilated[y_scan_top:y_scan_bot, x_scan_start:x_scan_end]

        if band.size == 0:
            continue

        # Composantes connexes dans la bande
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(band)

        for i in range(1, num_labels):
            bx, by, bw, bh, b_area = stats[i]
            bcx = int(centroids[i][0])  # centre X dans la bande
            bcy = int(centroids[i][1])  # centre Y dans la bande

            # Coordonnées absolues du centre du blob
            abs_cx = x_scan_start + bcx
            abs_cy = y_scan_top   + bcy

            # Filtres de taille du blob (doit ressembler à un chiffre)
            min_dim = int(H_ref * 0.2)
            max_dim = int(H_ref * 1.5)
            if bw < min_dim or bh < min_dim:
                continue
            if bw > max_dim * 2 or bh > max_dim:
                continue

            # Exclure les coins QR
            if ((abs_cx < qr_x and abs_cy < qr_y) or
                (abs_cx > img_w - qr_x and abs_cy < qr_y) or
                (abs_cx < qr_x and abs_cy > img_h - qr_y) or
                (abs_cx > img_w - qr_x and abs_cy > img_h - qr_y)):
                continue

            # Exclure si déjà couvert par une case grille
            check_r = max(5, H_ref // 4)
            roi_covered = covered_mask[
                max(0, abs_cy - check_r):min(img_h, abs_cy + check_r),
                max(0, abs_cx - check_r):min(img_w, abs_cx + check_r)
            ]
            if cv2.countNonZero(roi_covered) > 0:
                continue

            # Exclure si trop chevauchant avec les lignes de grille
            blob_region_grid = grid_mask[
                y_scan_top + by: y_scan_top + by + bh,
                x_scan_start + bx: x_scan_start + bx + bw
            ]
            if b_area > 0 and cv2.countNonZero(blob_region_grid) / b_area > 0.30:
                continue

            # Test manuscrit vs imprimé
            blob_gray_roi = gray[
                y_scan_top + by: y_scan_top + by + bh,
                x_scan_start + bx: x_scan_start + bx + bw
            ]
            if not is_handwritten(blob_gray_roi, min_stroke_thickness=1.8):
                continue

            # --- Créer la case virtuelle centrée sur le blob ---
            # Dimensions = W_ref × H_ref (mêmes que les cases voisines)
            tl_px = (int(abs_cx - W_ref // 2), int(abs_cy - H_ref // 2))
            br_px = (int(abs_cx + W_ref // 2), int(abs_cy + H_ref // 2))
            tr_px = (br_px[0], tl_px[1])
            bl_px = (tl_px[0], br_px[1])

            # Clipper aux bords de l'image
            tl_px = (max(0, tl_px[0]), max(0, tl_px[1]))
            br_px = (min(img_w, br_px[0]), min(img_h, br_px[1]))
            tr_px = (min(img_w, tr_px[0]), max(0, tr_px[1]))
            bl_px = (max(0, bl_px[0]), min(img_h, bl_px[1]))

            tl_mm = px_to_mm_fn(tl_px)

            # Vérifier doublon
            if is_duplicate(tl_mm, existing_cells, tol=3.0):
                continue
            # Vérifier doublon avec les nouvelles cases déjà ajoutées dans ce scan
            if any(abs(nc[2][0] - tl_mm[0]) < 3.0 and abs(nc[2][1] - tl_mm[1]) < 3.0
                   for nc in new_cells):
                continue

            new_cells.append((
                tl_px, br_px,
                tl_mm,
                px_to_mm_fn(tr_px),
                px_to_mm_fn(br_px),
                px_to_mm_fn(bl_px)
            ))

            # Marquer cette zone comme couverte pour éviter les doublons
            cv2.rectangle(covered_mask, tl_px, br_px, 255, -1)

    return new_cells


def generate_corners_and_debug_image(image_path, output_json, output_image):
    img = cv2.imread(image_path)
    if img is None:
        return
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Masque du manuscrit
    writing_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Lignes de grille
    bw_clean = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    h_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 40, 1))
    v_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 60))
    horiz_lines = cv2.morphologyEx(bw_clean, cv2.MORPH_OPEN, h_kernel)
    vert_lines  = cv2.morphologyEx(bw_clean, cv2.MORPH_OPEN, v_kernel)
    grid_mask   = cv2.bitwise_or(horiz_lines, vert_lines)
    grid_mask   = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Écriture pure (grille retirée)
    writing_only = cv2.bitwise_and(writing_mask, cv2.bitwise_not(grid_mask))

    # Modèle A4 (mm)
    pts_mm = np.array(
        [[16.25, 16.25], [193.75, 16.25], [16.25, 280.75], [193.75, 280.75]],
        dtype="float32")

    # Détection des 4 marqueurs QR (coins)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers_px = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            if ((cy < h_img * 0.15 or cy > h_img * 0.85) and
                    (cx < w_img * 0.15 or cx > w_img * 0.85) and
                    cv2.contourArea(cnt) > 500):
                centers_px.append([cx, cy])

    if len(centers_px) < 4:
        raise ValueError("Marqueurs QR manquants")

    centers_px = np.array(centers_px, dtype="float32")
    s    = centers_px.sum(axis=1)
    diff = np.diff(centers_px, axis=1)
    pts_px = np.zeros((4, 2), dtype="float32")
    pts_px[0] = centers_px[np.argmin(s)]
    pts_px[1] = centers_px[np.argmin(diff)]
    pts_px[2] = centers_px[np.argmax(diff)]
    pts_px[3] = centers_px[np.argmax(s)]

    matrix, _ = cv2.findHomography(pts_px, pts_mm)

    def px_to_mm(pt):
        p   = np.array([float(pt[0]), float(pt[1]), 1.0])
        res = np.dot(matrix, p)
        return [round(res[0] / res[2], 2), round(res[1] / res[2], 2)]

    # =========================================================
    # MÉTHODE A : Cases avec grille (intersections)
    # =========================================================
    intersections = cv2.bitwise_and(horiz_lines, vert_lines)
    _, _, _, centroids = cv2.connectedComponentsWithStats(intersections)

    raw_points      = [tuple(p) for p in centroids[1:]]
    detected_points = sort_points_logically(raw_points)

    final_cells   = {}
    debug_img     = img.copy()
    cell_idx      = 1
    px_t          = 25
    grid_cells_px = []

    for p in detected_points:
        tl = p
        tr = next((x for x in detected_points
                   if abs(x[1] - tl[1]) < px_t and x[0] > tl[0] + 10), None)
        bl = next((x for x in detected_points
                   if abs(x[0] - tl[0]) < px_t and x[1] > tl[1] + 10), None)

        if not (tr and bl):
            continue

        br = next((x for x in detected_points
                   if abs(x[0] - tr[0]) < px_t and abs(x[1] - bl[1]) < px_t), None)
        if br is None:
            br = (tr[0] + bl[0] - tl[0], tr[1] + bl[1] - tl[1])

        if not is_valid_cell(tl, br, w_img, h_img):
            continue

        cell_roi = writing_only[int(tl[1]):int(br[1]), int(tl[0]):int(br[0])]
        if not is_cell_with_writing(cell_roi):
            continue

        tl_mm = px_to_mm(tl)
        if is_duplicate(tl_mm, final_cells):
            continue

        tr_f = (float(tr[0]), float(tr[1]))
        br_f = (float(br[0]), float(br[1]))
        bl_f = (float(bl[0]), float(bl[1]))

        final_cells[str(cell_idx)] = {
            "tl": tl_mm, "tr": px_to_mm(tr_f),
            "br": px_to_mm(br_f), "bl": px_to_mm(bl_f),
            "type": "grid"
        }
        grid_cells_px.append((tl, br_f))

        # Debug : BLEU pour les cases grille
        box = np.array([tl, tr_f, br_f, bl_f], np.int32)
        cv2.polylines(debug_img, [box], True, (255, 0, 0), 2)
        cx_c = int((tl[0] + br_f[0]) / 2)
        cy_c = int((tl[1] + br_f[1]) / 2)
        cv2.putText(debug_img, f"c{cell_idx}", (cx_c - 10, cy_c + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cell_idx += 1

    # =========================================================
    # MÉTHODE B : Scan par ligne de référence horizontale
    # =========================================================
    new_free_cells = detect_free_digits_by_row_scan(
        writing_only, gray, w_img, h_img,
        grid_cells_px, grid_mask, final_cells, px_to_mm
    )

    for (tl_px, br_px, tl_mm, tr_mm, br_mm, bl_mm) in new_free_cells:
        tl = (float(tl_px[0]), float(tl_px[1]))
        br = (float(br_px[0]), float(br_px[1]))
        tr = (float(br_px[0]), float(tl_px[1]))
        bl = (float(tl_px[0]), float(br_px[1]))

        final_cells[str(cell_idx)] = {
            "tl": tl_mm, "tr": tr_mm,
            "br": br_mm, "bl": bl_mm,
            "type": "free"
        }

        # Debug : VERT pour les cases virtuelles
        box = np.array([tl, tr, br, bl], np.int32)
        cv2.polylines(debug_img, [box], True, (0, 180, 0), 2)
        cx_c = int((tl[0] + br[0]) / 2)
        cy_c = int((tl[1] + br[1]) / 2)
        cv2.putText(debug_img, f"c{cell_idx}", (cx_c - 10, cy_c + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 0), 1)

        # Ligne de référence horizontale en rouge (debug visuel)
        row_y = int(tl_px[1])
        cv2.line(debug_img, (0, row_y), (w_img, row_y), (0, 0, 200), 1)

        cell_idx += 1

    # Sauvegarde
    with open(output_json, 'w') as f:
        json.dump(final_cells, f, indent=4)
    cv2.imwrite(output_image, debug_img)

    grid_count = sum(1 for c in final_cells.values() if c.get("type") == "grid")
    free_count  = sum(1 for c in final_cells.values() if c.get("type") == "free")
    print(f"  → {grid_count} grille (bleu) + {free_count} libres (vert) = {cell_idx-1} total")



# ============================================================
# CLI + BOUCLE PRINCIPALE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection des coins et cases — Projet Hekzam"
    )
    parser.add_argument("--input",  default=INPUT_DIR,
                        help=f"Dossier des PNG corrigés (sortie de correction_distorsion.py) — défaut: {INPUT_DIR}")
    parser.add_argument("--output", default=OUTPUT_DIR,
                        help=f"Dossier de sortie pour les JSON — défaut: {OUTPUT_DIR}")
    parser.add_argument("--debug",  action="store_true",
                        help="Sauvegarde les images de debug avec les cases détectées")
    args = parser.parse_args()

    dossier_input  = args.input
    dossier_output = args.output
    os.makedirs(dossier_output, exist_ok=True)

    # Lister tous les PNG dans le dossier d'entrée
    pages_png = sorted([
        f for f in os.listdir(dossier_input)
        if f.lower().endswith(".png")
    ])

    if not pages_png:
        print(f"[ERREUR] Aucun PNG trouvé dans : {dossier_input}")
        print(f"  → Lance d'abord : python correction_distorsion.py --pdf scan.pdf --json atomic-boxes.json")
        exit(1)

    print(f"\n[Détection coins] Dossier entrée : {dossier_input}")
    print(f"[Détection coins] Dossier sortie : {dossier_output}")
    print(f"[Détection coins] {len(pages_png)} page(s) trouvée(s)\n")

    for png_file in pages_png:
        page_path  = os.path.join(dossier_input, png_file)
        nom_base   = os.path.splitext(png_file)[0]
        output_json  = os.path.join(dossier_output, f"{nom_base}.json")
        output_debug = os.path.join(dossier_output, f"{nom_base}_debug.png")

        print(f"  Traitement : {png_file} ...")
        try:
            generate_corners_and_debug_image(
                page_path,
                output_json,
                output_debug if args.debug else os.path.join(dossier_output, f"{nom_base}_debug.png")
            )
            print(f"  ✓ JSON → {output_json}")
            if args.debug:
                print(f"  ✓ Debug → {output_debug}")
        except Exception as e:
            print(f"  ✗ Erreur sur {png_file} : {e}")

    print(f"\n{'─'*50}")
    print(f"  Terminé. JSON disponibles dans : {dossier_output}/")
    print(f"{'─'*50}\n")