import json
import cv2
import numpy as np
from pdf2image import convert_from_path

PDF_PATH = "scans/1e-r-0.pdf"
JSON_PATH = "cases.json"
OUT_PATH = "overlay_debug_homography.png"
PDF_DPI = 200

def detecter_qr_centers_px(gray):
    h_img, w_img = gray.shape[:2]

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers_px = []
    margin = 0.15  # seulement dans les coins
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if (cy < h_img * margin or cy > h_img * (1 - margin)) and \
           (cx < w_img * margin or cx > w_img * (1 - margin)):
            centers_px.append([cx, cy])

    if len(centers_px) < 4:
        raise ValueError(f"Seulement {len(centers_px)} marqueurs trouvés sur 4 requis.")

    centers_px = np.array(centers_px, dtype=np.float32)

    # Tri TL, TR, BL, BR
    s = centers_px.sum(axis=1)
    diff = np.diff(centers_px, axis=1)

    pts_px = np.zeros((4, 2), dtype=np.float32)
    pts_px[0] = centers_px[np.argmin(s)]     # TL
    pts_px[1] = centers_px[np.argmin(diff)]  # TR
    pts_px[2] = centers_px[np.argmax(diff)]  # BL
    pts_px[3] = centers_px[np.argmax(s)]     # BR

    return pts_px

def transformer_point_mm_vers_px(pt_mm, H):
    if pt_mm is None:
        return None
    p = np.array([[[pt_mm[0], pt_mm[1]]]], dtype=np.float32)
    p2 = cv2.perspectiveTransform(p, H)
    x, y = p2[0, 0]
    return [int(round(x)), int(round(y))]

# 1) Rasteriser la 1re page du PDF
page = convert_from_path(PDF_PATH, dpi=PDF_DPI, first_page=1, last_page=1)[0]
img = np.array(page.convert("RGB"))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Image shape:", img.shape)

# 2) Charger le JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    cases = json.load(f)

print("RAW first case:", cases[0]["coins"])

# 3) Centres théoriques des QR codes en mm
# D'après ton document :
# QR TL : x=10.0, y=10.0, w=12.5, h=12.5
# QR TR : x=187.5, y=10.0, w=12.5, h=12.5
# QR BL : x=10.0, y=274.5, w=12.5, h=12.5
# QR BR : x=187.5, y=274.5, w=12.5, h=12.5

pts_mm = np.array([
    [16.25, 16.25],    # TL
    [193.75, 16.25],   # TR
    [16.25, 280.75],   # BL
    [193.75, 280.75],  # BR
], dtype=np.float32)

# 4) Détection des QR codes réels dans l'image
pts_px = detecter_qr_centers_px(gray)

print("QR centers px:", pts_px.tolist())
print("QR centers mm:", pts_mm.tolist())

# 5) Homographie mm -> px
H_mm_to_px, _ = cv2.findHomography(pts_mm, pts_px)
print("Homography matrix:\n", H_mm_to_px)

# 6) Conversion des coins JSON (mm -> px)
for case in cases:
    for k, pt in case["coins"].items():
        if pt is not None:
            case["coins"][k] = transformer_point_mm_vers_px(pt, H_mm_to_px)

print("PIXEL first case:", cases[0]["coins"])

# 7) Dessiner toutes les cases
for case in cases:
    cid = case.get("id", "?")
    coins = case["coins"]

    pts = {}
    for name in ["tl", "tr", "bl", "br"]:
        if coins.get(name) is not None:
            pts[name] = tuple(coins[name])
            cv2.circle(img, pts[name], 3, (0, 0, 255), -1)  # rouge

    if all(k in pts for k in ["tl", "tr", "br", "bl"]):
        poly = np.array([pts["tl"], pts["tr"], pts["br"], pts["bl"]], dtype=np.int32)
        cv2.polylines(img, [poly], True, (0, 255, 0), 1)  # vert

# 8) Sauvegarde
cv2.imwrite(OUT_PATH, img)
print("Saved:", OUT_PATH)