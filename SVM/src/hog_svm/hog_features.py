# hog_features.py
# Extraction des descripteurs HOG à partir d'images 28x28.
#
# HOG = Histogram of Oriented Gradients
# Principe :
#   - L'image est découpée en petites cellules (ici 4x4 pixels)
#   - Dans chaque cellule, on calcule un histogramme des directions de gradient
#   - Les histogrammes voisins sont regroupés en blocs et normalisés
#   - Le résultat est un vecteur de features représentant la "forme" de l'image

import numpy as np
import cv2
from skimage.feature import hog


# ---------------------------------------------------------------------------
# Paramètres HOG (centralisés ici pour cohérence dans tout le projet)
# ---------------------------------------------------------------------------
HOG_PARAMS = {
    "orientations":    9,       # nombre de directions (0° à 180°)
    "pixels_per_cell": (4, 4),  # taille d'une cellule
    "cells_per_block": (2, 2),  # nombre de cellules par bloc (pour normalisation)
    "block_norm":      "L2-Hys" # méthode de normalisation robuste
}

# Taille attendue des images en entrée
IMAGE_SIZE = (28, 28)


def extract_hog_features(image: np.ndarray, visualize: bool = False):
    """
    Extrait le vecteur de features HOG d'une image 28x28.

    Le vecteur HOG encode la distribution des contours dans l'image,
    ce qui représente efficacement la forme d'un chiffre manuscrit.

    Args:
        image    : image numpy, shape (28, 28) ou (28, 28, 3)
                   valeurs uint8 [0-255] ou float32 [0-1]
        visualize: si True, retourne aussi l'image de visualisation HOG

    Returns:
        features : vecteur numpy 1D (les features HOG)
        hog_img  : image de visualisation (seulement si visualize=True)

    Exemple :
        features = extract_hog_features(img_28x28)
        # features.shape → (81,) avec ces paramètres
    """
    # --- Étape 1 : conversion en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- Étape 2 : normalisation en uint8 si l'image est en float [0, 1]
    if gray.dtype in (np.float32, np.float64):
        gray = (gray * 255).astype(np.uint8)

    # --- Étape 3 : vérification de la taille
    if gray.shape != IMAGE_SIZE:
        gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # --- Étape 4 : extraction HOG
    if visualize:
        features, hog_img = hog(
            gray,
            **HOG_PARAMS,
            visualize=True,
            feature_vector=True   # retourne un vecteur 1D
        )
        return features, hog_img
    else:
        features = hog(
            gray,
            **HOG_PARAMS,
            visualize=False,
            feature_vector=True
        )
        return features


def get_feature_vector_size() -> int:
    """
    Calcule la taille du vecteur HOG produit pour une image IMAGE_SIZE.

    Formule :
        Pour une image (H, W) avec pixels_per_cell=(p,p) et cells_per_block=(c,c) :
        n_cells_x = W // p
        n_cells_y = H // p
        n_blocks_x = n_cells_x - c + 1
        n_blocks_y = n_cells_y - c + 1
        taille = n_blocks_x * n_blocks_y * c * c * orientations

    Returns:
        taille du vecteur de features (int)
    """
    dummy = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    features = extract_hog_features(dummy)
    return len(features)


if __name__ == "__main__":
    # Test rapide : vérifier que l'extraction fonctionne
    dummy_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    features = extract_hog_features(dummy_image)
    print(f"Taille du vecteur HOG : {len(features)} features")
    print(f"Valeurs min/max : {features.min():.4f} / {features.max():.4f}")
