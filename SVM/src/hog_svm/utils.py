# utils.py
# Sauvegarde / chargement du modèle et utilitaires de débogage.

import cv2
import numpy as np
import joblib

from pathlib import Path
from skimage.feature import hog as skimage_hog
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Sauvegarde et chargement du modèle
# ---------------------------------------------------------------------------

def save_model(model: SVC, path: str) -> None:
    """
    Sauvegarde le modèle SVM entraîné sur le disque au format joblib.

    joblib est préféré à pickle pour les objets scikit-learn car il est
    plus efficace pour les tableaux numpy internes au modèle SVM.

    Args:
        model : modèle SVC entraîné
        path  : chemin de sauvegarde (ex: "models/svm_digits.joblib")

    Exemple :
        save_model(model, "models/svm_digits.joblib")
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, save_path)
    size_kb = save_path.stat().st_size / 1024
    print(f"[Modèle] Sauvegardé : {save_path} ({size_kb:.1f} Ko)")


def load_model(path: str) -> SVC:
    """
    Charge un modèle SVM précédemment sauvegardé.

    Args:
        path : chemin vers le fichier .joblib

    Returns:
        model : SVC chargé, prêt pour la prédiction

    Raises:
        FileNotFoundError si le fichier n'existe pas.

    Exemple :
        model = load_model("models/svm_digits.joblib")
        label = model.predict(features.reshape(1, -1))[0]
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    model = joblib.load(load_path)
    print(f"[Modèle] Chargé depuis : {load_path}")
    return model


# ---------------------------------------------------------------------------
# Sauvegarde des images de debug
# ---------------------------------------------------------------------------

def save_debug_images(
    original: np.ndarray,
    normalized: np.ndarray,
    output_dir: str = "debug_outputs",
    prefix: str = "debug"
) -> None:
    """
    Sauvegarde 3 images de débogage pour une case extraite :
        1. Image originale (telle qu'extraite du formulaire)
        2. Image normalisée 28x28 (après preprocessing)
        3. Visualisation HOG (orientations des gradients)

    Structure de sortie :
        debug_outputs/
            debug_original.png
            debug_normalized.png
            debug_hog.png

    Args:
        original   : image brute extraite du formulaire (taille variable)
        normalized : image 28x28 normalisée (float32 [0,1] ou uint8)
        output_dir : dossier de sortie
        prefix     : préfixe des noms de fichiers
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Image 1 : originale
    orig_path = out_path / f"{prefix}_original.png"
    cv2.imwrite(str(orig_path), original)
    print(f"[Debug] Original      → {orig_path}")

    # --- Image 2 : normalisée 28x28
    # Convertir float32 [0,1] → uint8 [0,255] si nécessaire
    if normalized.dtype in (np.float32, np.float64):
        norm_uint8 = (normalized * 255).astype(np.uint8)
    else:
        norm_uint8 = normalized

    # Agrandir pour la lisibilité (28x28 → 280x280)
    norm_display = cv2.resize(norm_uint8, (280, 280), interpolation=cv2.INTER_NEAREST)
    norm_path = out_path / f"{prefix}_normalized.png"
    cv2.imwrite(str(norm_path), norm_display)
    print(f"[Debug] Normalisée    → {norm_path}")

    # --- Image 3 : visualisation HOG
    # La visualisation montre des traits orientés selon les gradients dominants
    gray_norm = norm_uint8 if len(norm_uint8.shape) == 2 else cv2.cvtColor(norm_uint8, cv2.COLOR_BGR2GRAY)

    _, hog_image = skimage_hog(
        gray_norm,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True
    )

    # Normaliser la visualisation HOG pour l'affichage
    hog_normalized = (hog_image / hog_image.max() * 255).astype(np.uint8) if hog_image.max() > 0 else hog_image.astype(np.uint8)

    # Agrandir pour la lisibilité
    hog_display = cv2.resize(hog_normalized, (280, 280), interpolation=cv2.INTER_NEAREST)

    # Appliquer une colormap pour mieux visualiser les orientations
    hog_colored = cv2.applyColorMap(hog_display, cv2.COLORMAP_JET)

    hog_path = out_path / f"{prefix}_hog.png"
    cv2.imwrite(str(hog_path), hog_colored)
    print(f"[Debug] Visualisation HOG → {hog_path}")

    # --- Image 4 : comparaison côte à côte (original | normalisée | HOG)
    h_target = 280
    orig_resized = cv2.resize(
        original if len(original.shape) == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR),
        (280, h_target)
    )
    norm_bgr = cv2.cvtColor(norm_display, cv2.COLOR_GRAY2BGR) if len(norm_display.shape) == 2 else norm_display
    comparison = np.hstack([orig_resized, norm_bgr, hog_colored])

    comp_path = out_path / f"{prefix}_comparison.png"
    cv2.imwrite(str(comp_path), comparison)
    print(f"[Debug] Comparaison   → {comp_path}")


# ---------------------------------------------------------------------------
# Utilitaires divers
# ---------------------------------------------------------------------------

def print_model_info(model: SVC) -> None:
    """
    Affiche des informations résumées sur le modèle SVM entraîné.

    Args:
        model : SVC entraîné
    """
    print(f"\n[Info Modèle]")
    print(f"  Type         : SVC (Support Vector Classifier)")
    print(f"  Kernel       : {model.kernel}")
    print(f"  C            : {model.C}")
    print(f"  Gamma        : {model.gamma}")
    print(f"  Classes      : {list(model.classes_)}")
    print(f"  Vecteurs de support : {model.n_support_}")
    print(f"  Total SV     : {sum(model.n_support_)}")


def load_single_image(image_path: str) -> np.ndarray | None:
    """
    Charge une seule image en niveaux de gris.

    Args:
        image_path : chemin vers l'image

    Returns:
        image numpy (uint8, niveaux de gris) ou None si erreur
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[ERREUR] Impossible de charger : {image_path}")
    return image
