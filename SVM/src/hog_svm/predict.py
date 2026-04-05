# predict.py
# Fonctions de prédiction avec mesure du temps d'exécution.

import time
import numpy as np
import cv2

from sklearn.svm import SVC
from hog_features import extract_hog_features


def predict_digit(image: np.ndarray, model: SVC) -> int:
    """
    Prédit le chiffre manuscrit contenu dans une image 28x28.

    Pipeline :
        image 28x28 → extraction HOG → reshape → SVM.predict() → label

    Args:
        image : image du chiffre, shape (28, 28) ou (28, 28, 3)
                uint8 [0-255] ou float32 [0-1]
        model : modèle SVM chargé et entraîné

    Returns:
        prediction : chiffre prédit (int entre 0 et 9)

    Exemple :
        img = cv2.imread("case_extraite.png", cv2.IMREAD_GRAYSCALE)
        label = predict_digit(img, model)
        print(f"Chiffre reconnu : {label}")
    """
    # Extraction des features HOG
    features = extract_hog_features(image)

    # Reshape en (1, n_features) pour sklearn
    features_2d = features.reshape(1, -1)

    # Prédiction
    prediction = model.predict(features_2d)[0]

    return int(prediction)


def predict_digit_timed(image: np.ndarray, model: SVC) -> tuple[int, float]:
    """
    Prédit le chiffre ET mesure le temps de traitement.

    Args:
        image : image 28x28
        model : modèle SVM entraîné

    Returns:
        (prediction, elapsed_ms) :
            - prediction  : chiffre prédit (int)
            - elapsed_ms  : temps de prédiction en millisecondes
    """
    t_start = time.perf_counter()   # perf_counter : précision µs
    prediction = predict_digit(image, model)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    return prediction, elapsed_ms


def predict_batch(images: list[np.ndarray], model: SVC) -> dict:
    """
    Prédit les chiffres pour un lot d'images et mesure les performances globales.

    Plus efficace que d'appeler predict_digit() en boucle car les features
    sont accumulées en une seule matrice avant d'appeler model.predict() une fois.

    Args:
        images : liste d'images 28x28
        model  : modèle SVM entraîné

    Returns:
        dict avec :
            "predictions"       : liste des chiffres prédits
            "time_total_ms"     : temps total en millisecondes
            "time_per_image_ms" : temps moyen par image en millisecondes
            "throughput"        : images par seconde

    Exemple :
        results = predict_batch(list_of_images, model)
        print(results["predictions"])
        print(f"Vitesse : {results['throughput']:.0f} images/s")
    """
    if not images:
        return {"predictions": [], "time_total_ms": 0, "time_per_image_ms": 0, "throughput": 0}

    # --- Extraction HOG pour toutes les images
    t_hog_start = time.perf_counter()
    feature_matrix = np.array([extract_hog_features(img) for img in images])
    t_hog_elapsed = (time.perf_counter() - t_hog_start) * 1000

    # --- Prédiction sur tout le batch d'un coup
    t_pred_start = time.perf_counter()
    predictions = model.predict(feature_matrix)
    t_pred_elapsed = (time.perf_counter() - t_pred_start) * 1000

    t_total_ms = t_hog_elapsed + t_pred_elapsed
    n = len(images)

    print(f"\n[Batch] {n} images traitées")
    print(f"  Extraction HOG : {t_hog_elapsed:.1f}ms ({t_hog_elapsed/n:.3f}ms/image)")
    print(f"  Prédiction SVM : {t_pred_elapsed:.1f}ms ({t_pred_elapsed/n:.3f}ms/image)")
    print(f"  Total          : {t_total_ms:.1f}ms ({t_total_ms/n:.3f}ms/image)")
    print(f"  Débit          : {n / (t_total_ms/1000):.0f} images/seconde")

    return {
        "predictions":       [int(p) for p in predictions],
        "time_total_ms":     t_total_ms,
        "time_per_image_ms": t_total_ms / n,
        "throughput":        n / (t_total_ms / 1000)
    }


def predict_with_confidence(image: np.ndarray, model: SVC) -> tuple[int, np.ndarray]:
    """
    Prédit le chiffre ET retourne les scores de décision (distance aux hyperplans SVM)
    pour chaque classe.

    Note : les scores SVM ne sont PAS des probabilités au sens strict.
    Pour de vraies probabilités, utiliser SVC(probability=True) à l'entraînement
    (mais c'est plus lent à cause de la calibration Platt).

    Args:
        image : image 28x28
        model : modèle SVM entraîné

    Returns:
        (prediction, scores) :
            - prediction : chiffre prédit (int)
            - scores     : vecteur de scores par classe
    """
    features = extract_hog_features(image).reshape(1, -1)
    prediction = int(model.predict(features)[0])

    # Scores de décision (distance aux hyperplans)
    scores = model.decision_function(features)[0]

    return prediction, scores
