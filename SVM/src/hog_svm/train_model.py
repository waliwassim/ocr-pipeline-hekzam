# train_model.py
# Préparation du dataset, entraînement du SVM, et évaluation des performances.

import os
import time
import numpy as np
import cv2

from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from hog_features import extract_hog_features


# ---------------------------------------------------------------------------
# Chargement et préparation du dataset
# ---------------------------------------------------------------------------

def prepare_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge les images depuis un dossier organisé par classe et extrait les
    features HOG pour chaque image.

    Structure attendue du dossier :
        dataset/
            0/   ← images du chiffre 0
            1/   ← images du chiffre 1
            ...
            9/   ← images du chiffre 9

    Args:
        dataset_dir : chemin vers le dossier racine du dataset

    Returns:
        X : matrice de features, shape (n_samples, n_features)
        y : vecteur de labels, shape (n_samples,)

    Exemple :
        X, y = prepare_dataset("dataset/")
        print(X.shape)  # (5000, 81)
        print(y.shape)  # (5000,)
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dossier dataset introuvable : {dataset_dir}")

    X_list = []
    y_list = []
    total  = 0
    errors = 0

    # Parcourir chaque sous-dossier (= une classe)
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"Aucun sous-dossier de classes trouvé dans : {dataset_dir}")

    print(f"[Dataset] {len(class_dirs)} classes trouvées : {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        # Le nom du dossier est le label (ex: "3" → label 3)
        try:
            label = int(class_dir.name)
        except ValueError:
            print(f"[AVERTISSEMENT] Dossier ignoré (nom non numérique) : {class_dir.name}")
            continue

        # Extensions d'images acceptées
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ]

        if not image_files:
            print(f"[AVERTISSEMENT] Aucune image dans la classe {label}.")
            continue

        class_count = 0
        for img_path in image_files:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                errors += 1
                continue

            # Extraction des features HOG
            try:
                features = extract_hog_features(image)
                X_list.append(features)
                y_list.append(label)
                class_count += 1
                total += 1
            except Exception as e:
                print(f"[ERREUR] HOG échoué pour {img_path.name} : {e}")
                errors += 1

        print(f"  Classe {label:2d} : {class_count} images chargées")

    if not X_list:
        raise ValueError("Aucune image n'a pu être chargée.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\n[Dataset] Total : {total} images, {errors} erreurs")
    print(f"[Dataset] Shape X : {X.shape}, Shape y : {y.shape}")

    return X, y


# ---------------------------------------------------------------------------
# Entraînement du SVM
# ---------------------------------------------------------------------------

def train_svm(X: np.ndarray, y: np.ndarray) -> SVC:
    """
    Entraîne un SVM avec noyau RBF sur les features HOG.

    Paramètres SVM choisis :
        - kernel="rbf"  : noyau gaussien, adapté aux données non-linéairement séparables
        - C=10          : pénalité des erreurs de classification
                          (valeur élevée = frontière plus stricte, risque de surapprentissage)
        - gamma="scale" : σ du noyau RBF calculé automatiquement depuis les données

    Args:
        X : features HOG, shape (n_samples, n_features)
        y : labels,        shape (n_samples,)

    Returns:
        model : SVC entraîné
    """
    print(f"\n[SVM] Entraînement sur {len(X)} exemples...")
    t_start = time.time()

    model = SVC(
        kernel="rbf",
        gamma="scale",
        C=10,
        decision_function_shape="ovr",  # one-vs-rest pour classification multi-classe
        random_state=42
    )
    model.fit(X, y)

    t_elapsed = time.time() - t_start
    print(f"[SVM] Entraînement terminé en {t_elapsed:.2f}s")

    return model


# ---------------------------------------------------------------------------
# Évaluation des performances
# ---------------------------------------------------------------------------

def evaluate_model(
    model: SVC,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None
) -> dict:
    """
    Évalue le modèle SVM sur un jeu de test et affiche les métriques.

    Métriques calculées :
        - Accuracy globale
        - Matrice de confusion
        - Classification report (précision, rappel, F1 par classe)

    Args:
        model      : SVM entraîné
        X_test     : features du jeu de test
        y_test     : labels réels du jeu de test
        class_names: noms des classes (ex: ["0","1",...,"9"])

    Returns:
        dict avec les métriques calculées
    """
    if class_names is None:
        class_names = [str(i) for i in sorted(np.unique(y_test))]

    print(f"\n[Évaluation] Prédiction sur {len(X_test)} exemples...")
    t_start = time.time()

    y_pred = model.predict(X_test)

    t_elapsed = time.time() - t_start

    # --- Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  ACCURACY : {acc*100:.2f}%")
    print(f"  Temps de prédiction : {t_elapsed*1000:.1f}ms pour {len(X_test)} images")
    print(f"  Soit {t_elapsed/len(X_test)*1000:.3f}ms par image")
    print(f"{'='*50}")

    # --- Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatrice de confusion :")
    print(f"{'':5}", end="")
    for c in class_names:
        print(f"{c:5}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{class_names[i]:5}", end="")
        for val in row:
            print(f"{val:5}", end="")
        print()

    # --- Classification report
    print(f"\nClassification report :")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return {
        "accuracy":   acc,
        "confusion":  cm,
        "y_pred":     y_pred,
        "time_total": t_elapsed,
        "time_per_image": t_elapsed / len(X_test)
    }


# ---------------------------------------------------------------------------
# Pipeline d'entraînement complet
# ---------------------------------------------------------------------------

def full_training_pipeline(
    dataset_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Pipeline complet :
        1. Charger le dataset
        2. Diviser en train/test
        3. Entraîner le SVM
        4. Évaluer les performances

    Args:
        dataset_dir  : chemin vers le dataset organisé par classes
        test_size    : proportion du dataset réservée au test (défaut : 20%)
        random_state : graine aléatoire pour reproductibilité

    Returns:
        (model, X_train, X_test, y_train, y_test, metrics)
    """
    # Chargement
    X, y = prepare_dataset(dataset_dir)

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # même proportion de chaque classe dans train et test
    )
    print(f"\n[Split] Train : {len(X_train)}, Test : {len(X_test)}")

    # Entraînement
    model = train_svm(X_train, y_train)

    # Évaluation
    metrics = evaluate_model(model, X_test, y_test)

    return model, X_train, X_test, y_train, y_test, metrics
