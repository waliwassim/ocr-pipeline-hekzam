# retrain.py
# Réentraîne le SVM en combinant MNIST + les crops extraits des formulaires.
#
# Stratégie :
#   - MNIST         : base générale (beaucoup d'images, style varié)
#   - Crops réels   : domaine cible (peu d'images mais exactement le bon style)
#   - sample_weight : on donne un poids WEIGHT_REAL fois plus élevé aux crops réels
#
# Utilisation :
#   python3 retrain.py \
#       --mnist   ../dataset \
#       --crops   ../results/crops \
#       --model   ../hog_svm/models/svm_digits_retrained.joblib

import argparse
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Ajouter hog_svm/ au path
sys.path.append(str(Path(__file__).parent.parent / "hog_svm"))

from hog_features import extract_hog_features
from train_model  import train_svm, evaluate_model
from utils        import save_model, load_model
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

# Poids relatif des crops réels vs MNIST
# 10 = chaque crop réel compte autant que 10 images MNIST
WEIGHT_REAL = 10

# Nombre max d'images MNIST par classe (pour ne pas noyer les crops réels)
MAX_MNIST_PER_CLASS = 1000


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_images_from_folder(folder: str, max_per_class: int = None) -> tuple:
    """
    Charge les images depuis un dossier organisé par classes (0-9).

    Args:
        folder        : dossier racine avec sous-dossiers 0/, 1/, ..., 9/
        max_per_class : limite d'images par classe (None = tout)

    Returns:
        (X, y) : features HOG et labels
    """
    X_list, y_list = [], []
    folder_path = Path(folder)

    # Chercher les sous-dossiers numériques (0-9) ou label_X
    class_dirs = []
    for d in sorted(folder_path.iterdir()):
        if not d.is_dir():
            continue
        # Accepter "0", "1", ... ou "label_0", "label_1", ...
        name = d.name.replace("label_", "")
        try:
            label = int(name)
            class_dirs.append((label, d))
        except ValueError:
            continue

    if not class_dirs:
        print(f"[AVERTISSEMENT] Aucun sous-dossier de classes dans : {folder}")
        return np.array([]), np.array([])

    for label, class_dir in class_dirs:
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        ]

        if max_per_class:
            image_files = image_files[:max_per_class]

        count = 0
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                features = extract_hog_features(img)
                X_list.append(features)
                y_list.append(label)
                count += 1
            except Exception:
                continue

        print(f"  Classe {label} : {count} images chargées depuis {class_dir.name}/")

    if not X_list:
        return np.array([]), np.array([])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_crops(crops_dir: str) -> tuple:
    """
    Charge les crops extraits des formulaires.
    Structure attendue :
        crops/
            page_1/
                label_0/  *.png
                label_1/  *.png
                ...
    """
    X_list, y_list = [], []
    crops_path = Path(crops_dir)

    # Parcourir toutes les pages
    for page_dir in sorted(crops_path.iterdir()):
        if not page_dir.is_dir():
            continue

        # Parcourir les sous-dossiers label_X
        for label_dir in sorted(page_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            name = label_dir.name.replace("label_", "")
            try:
                label = int(name)
            except ValueError:
                continue

            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                try:
                    features = extract_hog_features(img)
                    X_list.append(features)
                    y_list.append(label)
                except Exception:
                    continue

    if not X_list:
        return np.array([]), np.array([])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# Pipeline de réentraînement
# ---------------------------------------------------------------------------

def retrain(mnist_dir: str, crops_dir: str, model_output: str):
    """
    Réentraîne le SVM en combinant MNIST et les crops réels.
    """
    print("\n" + "="*60)
    print("  RÉENTRAÎNEMENT SVM — MNIST + Crops formulaires")
    print("="*60)

    # ------------------------------------------------------------------
    # 1. Charger MNIST (limité à MAX_MNIST_PER_CLASS par classe)
    # ------------------------------------------------------------------
    print(f"\n[1/4] Chargement MNIST depuis : {mnist_dir}")
    print(f"      (max {MAX_MNIST_PER_CLASS} images par classe)")
    X_mnist, y_mnist = load_images_from_folder(mnist_dir, max_per_class=MAX_MNIST_PER_CLASS)

    if len(X_mnist) == 0:
        print("[ERREUR] Aucune image MNIST chargée.")
        return

    print(f"  → {len(X_mnist)} images MNIST chargées")

    # ------------------------------------------------------------------
    # 2. Charger les crops réels
    # ------------------------------------------------------------------
    print(f"\n[2/4] Chargement des crops réels depuis : {crops_dir}")
    X_real, y_real = load_crops(crops_dir)

    if len(X_real) == 0:
        print("[AVERTISSEMENT] Aucun crop réel trouvé → entraînement sur MNIST seul")
        X_combined = X_mnist
        y_combined = y_mnist
        weights    = np.ones(len(X_mnist))
    else:
        print(f"  → {len(X_real)} crops réels chargés")

        # ------------------------------------------------------------------
        # 3. Combiner les deux datasets avec sample_weight
        # ------------------------------------------------------------------
        print(f"\n[3/4] Combinaison des datasets")
        print(f"  MNIST  : {len(X_mnist)} images  (poids = 1)")
        print(f"  Réels  : {len(X_real)} images  (poids = {WEIGHT_REAL})")
        print(f"  Équivalent à : {len(X_mnist) + len(X_real) * WEIGHT_REAL} images")

        X_combined = np.vstack([X_mnist, X_real])
        y_combined = np.concatenate([y_mnist, y_real])

        # Poids : 1 pour MNIST, WEIGHT_REAL pour les crops réels
        weights_mnist = np.ones(len(X_mnist))
        weights_real  = np.full(len(X_real), WEIGHT_REAL)
        weights       = np.concatenate([weights_mnist, weights_real])

    # ------------------------------------------------------------------
    # 4. Split train/test (stratifié)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_combined, y_combined, weights,
        test_size=0.2, random_state=42, stratify=y_combined
    )
    print(f"\n  Split → Train : {len(X_train)}, Test : {len(X_test)}")

    # ------------------------------------------------------------------
    # 5. Entraînement avec sample_weight
    # ------------------------------------------------------------------
    print(f"\n[4/4] Entraînement du SVM...")
    from sklearn.svm import SVC

    t_start = time.time()
    model = SVC(kernel="rbf", gamma="scale", C=10,
                decision_function_shape="ovr", random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)
    t_elapsed = time.time() - t_start
    print(f"  Entraînement terminé en {t_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 6. Évaluation
    # ------------------------------------------------------------------
    print(f"\n[Évaluation globale] (MNIST + réels mélangés)")
    metrics_global = evaluate_model(model, X_test, y_test)

    # Évaluation séparée sur les crops réels uniquement (le plus important)
    if len(X_real) > 0:
        # Prendre un sous-ensemble de crops réels pour le test
        n_test_real = max(10, len(X_real) // 5)
        idx = np.random.choice(len(X_real), n_test_real, replace=False)
        X_real_test = X_real[idx]
        y_real_test = y_real[idx]

        print(f"\n[Évaluation sur crops RÉELS uniquement] ({n_test_real} images)")
        metrics_real = evaluate_model(model, X_real_test, y_real_test)
        print(f"\n  ★ Accuracy sur formulaires réels : {metrics_real['accuracy']*100:.1f}%")

    # ------------------------------------------------------------------
    # 7. Sauvegarde
    # ------------------------------------------------------------------
    save_model(model, model_output)
    print(f"\n  Nouveau modèle sauvegardé → {model_output}")
    print(f"\n  Maintenant relance le pipeline avec --model {model_output}")
    print("="*60 + "\n")

    return model


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Réentraîne le SVM sur MNIST + crops réels des formulaires"
    )
    parser.add_argument("--mnist",  default="../dataset",
                        help="Dossier MNIST (défaut: ../dataset)")
    parser.add_argument("--crops",  default="../results/crops",
                        help="Dossier crops extraits (défaut: ../results/crops)")
    parser.add_argument("--model",  default="../hog_svm/models/svm_digits_retrained.joblib",
                        help="Chemin du nouveau modèle sauvegardé")
    parser.add_argument("--weight", type=int, default=WEIGHT_REAL,
                        help=f"Poids des crops réels vs MNIST (défaut: {WEIGHT_REAL})")

    args = parser.parse_args()
    WEIGHT_REAL = args.weight

    retrain(
        mnist_dir    = args.mnist,
        crops_dir    = args.crops,
        model_output = args.model
    )
