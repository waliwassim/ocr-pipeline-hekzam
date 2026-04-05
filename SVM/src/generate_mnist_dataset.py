# generate_mnist_dataset.py
# Télécharge MNIST et génère le dossier dataset/ au format attendu par HOG+SVM.
#
# Structure produite :
#   dataset/
#     0/  → 6903 images
#     1/  → 7877 images
#     ...
#     9/  → 6958 images
#
# Utilisation :
#   python generate_mnist_dataset.py
#   python generate_mnist_dataset.py --output mon_dataset/ --max 5000

import argparse
import cv2
import numpy as np
from pathlib import Path


def generate_mnist_dataset(output_dir: str = "dataset", max_per_class: int = None):
    """
    Télécharge MNIST via scikit-learn et sauvegarde les images
    dans des sous-dossiers par classe (0 à 9).

    Args:
        output_dir    : dossier de sortie (défaut : "dataset/")
        max_per_class : limite d'images par classe (None = tout garder)
    """
    # Import ici pour un message d'erreur clair si manquant
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        raise ImportError("scikit-learn requis : pip install scikit-learn")

    print("="*55)
    print("  Génération du dataset MNIST")
    print("="*55)
    print(f"\n[1/3] Téléchargement de MNIST (peut prendre ~30s la première fois)...")

    # Téléchargement — mis en cache dans ~/scikit_learn_data/ après la 1ère fois
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.uint8)   # shape (70000, 784), valeurs 0-255
    y = mnist.target.astype(int)      # shape (70000,), labels 0-9

    print(f"    MNIST chargé : {len(X)} images de 28x28")

    # Comptage par classe avant filtrage
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n[2/3] Distribution originale :")
    for cls, cnt in zip(unique, counts):
        limit_str = f" → limité à {max_per_class}" if max_per_class else ""
        print(f"    Classe {cls} : {cnt} images{limit_str}")

    # Création des dossiers
    out_path = Path(output_dir)
    for cls in range(10):
        (out_path / str(cls)).mkdir(parents=True, exist_ok=True)

    print(f"\n[3/3] Sauvegarde dans '{output_dir}/'...")

    # Index par classe pour appliquer max_per_class
    class_counters = {i: 0 for i in range(10)}
    saved_total = 0
    skipped_total = 0

    for idx, (pixels, label) in enumerate(zip(X, y)):
        # Vérification limite par classe
        if max_per_class and class_counters[label] >= max_per_class:
            skipped_total += 1
            continue

        # Reshape en image 28x28
        img = pixels.reshape(28, 28)

        # Nom de fichier unique
        filename = out_path / str(label) / f"{label}_{idx:05d}.png"
        cv2.imwrite(str(filename), img)

        class_counters[label] += 1
        saved_total += 1

        # Progression
        if saved_total % 5000 == 0:
            print(f"    {saved_total} images sauvegardées...")

    # Résumé
    print(f"\n{'='*55}")
    print(f"  Dataset généré avec succès !")
    print(f"{'='*55}")
    print(f"  Dossier    : {out_path.resolve()}")
    print(f"  Total      : {saved_total} images sauvegardées")
    if skipped_total:
        print(f"  Ignorées   : {skipped_total} (limite max_per_class={max_per_class})")
    print(f"\n  Par classe :")
    for cls in range(10):
        print(f"    {cls}/ → {class_counters[cls]} images")
    print(f"\n  Prêt pour : python main.py --dataset {output_dir}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génère le dataset MNIST dans le format attendu par HOG+SVM"
    )
    parser.add_argument("--output", default="dataset",
                        help="Dossier de sortie (défaut: dataset/)")
    parser.add_argument("--max", type=int, default=None,
                        help="Nombre max d'images par classe (défaut: tout)")
    args = parser.parse_args()

    generate_mnist_dataset(output_dir=args.output, max_per_class=args.max)
