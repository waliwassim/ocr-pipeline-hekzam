# main.py
# Pipeline complet HOG + SVM pour la reconnaissance de chiffres manuscrits.
#
# Enchaîne :
#   1. Chargement du dataset organisé par classes
#   2. Entraînement du SVM sur les features HOG
#   3. Affichage des métriques (accuracy, matrice de confusion, F1...)
#   4. Sauvegarde du modèle
#   5. Rechargement du modèle
#   6. Test sur quelques images individuelles
#   7. Test batch avec mesure du temps

import cv2
import numpy as np
from pathlib import Path

from hog_features import extract_hog_features, get_feature_vector_size
from train_model  import prepare_dataset, train_svm, evaluate_model, full_training_pipeline
from predict      import predict_digit, predict_digit_timed, predict_batch
from utils        import save_model, load_model, save_debug_images, print_model_info


# ---------------------------------------------------------------------------
# Paramètres globaux (modifier selon votre setup)
# ---------------------------------------------------------------------------

DATASET_DIR  = "dataset/"           # dossier dataset organisé en classes 0-9
MODEL_PATH   = "models/svm_digits.joblib"
DEBUG_DIR    = "debug_outputs/"
TEST_SIZE    = 0.2                  # 20% pour le test


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  PIPELINE HOG + SVM — Reconnaissance de chiffres manuscrits")
    print("="*60)

    # ------------------------------------------------------------------
    # Étape 0 : Info sur le vecteur HOG
    # ------------------------------------------------------------------
    feat_size = get_feature_vector_size()
    print(f"\n[Info] Taille du vecteur HOG par image : {feat_size} features")
    print(f"[Info] Paramètres : orientations=9, pixels_per_cell=4x4, cells_per_block=2x2")

    # ------------------------------------------------------------------
    # Étape 1 : Charger dataset + split + entraîner + évaluer
    # ------------------------------------------------------------------
    print(f"\n[Étape 1] Chargement du dataset depuis : {DATASET_DIR}")

    model, X_train, X_test, y_train, y_test, metrics = full_training_pipeline(
        dataset_dir=DATASET_DIR,
        test_size=TEST_SIZE
    )

    print(f"\n[Résultat] Accuracy : {metrics['accuracy']*100:.2f}%")

    # ------------------------------------------------------------------
    # Étape 2 : Informations sur le modèle
    # ------------------------------------------------------------------
    print_model_info(model)

    # ------------------------------------------------------------------
    # Étape 3 : Sauvegarder le modèle
    # ------------------------------------------------------------------
    print(f"\n[Étape 3] Sauvegarde du modèle → {MODEL_PATH}")
    save_model(model, MODEL_PATH)

    # ------------------------------------------------------------------
    # Étape 4 : Recharger le modèle (pour vérifier la sauvegarde)
    # ------------------------------------------------------------------
    print(f"\n[Étape 4] Rechargement du modèle depuis {MODEL_PATH}")
    loaded_model = load_model(MODEL_PATH)

    # ------------------------------------------------------------------
    # Étape 5 : Test sur quelques images individuelles avec timer
    # ------------------------------------------------------------------
    print(f"\n[Étape 5] Test sur 10 images individuelles")
    print(f"{'Index':>6} | {'Label réel':>10} | {'Prédit':>6} | {'Temps (ms)':>10} | {'Correct':>8}")
    print("-" * 55)

    n_correct = 0
    for i in range(min(10, len(X_test))):
        # Reconstruire une image factice 28x28 depuis les features
        # (en production : charger directement l'image)
        # Ici on reconstruit une image depuis les features pour la démo
        dummy_img = _features_to_demo_image(X_test[i])

        predicted, elapsed_ms = predict_digit_timed(dummy_img, loaded_model)
        true_label = int(y_test[i])
        correct = "✓" if predicted == true_label else "✗"
        if predicted == true_label:
            n_correct += 1

        print(f"{i:>6} | {true_label:>10} | {predicted:>6} | {elapsed_ms:>9.3f}ms | {correct:>8}")

    print(f"\n  Résultat : {n_correct}/10 corrects sur cet échantillon")

    # ------------------------------------------------------------------
    # Étape 6 : Test batch avec mesure du débit
    # ------------------------------------------------------------------
    print(f"\n[Étape 6] Test batch sur {len(X_test)} images")

    # Construire des images démo depuis les features de test
    batch_images = [_features_to_demo_image(x) for x in X_test[:100]]

    results = predict_batch(batch_images, loaded_model)

    print(f"\n  Prédictions (20 premières) : {results['predictions'][:20]}")
    print(f"  Labels réels  (20 premiers): {list(y_test[:20])}")

    # ------------------------------------------------------------------
    # Étape 7 : Sauvegarde d'images de debug
    # ------------------------------------------------------------------
    print(f"\n[Étape 7] Sauvegarde d'images de debug → {DEBUG_DIR}")

    # Prendre la première image de test comme exemple
    demo_img = _features_to_demo_image(X_test[0])
    true_label_demo = int(y_test[0])

    # Image "originale" simulée (plus grande)
    original_demo = cv2.resize(demo_img, (84, 84), interpolation=cv2.INTER_NEAREST)

    save_debug_images(
        original   = original_demo,
        normalized = demo_img,
        output_dir = DEBUG_DIR,
        prefix     = f"exemple_label_{true_label_demo}"
    )

    # ------------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ FINAL")
    print(f"{'='*60}")
    print(f"  Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"  Temps/image (batch): {results['time_per_image_ms']:.3f}ms")
    print(f"  Débit             : {results['throughput']:.0f} images/seconde")
    print(f"  Modèle sauvegardé : {MODEL_PATH}")
    print(f"  Debug             : {DEBUG_DIR}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Intégration avec le pipeline OCR existant
# ---------------------------------------------------------------------------

def predict_from_ocr_pipeline(digit_image_28x28: np.ndarray, model_path: str) -> int:
    """
    Fonction d'intégration : prédit un chiffre depuis une image 28x28
    produite par le pipeline OCR (warp + extraction + preprocessing).

    C'est le point d'entrée à appeler depuis main.py du projet OCR.

    Args:
        digit_image_28x28 : image 28x28, float32 [0,1] ou uint8
        model_path        : chemin vers le modèle SVM sauvegardé

    Returns:
        chiffre prédit (int)

    Exemple d'utilisation dans le pipeline OCR :
        from hog_svm.main import predict_from_ocr_pipeline
        label = predict_from_ocr_pipeline(digit_img, "models/svm_digits.joblib")
    """
    model = load_model(model_path)
    prediction, elapsed_ms = predict_digit_timed(digit_image_28x28, model)
    print(f"[OCR→HOG+SVM] Prédit : {prediction} ({elapsed_ms:.3f}ms)")
    return prediction


# ---------------------------------------------------------------------------
# Utilitaire interne pour la démo (pas nécessaire en production)
# ---------------------------------------------------------------------------

def _features_to_demo_image(features: np.ndarray) -> np.ndarray:
    """
    Crée une image 28x28 de démo depuis un vecteur de features HOG.
    
    Note : ceci est UNIQUEMENT pour la démo dans main.py.
    En production, charger directement les images 28x28.
    Les features HOG ne sont pas inversibles exactement.
    On utilise simplement une image aléatoire normalisée pour simuler.
    """
    # Normaliser les features entre 0 et 1, reshape en 28x28
    f = features - features.min()
    if f.max() > 0:
        f = f / f.max()
    
    # Créer une image 28x28 en répétant les valeurs disponibles
    dummy = np.zeros(28 * 28, dtype=np.float32)
    n = min(len(f), len(dummy))
    dummy[:n] = f[:n]
    return dummy.reshape(28, 28)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline HOG+SVM pour la reconnaissance de chiffres manuscrits"
    )
    parser.add_argument("--dataset", default=DATASET_DIR,
                        help=f"Dossier dataset (défaut: {DATASET_DIR})")
    parser.add_argument("--model",   default=MODEL_PATH,
                        help=f"Chemin modèle (défaut: {MODEL_PATH})")
    parser.add_argument("--debug",   default=DEBUG_DIR,
                        help=f"Dossier debug (défaut: {DEBUG_DIR})")
    parser.add_argument("--predict", default=None,
                        help="Chemin vers une image à prédire (mode inférence seule)")

    args = parser.parse_args()

    # Mode inférence seule sur une image
    if args.predict:
        print(f"\n[Mode inférence] Image : {args.predict}")
        image = cv2.imread(args.predict, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[ERREUR] Image introuvable : {args.predict}")
        else:
            model = load_model(args.model)
            prediction, elapsed_ms = predict_digit_timed(image, model)
            print(f"Chiffre reconnu : {prediction} ({elapsed_ms:.3f}ms)")

    # Mode entraînement complet
    else:
        DATASET_DIR = args.dataset
        MODEL_PATH  = args.model
        DEBUG_DIR   = args.debug
        main()
