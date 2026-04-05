# batch_pipeline.py
# Lance le pipeline OCR sur tous les PDFs d'un dossier.
#
# Utilisation :
#   python3 batch_pipeline.py \
#       --input  ../dossier_pdfs \
#       --json   ../atomic-boxes.json \
#       --model  ../hog_svm/models/svm_retrained.joblib \
#       --output ../results_batch

import argparse
import time
import csv
import sys
from pathlib import Path

# Importer le pipeline existant
sys.path.append(str(Path(__file__).parent))
from run_pipeline import run_pipeline


def run_batch(
    input_dir:  str,
    json_path:  str,
    model_path: str,
    output_dir: str,
    dpi:        int   = 300,
    margin:     float = 0.15,
    save_debug: bool  = False,   # désactivé par défaut en batch pour aller plus vite
    save_crops: bool  = True,
):
    """
    Lance le pipeline sur tous les PDFs d'un dossier.

    Args:
        input_dir  : dossier contenant les fichiers PDF
        json_path  : chemin vers atomic-boxes.json
        model_path : chemin vers le modèle SVM (.joblib)
        output_dir : dossier de sortie (un sous-dossier par PDF)
        dpi        : résolution de conversion PDF
        margin     : marge autour des cases
        save_debug : sauvegarder les images de debug (lent)
        save_crops : sauvegarder les crops extraits
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Lister tous les PDFs du dossier
    pdf_files = sorted(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"[ERREUR] Aucun fichier PDF trouvé dans : {input_dir}")
        return

    print("\n" + "="*60)
    print(f"  BATCH OCR — {len(pdf_files)} fichier(s) PDF trouvé(s)")
    print("="*60)
    for p in pdf_files:
        print(f"  - {p.name}")
    print()

    # Résumé global
    batch_summary = []
    t_batch_start = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_name   = pdf_path.stem   # nom sans extension
        pdf_output = output_path / pdf_name

        print(f"\n[{i}/{len(pdf_files)}] Traitement : {pdf_path.name}")
        print("-" * 50)

        t_start = time.time()
        try:
            results = run_pipeline(
                pdf_path   = str(pdf_path),
                json_path  = json_path,
                model_path = model_path,
                output_dir = str(pdf_output),
                dpi        = dpi,
                margin     = margin,
                save_debug = save_debug,
                save_crops = save_crops,
            )

            # Calculer les stats pour ce PDF
            total   = len(results)
            labeled = [r for r in results if r["correct"] is not None]
            correct = [r for r in labeled  if r["correct"]]
            acc     = len(correct) / len(labeled) * 100 if labeled else 0
            elapsed = time.time() - t_start

            batch_summary.append({
                "fichier":   pdf_path.name,
                "total":     total,
                "correct":   len(correct),
                "incorrect": len(labeled) - len(correct),
                "accuracy":  f"{acc:.1f}%",
                "temps_s":   f"{elapsed:.1f}",
                "statut":    "OK"
            })

            print(f"  → {len(correct)}/{total} correctes ({acc:.1f}%) en {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t_start
            print(f"  [ERREUR] {pdf_path.name} : {e}")
            batch_summary.append({
                "fichier":   pdf_path.name,
                "total":     0,
                "correct":   0,
                "incorrect": 0,
                "accuracy":  "N/A",
                "temps_s":   f"{elapsed:.1f}",
                "statut":    f"ERREUR: {e}"
            })

    # ------------------------------------------------------------------
    # Rapport global
    # ------------------------------------------------------------------
    t_total = time.time() - t_batch_start

    print("\n" + "="*60)
    print("  RAPPORT BATCH FINAL")
    print("="*60)
    print(f"  {'Fichier':<30} {'Cases':>6} {'OK':>5} {'Acc':>8} {'Temps':>7}")
    print("-" * 60)

    for s in batch_summary:
        status = "✓" if s["statut"] == "OK" else "✗"
        print(f"  {status} {s['fichier']:<28} {s['total']:>6} {s['correct']:>5} "
              f"{s['accuracy']:>8} {s['temps_s']:>6}s")

    # Totaux
    total_cases   = sum(s["total"]   for s in batch_summary if s["statut"] == "OK")
    total_correct = sum(s["correct"] for s in batch_summary if s["statut"] == "OK")
    global_acc    = total_correct / total_cases * 100 if total_cases > 0 else 0
    errors        = sum(1 for s in batch_summary if s["statut"] != "OK")

    print("-" * 60)
    print(f"  TOTAL : {total_cases} cases, {total_correct} correctes → {global_acc:.1f}%")
    print(f"  PDFs traités : {len(pdf_files) - errors}/{len(pdf_files)}")
    print(f"  Temps total  : {t_total:.1f}s")
    print("="*60)

    # Sauvegarde du rapport CSV global
    summary_path = output_path / "rapport_batch.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fichier","total","correct","incorrect",
                                               "accuracy","temps_s","statut"])
        writer.writeheader()
        writer.writerows(batch_summary)

    print(f"\n  Rapport global → {summary_path.resolve()}\n")


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lance le pipeline OCR sur tous les PDFs d'un dossier"
    )
    parser.add_argument("--input",    required=True,
                        help="Dossier contenant les PDFs à traiter")
    parser.add_argument("--json",     required=True,
                        help="Fichier atomic-boxes.json")
    parser.add_argument("--model",    required=True,
                        help="Modèle SVM (.joblib)")
    parser.add_argument("--output",   default="results_batch",
                        help="Dossier de sortie (défaut: results_batch)")
    parser.add_argument("--dpi",      type=int,   default=300,
                        help="DPI de conversion PDF (défaut: 300)")
    parser.add_argument("--margin",   type=float, default=0.15,
                        help="Marge autour des cases (défaut: 0.15)")
    parser.add_argument("--debug",    action="store_true",
                        help="Sauvegarder les images de debug (plus lent)")
    parser.add_argument("--no-crops", action="store_true",
                        help="Ne pas sauvegarder les crops")

    args = parser.parse_args()

    run_batch(
        input_dir  = args.input,
        json_path  = args.json,
        model_path = args.model,
        output_dir = args.output,
        dpi        = args.dpi,
        margin     = args.margin,
        save_debug = args.debug,
        save_crops = not args.no_crops,
    )
