# Projet Hekzam — Pipeline OCR de chiffres manuscrits

## Structure du projet

```
TER/
├── PDF_INPUT/                  ← Dossier des PDFs scannés à traiter
├── correction_distorsion.py    ← Étape 1 : suppression grille + redressement
├── detection_coins.py          ← Étape 2 : détection des cases et coins
├── convertir_json.py           ← Étape 3 : conversion format JSON
├── pipeline_ocr.py             ← Étape 4 : extraction et normalisation des cases
├── entrainement_hekzam.py      ← Fine-tuning du modèle STN-LeNet
├── reconnaissance.py           ← Étape 5 : reconnaissance + matrice de confusion
├── run_pipeline.py             ← Script maître (traite tout le dossier PDF_INPUT/)
├── stn_lenet_mnist.pth         ← Modèle pré-entraîné sur MNIST
└── stn_hekzam.pth              ← Modèle fine-tuné sur données Hekzam
```

## Installation

```bash
pip install opencv-python numpy pillow torch torchvision pdf2image
# Linux : sudo apt install poppler-utils
# macOS : brew install poppler
```

## Utilisation

### Option 1 — Traiter tout le dossier PDF_INPUT/ (recommandé)

```bash
python run_pipeline.py --pdfs PDF_INPUT/ --json atomic-boxes.json
```

### Option 2 — Traiter un seul PDF manuellement

```bash
# Étape 1 — Correction distorsion
python correction_distorsion.py --pdf PDF_INPUT/scan.pdf --json atomic-boxes.json

# Étape 2 — Détection des coins
python detection_coins.py --input scans_corriges/scan/ --output results_coins/scan/

# Étape 3 — Conversion JSON
python convertir_json.py --input results_coins/scan/page_001.json --output results/scan/cases.json

# Étape 4 — Pipeline OCR
python pipeline_ocr.py --pdf PDF_INPUT/scan.pdf --json results/scan/cases.json

# Étape 5 — Reconnaissance
python reconnaissance.py --pdf PDF_INPUT/scan.pdf
```

### Fine-tuning du modèle (optionnel)

```bash
python entrainement_hekzam.py --dossier results/scan/cases_hekzam
```

## Chaîne de traitement

```
PDF scanné
    ↓
correction_distorsion.py   → PNG corrigés (grille supprimée)
    ↓
detection_coins.py         → JSON des coins de chaque case
    ↓
convertir_json.py          → JSON format pipeline
    ↓
pipeline_ocr.py            → PNG 28×28 normalisés par case
    ↓
reconnaissance.py          → chiffre reconnu + matrice de confusion
```
