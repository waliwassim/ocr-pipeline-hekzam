import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np


class STN_LeNet(nn.Module):
    def __init__(self):
        super(STN_LeNet, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float
        ))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x
    def forward(self, x):
        x = self.stn(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STN_LeNet().to(device)
model.load_state_dict(torch.load("stn_lenet_mnist.pth", map_location=device))
model.eval()


def reconnaitre_chiffre(chemin_image):
    image = Image.open(chemin_image)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        proba = torch.softmax(output, dim=1)
        chiffre = torch.argmax(proba).item()
        confiance = proba[0][chiffre].item() * 100
    return chiffre, confiance


def afficher_matrice_confusion(y_vrai, y_pred):
    """
    Affiche la matrice de confusion 10×10 et l'accuracy en console.
    y_vrai / y_pred : listes d'entiers 0-9.
    """
    classes = list(range(10))
    n = len(classes)

    # Matrice de confusion
    matrice = np.zeros((n, n), dtype=int)
    for vrai, pred in zip(y_vrai, y_pred):
        matrice[vrai][pred] += 1

    accuracy = np.trace(matrice) / matrice.sum() * 100 if matrice.sum() > 0 else 0.0

    print("\n" + "=" * 60)
    print("  MATRICE DE CONFUSION")
    print("=" * 60)

    # En-tête colonnes (prédictions)
    header = "       " + "  ".join(f"P{c}" for c in classes)
    print(header)
    print("       " + "───" * n)

    for i in classes:
        ligne = f"  R{i} │ " + "  ".join(
            f"\033[92m{matrice[i][j]:2d}\033[0m" if i == j else f"{matrice[i][j]:2d}"
            for j in classes
        )
        print(ligne)

    print("=" * 60)
    print(f"  Accuracy : {accuracy:.2f}%  ({np.trace(matrice)}/{matrice.sum()} bien classés)")
    print("=" * 60 + "\n")

    return matrice, accuracy


def reconnaitre_dossier(dossier, labels_json=None):
    """
    Parcourt le dossier, reconnaît chaque chiffre, puis affiche la matrice
    de confusion et l'accuracy si des labels ground-truth sont disponibles.

    labels_json (optionnel) : chemin vers un fichier JSON de la forme
        {"c1": 3, "c2": 7, ...}   (id de case → chiffre attendu)
    Si absent, cherche automatiquement 'labels.json' dans le même dossier.
    """
    print(f"\nReconnaissance des chiffres dans : {dossier}")
    print("-" * 40)

    # Chargement des labels ground-truth (optionnel)
    if labels_json is None:
        chemin_labels = os.path.join(dossier, "labels.json")
        labels_json = chemin_labels if os.path.exists(chemin_labels) else None

    labels = {}
    if labels_json and os.path.exists(labels_json):
        with open(labels_json, "r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"  Labels ground-truth chargés : {len(labels)} entrées ({labels_json})")
    else:
        print("  (Pas de labels.json — matrice de confusion non disponible)")

    predictions = {}   # id_case → chiffre_prédit

    for fichier in sorted(os.listdir(dossier)):
        if not fichier.endswith(('.png', '.jpg', '.jpeg')):
            continue

        chemin = os.path.join(dossier, fichier)

        # Ignorer les images quasi vides
        img = Image.open(chemin).convert("L")
        if sum(img.getdata()) < 50:
            print(f"{fichier} → image vide ignorée")
            continue

        chiffre, confiance = reconnaitre_chiffre(chemin)

        # Extraire l'id de case depuis le nom de fichier (ex: case_c12.png → c12)
        nom_sans_ext = os.path.splitext(fichier)[0]
        case_id = nom_sans_ext.replace("case_", "", 1)
        predictions[case_id] = chiffre

        # Indicateur ✓/✗ si on a le label
        indicateur = ""
        if case_id in labels:
            indicateur = " ✓" if chiffre == labels[case_id] else f" ✗ (attendu: {labels[case_id]})"

        print(f"{fichier} → Chiffre reconnu : {chiffre}  (confiance : {confiance:.1f}%){indicateur}")

    # ── Matrice de confusion et accuracy ──
    if labels:
        ids_communs = [k for k in predictions if k in labels]
        if ids_communs:
            y_vrai = [labels[k]      for k in ids_communs]
            y_pred = [predictions[k] for k in ids_communs]
            afficher_matrice_confusion(y_vrai, y_pred)
        else:
            print("\n  Aucune correspondance entre les ids prédits et les labels — vérifiez labels.json.")
    else:
        print(f"\n  {len(predictions)} case(s) traitée(s). Fournissez labels.json pour la matrice de confusion.\n")

    return predictions


reconnaitre_dossier("results/1e-r-0/cases_hekzam")