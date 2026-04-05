import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


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


# --- Prétraitement ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- Charger le modèle ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = STN_LeNet().to(device)
model.load_state_dict(torch.load("stn_hekzam.pth", map_location=device))
model.eval()
print("Modèle Hekzam chargé !")


# --- Reconnaître un chiffre ---
def reconnaitre_chiffre(chemin_image):
    image = Image.open(chemin_image)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        proba  = torch.softmax(output, dim=1)
        chiffre   = torch.argmax(proba).item()
        confiance = proba[0][chiffre].item() * 100
    return chiffre, confiance


# --- Reconnaître tout un dossier ---
def reconnaitre_dossier(dossier):
    """
    Reconnaît tous les chiffres du dossier, sauvegarde les prédictions
    dans results.json, et affiche la matrice de confusion + accuracy
    si labels.json est présent.
    """
    import json
    import numpy as np

    print(f"\nReconnaissance des chiffres dans : {dossier}")
    print("-" * 40)

    # ── Chargement des labels ground truth (optionnel) ──
    chemin_labels = os.path.join(dossier, "labels.json")
    labels = {}
    if os.path.exists(chemin_labels):
        with open(chemin_labels, "r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"  Labels ground-truth chargés : {len(labels)} entrées\n")
    else:
        print("  (Pas de labels.json — matrice de confusion non disponible)\n")

    # ── Reconnaissance ──
    predictions = {}   # case_id → chiffre_prédit
    confidences = {}   # case_id → confiance

    for fichier in sorted(os.listdir(dossier)):
        if not fichier.endswith(('.png', '.jpg', '.jpeg')):
            continue

        chemin = os.path.join(dossier, fichier)

        # Ignorer les images vides
        from PIL import Image as PILImage
        img = PILImage.open(chemin).convert("L")
        if sum(img.getdata()) < 50:
            print(f"  {fichier} → image vide ignorée")
            continue

        chiffre, confiance = reconnaitre_chiffre(chemin)

        # Extraire l'id de case depuis le nom (case_c12.png → c12)
        nom_sans_ext = os.path.splitext(fichier)[0]
        case_id = nom_sans_ext.replace("case_", "", 1)
        predictions[case_id] = chiffre
        confidences[case_id] = confiance

        # Indicateur ✓/✗ si label connu
        indicateur = ""
        if case_id in labels:
            indicateur = " ✓" if chiffre == labels[case_id] else f" ✗ (attendu: {labels[case_id]})"

        print(f"  {fichier} → {chiffre}  ({confiance:.1f}%){indicateur}")

    # ── Sauvegarde dans results.json ──
    chemin_results = os.path.join(os.path.dirname(dossier), "results.json")
    resultats_json = []
    if os.path.exists(chemin_results):
        with open(chemin_results, "r", encoding="utf-8") as f:
            resultats_json = json.load(f)

    # Enrichir chaque entrée avec chiffre_predit, confiance et label_reel
    for r in resultats_json:
        case_id = r["id"].replace("case_", "", 1) if r["id"].startswith("case_") else r["id"]
        pred = predictions.get(case_id) or predictions.get(r["id"])
        conf = confidences.get(case_id) or confidences.get(r["id"])
        if pred is not None:
            r["chiffre_predit"] = pred
            r["confiance"]      = round(conf, 1)
        # Ajouter le label réel si disponible
        if case_id in labels:
            r["label_reel"] = labels[case_id]

    # ── Calculer accuracy globale et la stocker dans le premier élément ──
    ids_communs = [k for k in predictions if k in labels]
    accuracy = None
    if ids_communs:
        correct = sum(1 for k in ids_communs if predictions[k] == labels[k])
        accuracy = round(correct / len(ids_communs) * 100, 2)
        # Stocker l'accuracy dans results.json (sur le premier élément)
        if resultats_json:
            resultats_json[0]["accuracy"] = accuracy

    with open(chemin_results, "w", encoding="utf-8") as f:
        json.dump(resultats_json, f, ensure_ascii=False, indent=2)
    print(f"\n  Prédictions sauvegardées → {chemin_results}")

    # ── Matrice de confusion + accuracy ──
    if labels:
        ids_communs = [k for k in predictions if k in labels]
        if ids_communs:
            y_vrai = [labels[k]      for k in ids_communs]
            y_pred = [predictions[k] for k in ids_communs]

            n = 10
            matrice = np.zeros((n, n), dtype=int)
            for vrai, pred in zip(y_vrai, y_pred):
                if 0 <= vrai < n and 0 <= pred < n:
                    matrice[vrai][pred] += 1

            accuracy = np.trace(matrice) / matrice.sum() * 100 if matrice.sum() > 0 else 0.0

            print(f"\n{'='*60}")
            print(f"  MATRICE DE CONFUSION")
            print(f"{'='*60}")
            header = "       " + "  ".join(f"P{c}" for c in range(n))
            print(header)
            print("       " + "───" * n)
            for i in range(n):
                ligne = f"  R{i} │ " + "  ".join(
                    f"\033[92m{matrice[i][j]:2d}\033[0m" if i == j else f"{matrice[i][j]:2d}"
                    for j in range(n)
                )
                print(ligne)
            print(f"{'='*60}")
            print(f"  Accuracy : {accuracy:.2f}%  ({int(np.trace(matrice))}/{int(matrice.sum())} bien classés)")
            print(f"{'='*60}\n")
        else:
            print("\n  Aucune correspondance ids prédits / labels.")
    else:
        print(f"\n  {len(predictions)} case(s) traitée(s). Ajoutez labels.json pour la matrice.\n")

    return predictions


# --- LANCER ---
import argparse
import os

parser = argparse.ArgumentParser(description="Reconnaissance STN-LeNet — Projet Hekzam")
parser.add_argument("--dossier", default=None,
                    help="Dossier des cases PNG (ex: results/2e-r-0/cases_hekzam)")
parser.add_argument("--pdf",     default=None,
                    help="Nom du PDF traité (ex: 2e-r-0) pour construire le chemin automatiquement")
args = parser.parse_args()

if args.dossier:
    dossier = args.dossier
elif args.pdf:
    pdf_name = os.path.splitext(os.path.basename(args.pdf))[0]
    dossier  = os.path.join("results", pdf_name, "cases_hekzam")
else:
    # Chemin par défaut : cherche le dossier results/ le plus récent
    dossier = "results/2e-r-0/cases_hekzam"
    print(f"[INFO] Aucun argument fourni — dossier par défaut : {dossier}")
    print(f"[INFO] Usage : python reconnaissance.py --pdf 2e-r-0.pdf")
    print(f"[INFO]      ou: python reconnaissance.py --dossier results/2e-r-0/cases_hekzam\n")

reconnaitre_dossier(dossier)
