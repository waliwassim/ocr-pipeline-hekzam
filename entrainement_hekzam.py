import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os

class HekzamDataset(Dataset):
    def __init__(self, dossier, labels_json, transform=None):
        self.dossier = dossier
        self.transform = transform
        with open(labels_json, 'r') as f:
            labels = json.load(f)
        self.samples = []
        for nom, chiffre in labels.items():
            chemin = os.path.join(dossier, f"case_{nom}.png")
            if os.path.exists(chemin):
                self.samples.append((chemin, chiffre))
        print(f"Images chargées : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chemin, label = self.samples[idx]
        image = Image.open(chemin).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label


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



transform_train = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

import argparse
import os

parser = argparse.ArgumentParser(description="Fine-tuning STN-LeNet sur données Hekzam")
parser.add_argument("--dossier", default="cases_hekzam",
                    help="Dossier des cases PNG (défaut: cases_hekzam)")
parser.add_argument("--labels",  default=None,
                    help="Chemin vers labels.json (défaut: <dossier>/labels.json)")
parser.add_argument("--modele_entree", default="stn_lenet_mnist.pth",
                    help="Modèle MNIST de départ (défaut: stn_lenet_mnist.pth)")
parser.add_argument("--modele_sortie", default="stn_hekzam.pth",
                    help="Modèle fine-tuné à sauvegarder (défaut: stn_hekzam.pth)")
parser.add_argument("--epochs",  type=int, default=30,
                    help="Nombre d'époques (défaut: 30)")
args = parser.parse_args()

dossier_cases = args.dossier
labels_json   = args.labels if args.labels else os.path.join(dossier_cases, "labels.json")

# --- Chargement données ---
dataset_train = HekzamDataset(dossier_cases, labels_json, transform=transform_train)
dataset_test  = HekzamDataset(dossier_cases, labels_json, transform=transform_test)

train_size = int(0.8 * len(dataset_train))
test_size  = len(dataset_train) - train_size

train_data, _ = torch.utils.data.random_split(dataset_train, [train_size, test_size])
_, test_data  = torch.utils.data.random_split(dataset_test,  [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=16, shuffle=False)

# --- Charger le modèle MNIST et faire le fine-tuning ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = STN_LeNet().to(device)

# Charger les poids MNIST
model.load_state_dict(torch.load(args.modele_entree, map_location=device))
print(f"Modèle chargé : {args.modele_entree} — début du fine-tuning sur Hekzam")

# Learning rate faible pour fine-tuning
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{args.epochs} — Loss: {total_loss/len(train_loader):.4f}")


model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nPrécision sur images Hekzam : {100*correct/total:.2f}%")

torch.save(model.state_dict(), args.modele_sortie)
print(f"Modèle sauvegardé : {args.modele_sortie}")
