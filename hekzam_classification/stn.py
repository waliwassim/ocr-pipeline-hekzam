import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Architecture CNN + STN ---
class STN_LeNet(nn.Module):
    def __init__(self):
        super(STN_LeNet, self).__init__()

        # --- Module STN ---
        # Localization network : apprend à détecter la transformation
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )

        # Regression network : prédit les 6 paramètres de transformation
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 6 paramètres = matrice de transformation affine
        )

        # Initialisation importante : commencer par la transformation identité
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float
        ))

        # --- LeNet-5 classique ---
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
        # Étape 1 : extraire les features pour estimer la transformation
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)

        # Étape 2 : prédire les 6 paramètres
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Étape 3 : appliquer la transformation géométrique sur l'image
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, x):
        x = self.stn(x)         # redresser l'image d'abord
        x = self.conv_layers(x) # puis classifier
        x = self.fc_layers(x)
        return x


# --- Chargement MNIST ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)


# --- Entraînement ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = STN_LeNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.4f}")


# --- Évaluation ---
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Précision : {100 * correct / total:.2f}%")


# --- Lancer ---
print("=== Entraînement CNN + STN ===")
train(model, train_loader, epochs=5)
evaluate(model, test_loader)

torch.save(model.state_dict(), "stn_lenet_mnist.pth")
print("Modèle STN sauvegardé.")
