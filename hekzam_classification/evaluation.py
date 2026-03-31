import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# --- Définition des deux modèles ---
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

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


# --- Chargement MNIST ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data   = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Fonction de prédiction ---
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# --- Fonction matrice de confusion ---
def plot_confusion_matrix(labels, preds, title):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Matrice sauvegardée : {title.replace(' ', '_')}.png")
    plt.show()


# --- Charger les modèles ---
model_cnn = LeNet5().to(device)
model_cnn.load_state_dict(torch.load("lenet5_mnist.pth", map_location=device))

model_stn = STN_LeNet().to(device)
model_stn.load_state_dict(torch.load("stn_lenet_mnist.pth", map_location=device))


# --- Évaluation CNN seul ---
print("=== CNN seul (LeNet-5) ===")
labels_cnn, preds_cnn = get_predictions(model_cnn, test_loader)
print(classification_report(labels_cnn, preds_cnn, digits=4))
plot_confusion_matrix(labels_cnn, preds_cnn, "CNN seul LeNet5")


# --- Évaluation CNN + STN ---
print("=== CNN + STN ===")
labels_stn, preds_stn = get_predictions(model_stn, test_loader)
print(classification_report(labels_stn, preds_stn, digits=4))
plot_confusion_matrix(labels_stn, preds_stn, "CNN + STN")


# --- Comparaison finale ---
acc_cnn = (labels_cnn == preds_cnn).mean() * 100
acc_stn = (labels_stn == preds_stn).mean() * 100

print("\n========== COMPARAISON FINALE ==========")
print(f"CNN seul  : {acc_cnn:.2f}%")
print(f"CNN + STN : {acc_stn:.2f}%")
print(f"Différence : {acc_stn - acc_cnn:+.2f}%")
