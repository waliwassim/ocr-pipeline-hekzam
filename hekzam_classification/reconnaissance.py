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



transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = STN_LeNet().to(device)
model.load_state_dict(torch.load("stn_hekzam.pth", map_location=device))
model.eval()
print("Modèle Hekzam chargé !")



def reconnaitre_chiffre(chemin_image):
    image = Image.open(chemin_image)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        proba  = torch.softmax(output, dim=1)
        chiffre   = torch.argmax(proba).item()
        confiance = proba[0][chiffre].item() * 100
    return chiffre, confiance



def reconnaitre_dossier(dossier):
    print(f"\nReconnaissance des chiffres dans : {dossier}")
    print("-" * 40)
    for fichier in sorted(os.listdir(dossier)):
        if fichier.endswith(('.png', '.jpg', '.jpeg')):
            chemin = os.path.join(dossier, fichier)
            chiffre, confiance = reconnaitre_chiffre(chemin)
            print(f"{fichier} → Chiffre : {chiffre}  (confiance : {confiance:.1f}%)")
