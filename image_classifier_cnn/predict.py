# cnn_image_predict.py
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from rich import print

# ------------------------------
# Paths and device
# ------------------------------
cnn_weights_path = Path("trained_net.pth")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# CIFAR-10 class names
# ------------------------------
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# ------------------------------
# CNN Model
# ------------------------------
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------
# Image preprocessing
# ------------------------------
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(img).unsqueeze(0).to(device)  # (1,3,32,32)


# ------------------------------
# Load model
# ------------------------------
cnn_model = NeuralNet().to(device)
if cnn_weights_path.is_file():
    cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location=device))
    print("Loaded CNN weights.")
else:
    print("[red]CNN weights not found![/red]")
cnn_model.eval()


# ------------------------------
# Prediction with probabilities
# ------------------------------
def predict(img_path):
    img_tensor = preprocess_image(img_path)

    with torch.no_grad():
        output = cnn_model(img_tensor)
        probs = torch.softmax(output, dim=1)  # convert logits to probabilities
        top_prob, top_idx = torch.max(probs, 1)

    print(f"\n[bold green]Predictions for {img_path}[/bold green]")
    print(
        f"CNN predicts: {classes[top_idx.item()]} "
        f"({top_prob.item() * 100:.2f}% confidence)"
    )

    # Optional: show full probabilities
    print("\n[bold yellow]All class probabilities:[/bold yellow]")
    for i, c in enumerate(classes):
        print(f"{c}: {probs[0, i].item() * 100:.2f}%")


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    predict("dog.jpg")
