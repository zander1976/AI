from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from rich import print

trained_weights = Path("trained_net.pth")

# ------------------------------
# Device
# ------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Transforms
# ------------------------------
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# ------------------------------
# Datasets
# ------------------------------
train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=transform_train, download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transform_test, download=True
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


# ------------------------------
# Model
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


net = NeuralNet().to(device)

# ------------------------------
# Resume if possible
# ------------------------------
best_acc = 0.0
if trained_weights.is_file():
    print("Loading trained weights...")
    net.load_state_dict(torch.load(trained_weights, map_location=device))

# ------------------------------
# Training setup
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# ------------------------------
# Training loop
# ------------------------------
num_epochs = 100

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    avg_loss = running_loss / len(train_loader)

    # ---- Evaluation ----
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | LR: {scheduler.get_last_lr()[0]:.4f}"
    )

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(net.state_dict(), trained_weights)
        print(f"✅ New best model saved ({best_acc:.2f}%)")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")
