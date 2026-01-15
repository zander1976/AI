# train_vit_cifar10.py
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from rich import print

# ------------------------------
# Paths
# ------------------------------
trained_weights = Path("trained_vit_cifar10.pth")

# ------------------------------
# Device
# ------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Transforms (stronger augmentation)
# ------------------------------
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
# Datasets + loaders
# ------------------------------
train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=transform_train, download=True
)
test_data = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transform_test, download=True
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)


# ------------------------------
# Patch embedding
# ------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ------------------------------
# Vision Transformer (small for CIFAR-10)
# ------------------------------
class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        num_classes=10,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed  # add positional embeddings
        x = self.transformer(x)  # (B, num_patches, embed_dim)
        x = x.mean(dim=1)  # global average pooling
        x = self.mlp_head(x)  # (B, num_classes)
        return x


# ------------------------------
# Instantiate model
# ------------------------------
model = ViT().to(device)

# Resume training if weights exist
best_acc = 0.0
if trained_weights.is_file():
    print("Loading trained weights...")
    model.load_state_dict(torch.load(trained_weights, map_location=device))

# ------------------------------
# Training setup
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# ------------------------------
# Training loop
# ------------------------------
num_epochs = 200  # more epochs for ViT
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    avg_loss = running_loss / len(train_loader)

    # ---- Evaluation ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}"
    )

    # Save best model
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), trained_weights)
        print(f"✅ New best model saved ({best_acc:.2f}%)")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")
