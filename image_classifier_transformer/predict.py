# image_predict_vit.py
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from rich import print

# ------------------------------
# Paths and device
# ------------------------------
vit_weights_path = Path("trained_vit_cifar10.pth")  # matches training
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
# Preprocess image
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
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device)


# ------------------------------
# Load ViT model
# ------------------------------
vit_model = ViT().to(device)
if vit_weights_path.is_file():
    vit_model.load_state_dict(torch.load(vit_weights_path, map_location=device))
    print("Loaded ViT weights.")
vit_model.eval()


# ------------------------------
# Predict function with top-3
# ------------------------------
def predict(img_path, topk=3):
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        output = vit_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=topk, dim=1)

    print(f"\n[bold green]Predictions for {img_path}[/bold green]")
    for i in range(topk):
        print(
            f"{i + 1}: {classes[top_idxs[0, i].item()]} ({top_probs[0, i].item() * 100:.2f}%)"
        )

    print("\n[bold yellow]All class probabilities:[/bold yellow]")
    for i, c in enumerate(classes):
        print(f"{c}: {probs[0, i].item() * 100:.2f}%")


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    predict("dog.jpg")
