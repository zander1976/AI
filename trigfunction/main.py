import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console

console = Console()

# --- Device setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
console.print(f"[bold green]Using device:[/bold green] {device}")

# --- Generate training data ---
N = 10000  # more points for better edge behavior
x_train = torch.linspace(-2 * torch.pi, 2 * torch.pi, N).unsqueeze(1)  # shape [N,1]
y_train = torch.sin(5 * x_train) + torch.cos(3 * x_train)

# Move to GPU
x_train = x_train.to(device)
y_train = y_train.to(device)


# --- Define neural network with tanh activations ---
class TrigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 256)
        self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(256, 256)
        self.act2 = nn.Tanh()
        self.hidden3 = nn.Linear(256, 128)
        self.act3 = nn.Tanh()
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x


model = TrigNet().to(device)

# --- Training setup ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100000  # train longer for smooth fit

console.print("[bold blue]Starting training...[/bold blue]")

# --- Training loop ---
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        console.print(
            f"[cyan]Epoch {epoch}[/cyan] | [yellow]Loss[/yellow]: {loss.item():.6f}"
        )

console.print(
    f"[bold green]Training finished[/bold green] | Final loss: {loss.item():.6f}"
)

# --- Evaluate predictions ---
model.eval()
with torch.no_grad():
    y_pred_train = model(x_train).cpu()
    x_train_cpu = x_train.cpu()
    y_train_cpu = y_train.cpu()

# --- Plot true vs learned output ---
plt.figure(figsize=(10, 5))
plt.plot(x_train_cpu, y_train_cpu, label="True function", color="blue")
plt.plot(
    x_train_cpu, y_pred_train, label="Network prediction", color="red", linestyle="--"
)
plt.title("Neural Network Learning y = sin(5x) + cos(3x) with Tanh")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
