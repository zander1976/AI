import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.table import Table

console = Console()

# --- Device setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
console.print(f"[bold green]Using device:[/bold green] {device}")

# --- XOR dataset ---
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# --- Repeat XOR to create a large batch to stress GPU ---
batch_size = 10000  # repeat 10k times
X_big = X.repeat(batch_size, 1).to(device)
Y_big = Y.repeat(batch_size, 1).to(device)


# --- Neural network definition ---
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 128)  # large hidden layer
        self.output = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


model = XORNet().to(device)

# --- Training setup ---
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# --- Train until loss threshold ---
target_loss = 0.001
epoch = 0
max_epochs = 10000  # safety cap

console.print("[bold blue]Starting training...[/bold blue]")

while True:
    optimizer.zero_grad()
    output = model(X_big)
    loss = criterion(output, Y_big)
    loss.backward()
    optimizer.step()

    epoch += 1
    if epoch % 10 == 0:
        console.print(
            f"[cyan]Epoch {epoch}[/cyan] | [yellow]Loss[/yellow]: {loss.item():.6f}"
        )

    if loss.item() < target_loss or epoch >= max_epochs:
        break

console.print(
    f"[bold green]Training stopped at epoch {epoch} with loss {loss.item():.6f}[/bold green]\n"
)

# --- Testing on original XOR inputs ---
X_test = X.to(device)
Y_test = Y.to(device)

with torch.no_grad():
    predictions = model(X_test)
    rounded = (predictions > 0.5).float()

    # Rich table for results
    table = Table(title="XOR Predictions")
    table.add_column("Input", style="cyan")
    table.add_column("Rounded Prediction", style="green")
    table.add_column("Raw Prediction", style="yellow")

    for i in range(len(X_test)):
        inp = X_test[i].cpu().numpy().tolist()  # convert tensor to list
        pred = int(rounded[i].item())  # rounded prediction as int
        raw = float(predictions[i].item())  # raw prediction as float
        table.add_row(str(inp), str(pred), f"{raw:.4f}")

    console.print(table)
