import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import print

# ============================================================
# Device Selection
# ============================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Generate Mixed / Overlapping Training Data
# ============================================================
np.random.seed(42)
num_points = 50

# Slightly overlapping clusters
X_pos = np.random.randn(num_points, 2) * 0.9 + np.array([2, 2])
X_neg = np.random.randn(num_points, 2) * 0.9 + np.array([0, 0])

X_np = np.vstack([X_pos, X_neg])
y_np = np.hstack([np.ones(num_points), -np.ones(num_points)])

# Convert to torch tensors
X = torch.tensor(X_np, dtype=torch.float32, device=device)
y = torch.tensor(y_np, dtype=torch.float32, device=device)

# ============================================================
# Visualize the raw data
# ============================================================
plt.scatter(X_np[y_np == 1][:, 0], X_np[y_np == 1][:, 1], label="+1")
plt.scatter(X_np[y_np == -1][:, 0], X_np[y_np == -1][:, 1], label="-1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Soft-Margin Example: Overlapping Classes")
plt.grid(True)
# plt.show()

# ============================================================
# Initialize SVM Parameters
# ============================================================
w = torch.randn(2, device=device, requires_grad=True)
w.data *= 0.1  # small initial weights
b = torch.randn(1, device=device, requires_grad=True)
b.data *= 0.1

lr = 0.01  # learning rate
C = 10.5  # soft-margin penalty
num_steps = 500

# ============================================================
# Training Loop (Autograd)
# ============================================================
for step in range(num_steps):
    # Forward pass
    margins = 1 - y * (X @ w + b)  # margin violations
    hinge_loss = torch.clamp(margins, min=0).mean()
    reg_loss = 0.5 * torch.dot(w, w)
    loss = reg_loss + C * hinge_loss

    # Backward pass
    loss.backward()

    # Update parameters manually
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    # Print every 50 steps
    if step % 50 == 0:
        print(
            f"Step {step}: loss={loss.item():.4f}, w={w.detach().cpu().numpy()}, b={b.item():.4f}"
        )

# ============================================================
# Plot Decision Boundary, Margins, Weight Vector, Violators
# ============================================================
margins_vals = y * (X @ w + b)
violating = margins_vals < 1
safe = margins_vals >= 1

# Points outside margin
plt.scatter(
    X[~violating & (y == 1)].cpu()[:, 0],
    X[~violating & (y == 1)].cpu()[:, 1],
    color="blue",
    label="y=+1, safe",
)
plt.scatter(
    X[~violating & (y == -1)].cpu()[:, 0],
    X[~violating & (y == -1)].cpu()[:, 1],
    color="red",
    label="y=-1, safe",
)
# Points violating margin
plt.scatter(
    X[violating & (y == 1)].cpu()[:, 0],
    X[violating & (y == 1)].cpu()[:, 1],
    color="cyan",
    edgecolor="black",
    s=80,
    label="y=+1, violation",
)
plt.scatter(
    X[violating & (y == -1)].cpu()[:, 0],
    X[violating & (y == -1)].cpu()[:, 1],
    color="orange",
    edgecolor="black",
    s=80,
    label="y=-1, violation",
)

# Decision boundary and margins
x_vals = np.linspace(X.cpu()[:, 0].min() - 1, X.cpu()[:, 0].max() + 1, 100)
w_vec = w.detach().cpu().numpy()
b_val = b.item()


def compute_line(x, w, b, c):
    return (c - w[0] * x - b) / w[1]


plt.plot(
    x_vals, compute_line(x_vals, w_vec, b_val, 0), "k-", label="H0: decision boundary"
)
plt.plot(x_vals, compute_line(x_vals, w_vec, b_val, 1), "k--", label="H1: margin +1")
plt.plot(x_vals, compute_line(x_vals, w_vec, b_val, -1), "k--", label="H2: margin -1")

# Weight vector
plt.arrow(
    0,
    0,
    2 * w_vec[0],
    2 * w_vec[1],
    color="red",
    linewidth=2,
    head_width=0.2,
    head_length=0.2,
)
plt.plot([], [], color="red", linewidth=2, label="w vector")  # for legend

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Soft-Margin SVM: Decision Boundary, Margins, Violations")
plt.legend()
plt.grid(True)
plt.show()
