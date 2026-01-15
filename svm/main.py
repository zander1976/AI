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
# Training Data (Linearly Separable)
# ============================================================
X = torch.tensor(
    [
        [2.0, 2.0],
        [3.0, 1.0],
        [2.0, 3.0],
        [1.0, 2.0],  # +1 class
        [-2.0, -2.0],
        [-3.0, -1.0],
        [-2.0, -3.0],
        [-1.0, -2.0],  # -1 class
    ],
    device=device,
)
y = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], device=device)

# ============================================================
# Visualize Training Data
# ============================================================
X_cpu = X.detach().cpu()
y_cpu = y.detach().cpu()

plt.scatter(X_cpu[y_cpu == 1][:, 0], X_cpu[y_cpu == 1][:, 1], label="+1")
plt.scatter(X_cpu[y_cpu == -1][:, 0], X_cpu[y_cpu == -1][:, 1], label="-1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Training Data (Linearly Separable)")
plt.grid(True)
# plt.show()

# ============================================================
# Initialize SVM parameters
# ============================================================
w = torch.randn(2, device=device) * 0.1  # weight vector
b = torch.randn(1, device=device) * 0.1  # bias
lr = 0.1  # learning rate
C = 0.1  # soft-margin parameter
num_steps = 500  # number of gradient descent steps

# ============================================================
# Training loop (manual gradient descent)
# ============================================================
for step in range(num_steps):
    # Compute margins: y_i * (w · x_i + b)
    margins = y * (X @ w + b)

    # Compute hinge loss
    hinge_loss = torch.clamp(1 - margins, min=0)

    # Total loss = regularization + hinge loss
    loss = 0.5 * torch.dot(w, w) + C * torch.sum(hinge_loss)

    # Vectorized gradient calculation
    mask = margins < 1  # points violating the margin
    grad_w = w - C * torch.sum((y[mask].unsqueeze(1) * X[mask]), dim=0)
    grad_b = -C * torch.sum(y[mask])

    # Gradient descent update
    w = w - lr * grad_w
    b = b - lr * grad_b

    # Print every 50 steps
    if step % 50 == 0:
        print(
            f"Step {step}: loss={loss.item():.4f}, w={w.cpu().numpy()}, b={b.item():.4f}"
        )

# ============================================================
# Plot Decision Boundary, Margins, and Weight Vector
# ============================================================
X_pos = X[y == 1].cpu().numpy()
X_neg = X[y == -1].cpu().numpy()
x_vals = np.linspace(
    X.cpu().numpy()[:, 0].min() - 1, X.cpu().numpy()[:, 0].max() + 1, 100
)


# Helper to compute y-values of a line: w0*x + w1*y + b = c → y = (c - w0*x - b) / w1
def compute_line(x, w, b, c):
    return (c - w[0] * x - b) / w[1]


w_vec = w.cpu().numpy()
b_val = b.item()

y_h0 = compute_line(x_vals, w_vec, b_val, 0)  # decision boundary
y_h1 = compute_line(x_vals, w_vec, b_val, 1)  # margin +1
y_h2 = compute_line(x_vals, w_vec, b_val, -1)  # margin -1

plt.scatter(X_pos[:, 0], X_pos[:, 1], color="blue", label="y=+1")
plt.scatter(X_neg[:, 0], X_neg[:, 1], color="red", label="y=-1")

plt.plot(x_vals, y_h0, "k-", label="H0: decision boundary")
plt.plot(x_vals, y_h1, "k--", label="H1: margin +1")
plt.plot(x_vals, y_h2, "k--", label="H2: margin -1")

# Plot weight vector w from origin
scale_factor = 2.0
plt.arrow(
    0,
    0,
    scale_factor * w_vec[0],
    scale_factor * w_vec[1],
    color="red",
    linewidth=2,
    head_width=0.2,
    head_length=0.2,
)
plt.plot([], [], color="red", linewidth=2, label="w vector")  # for legend

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("SVM Decision Boundary and Margins with Weight Vector")
plt.legend()
plt.grid(True)
plt.show()
