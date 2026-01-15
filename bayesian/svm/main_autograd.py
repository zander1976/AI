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

y = torch.tensor(
    [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    device=device,
)

# ============================================================
# Visualize Training Data
# ============================================================
X_cpu = X.cpu()
y_cpu = y.cpu()

plt.scatter(X_cpu[y_cpu == 1][:, 0], X_cpu[y_cpu == 1][:, 1], label="+1")
plt.scatter(X_cpu[y_cpu == -1][:, 0], X_cpu[y_cpu == -1][:, 1], label="-1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Training Data (Linearly Separable)")
plt.grid(True)
plt.show()

# ============================================================
# Linear SVM with Autograd
# ============================================================
# Parameters must have requires_grad=True so autograd tracks them
w = torch.randn(2, device=device, requires_grad=True)
w.data *= 0.1  # modify leaf tensor in place

# Bias term
b = torch.randn(1, device=device, requires_grad=True)
b.data *= 0.1  # modify leaf tensor in place

# Learning rate for gradient descent
lr = 0.01

# Soft-margin parameter (needed if you add hinge loss)
C = 1.0

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
for step in range(20):
    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    # Calculate all the point distances from the center to the boundary
    # Compute per-point margin violations: 1 - y_i * (w^T x_i + b)
    margins = 1 - y * (X @ w + b)

    # Compute average hinge loss across all points
    hinge_loss = torch.clamp(margins, min=0).mean()

    # Regularization term
    reg_loss = 0.5 * torch.dot(w, w)

    # Total loss
    loss = reg_loss + C * hinge_loss

    # --------------------------------------------------------
    # Backward pass (automatic differentiation)
    # --------------------------------------------------------
    loss.backward()

    # --------------------------------------------------------
    # Parameter update (manual SGD)
    # --------------------------------------------------------
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        # Important: clear gradients for next iteration
        w.grad.zero_()
        b.grad.zero_()

    # --------------------------------------------------------
    # Debugging / geometry
    # --------------------------------------------------------
    slope_w = w[1].item() / w[0].item()
    slope_h0 = -w[0].item() / w[1].item()

    print(
        f"Step {step}: loss={loss.item():.4f}, "
        f"w={w.detach().cpu().numpy()}, b={b.item():.4f}, "
        f"slope w={slope_w:.4f}, slope H0={slope_h0:.4f}"
    )

# ============================================================
# Plot Decision Boundary, Margins, and Weight Vector
# ============================================================
X_pos = X[y == 1].cpu().numpy()
X_neg = X[y == -1].cpu().numpy()

x_vals = np.linspace(
    X.cpu().numpy()[:, 0].min() - 1, X.cpu().numpy()[:, 0].max() + 1, 100
)


def compute_line(x, w, b, c):
    return (c - w[0] * x - b) / w[1]


w_vec = w.detach().cpu().numpy()
b_val = b.item()

y_h0 = compute_line(x_vals, w_vec, b_val, 0)
y_h1 = compute_line(x_vals, w_vec, b_val, 1)
y_h2 = compute_line(x_vals, w_vec, b_val, -1)

plt.scatter(X_pos[:, 0], X_pos[:, 1], color="blue", label="y=+1")
plt.scatter(X_neg[:, 0], X_neg[:, 1], color="red", label="y=-1")

plt.plot(x_vals, y_h0, "k-", label="H0: decision boundary")
plt.plot(x_vals, y_h1, "k--", label="H1: margin +1")
plt.plot(x_vals, y_h2, "k--", label="H2: margin -1")

# Plot w vector
plt.arrow(
    0,
    0,
    2.0 * w_vec[0],
    2.0 * w_vec[1],
    color="red",
    linewidth=2,
    head_width=0.2,
    head_length=0.2,
)
plt.plot([], [], color="red", linewidth=2, label="w vector")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("SVM Decision Boundary and Margins (Autograd)")
plt.legend()
plt.grid(True)
plt.show()
