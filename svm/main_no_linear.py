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
# Generate Circular / Overlapping Training Data
# ============================================================
np.random.seed(42)
num_inner = 50
num_outer = 100

# Inner circle: class +1
r_inner = 0.5 * np.random.rand(num_inner)
theta_inner = 2 * np.pi * np.random.rand(num_inner)
X_inner = np.c_[r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]
y_inner = np.ones(num_inner)

# Outer circle: class -1
r_outer = 1.0 + 0.5 * np.random.rand(num_outer)
theta_outer = 2 * np.pi * np.random.rand(num_outer)
X_outer = np.c_[r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)]
y_outer = -np.ones(num_outer)

# Combine data
X_np = np.vstack([X_inner, X_outer])
y_np = np.hstack([y_inner, y_outer])

# Convert to torch tensors
X = torch.tensor(X_np, dtype=torch.float32, device=device)
y = torch.tensor(y_np, dtype=torch.float32, device=device)

# ============================================================
# Visualize raw 2D data
# ============================================================
plt.figure(figsize=(6, 6))
plt.scatter(
    X_np[y_np == 1][:, 0], X_np[y_np == 1][:, 1], color="blue", label="+1 (inner)"
)
plt.scatter(
    X_np[y_np == -1][:, 0], X_np[y_np == -1][:, 1], color="red", label="-1 (outer)"
)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Circular Data (2D)")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# Lift data into 3D (kernel trick simulation)
# z = x1^2 + x2^2
# ============================================================
z = X[:, 0] ** 2 + X[:, 1] ** 2
X_3d = torch.stack([X[:, 0], X[:, 1], z], dim=1)

# ============================================================
# Visualize 3D data
# ============================================================
X_3d_np = X_3d.cpu().numpy()
y_np_cpu = y.cpu().numpy()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    X_3d_np[y_np_cpu == 1, 0],
    X_3d_np[y_np_cpu == 1, 1],
    X_3d_np[y_np_cpu == 1, 2],
    color="blue",
    label="+1 (inner)",
)
ax.scatter(
    X_3d_np[y_np_cpu == -1, 0],
    X_3d_np[y_np_cpu == -1, 1],
    X_3d_np[y_np_cpu == -1, 2],
    color="red",
    label="-1 (outer)",
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z = x1^2 + x2^2")
ax.set_title("3D Lifted Circular Data")
ax.legend()
plt.show()

# ============================================================
# Initialize SVM Parameters in 3D
# ============================================================
w = torch.randn(3, device=device, requires_grad=True)
w.data *= 0.1  # scale weights without breaking autograd

b = torch.randn(1, device=device, requires_grad=True)
b.data *= 0.1


lr = 0.01  # learning rate
C = 10.0  # soft-margin penalty
num_steps = 500

# ============================================================
# Training Loop (Autograd)
# ============================================================
for step in range(num_steps):
    # Forward pass: margins and hinge loss
    margins = 1 - y * (X_3d @ w + b)
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

    # Print progress every 50 steps
    if step % 50 == 0:
        print(
            f"Step {step}: loss={loss.item():.4f}, w={w.detach().cpu().numpy()}, b={b.item():.4f}"
        )

# ============================================================
# Visualize 3D Decision Plane
# ============================================================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot points
ax.scatter(
    X_3d_np[y_np_cpu == 1, 0],
    X_3d_np[y_np_cpu == 1, 1],
    X_3d_np[y_np_cpu == 1, 2],
    color="blue",
    label="+1 (inner)",
)
ax.scatter(
    X_3d_np[y_np_cpu == -1, 0],
    X_3d_np[y_np_cpu == -1, 1],
    X_3d_np[y_np_cpu == -1, 2],
    color="red",
    label="-1 (outer)",
)

# Create meshgrid for decision plane
xx, yy = np.meshgrid(
    np.linspace(X_3d_np[:, 0].min() - 0.5, X_3d_np[:, 0].max() + 0.5, 30),
    np.linspace(X_3d_np[:, 1].min() - 0.5, X_3d_np[:, 1].max() + 0.5, 30),
)
zz = (-w[0].item() * xx - w[1].item() * yy - b.item()) / w[2].item()

ax.plot_surface(xx, yy, zz, alpha=0.3, color="green", label="Decision Plane")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z = x1^2 + x2^2")
ax.set_title("Soft-Margin SVM Decision Plane in 3D")
ax.legend()
plt.show()

# ============================================================
# 2D Projection of Violations
# ============================================================
margins_vals = y * (X_3d @ w + b)
violating = margins_vals < 1

plt.figure(figsize=(6, 6))

# Safe points
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

# Violating points
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

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("2D Projection: SVM Violations (Soft-Margin)")
plt.legend()
plt.grid(True)
plt.show()
