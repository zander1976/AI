import matplotlib.cm as cm
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
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=device
)  # Bernoulli labels

# ============================================================
# Visualize Training Data
# ============================================================
X_cpu = X.detach().cpu()
y_cpu = y.detach().cpu()

plt.figure(figsize=(6, 6))
plt.scatter(X_cpu[y_cpu == 1][:, 0], X_cpu[y_cpu == 1][:, 1], color="blue", label="y=1")
plt.scatter(X_cpu[y_cpu == 0][:, 0], X_cpu[y_cpu == 0][:, 1], color="red", label="y=0")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Training Data (Linearly Separable)")
plt.grid(True)
plt.show()

# ============================================================
# Bayesian Logistic Regression: MCMC
# ============================================================
# Prior std
sigma_w = 1.0
sigma_b = 1.0

# MCMC params
num_samples = 2000
burn_in = 500
step_size = 0.05

# Initialize parameters
w = torch.randn(2, device=device) * 0.1
b = torch.randn(1, device=device) * 0.1

samples_w = []
samples_b = []


# ============================================================
# Log prior, likelihood, posterior
# ============================================================
def log_prior(w, b):
    return -0.5 * torch.sum(w**2) / sigma_w**2 - 0.5 * torch.sum(b**2) / sigma_b**2


def log_likelihood(w, b, X, y):
    z = X @ w + b
    return torch.sum(
        y * torch.log(torch.sigmoid(z)) + (1 - y) * torch.log(1 - torch.sigmoid(z))
    )


def log_posterior(w, b, X, y):
    return log_likelihood(w, b, X, y) + log_prior(w, b)


# ============================================================
# MCMC Sampling
# ============================================================
for step in range(num_samples):
    # propose new w', b'
    w_new = w + step_size * torch.randn_like(w)
    b_new = b + step_size * torch.randn_like(b)

    log_r = log_posterior(w_new, b_new, X, y) - log_posterior(w, b, X, y)
    if torch.log(torch.rand(1, device=device)) < log_r:
        w = w_new
        b = b_new

    if step >= burn_in:
        samples_w.append(w.clone())
        samples_b.append(b.clone())

# stack samples and ensure they are on the same device
samples_w = torch.stack(samples_w).to(device)
samples_b = torch.stack(samples_b).to(device)

# ============================================================
# Visualize Prior vs Posterior for w with bias b as color
# ============================================================
# Prior samples
prior_samples_w = torch.randn(1000, 2, device=device) * sigma_w
prior_samples_b = torch.randn(1000, 1, device=device) * sigma_b

fig, ax = plt.subplots(figsize=(6, 6))

# Plot prior (weights only)
ax.scatter(
    prior_samples_w[:, 0].cpu(),
    prior_samples_w[:, 1].cpu(),
    color="blue",
    alpha=0.1,
    label="Prior w ~ N(0, σ²)",
)

# Map bias b to color for posterior samples
b_vals = samples_b.squeeze().cpu().numpy()
norm = plt.Normalize(b_vals.min(), b_vals.max())
colors = cm.viridis(norm(b_vals))  # viridis colormap

# Plot posterior w samples, colored by bias
ax.scatter(
    samples_w[:, 0].cpu(),
    samples_w[:, 1].cpu(),
    color=colors,
    alpha=0.6,
    label="Posterior w (color=bias)",
)

# Colorbar for bias
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)  # <-- attach to ax
cbar.set_label("Bias b")

ax.set_xlabel("w0")
ax.set_ylabel("w1")
ax.set_title("Prior vs Posterior for Weights w (Bias b in color)")
ax.legend()
ax.grid(True)
plt.show()

# ============================================================
# Posterior Predictive Heat Map
# ============================================================
x1_grid = np.linspace(X[:, 0].min().item() - 1, X[:, 0].max().item() + 1, 100)
x2_grid = np.linspace(X[:, 1].min().item() - 1, X[:, 1].max().item() + 1, 100)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
Z = np.zeros_like(X1)

# compute posterior predictive probabilities
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x_point = torch.tensor([X1[i, j], X2[i, j]], dtype=torch.float32, device=device)
        probs = torch.sigmoid(samples_w @ x_point + samples_b.squeeze())
        Z[i, j] = probs.mean().item()

plt.figure(figsize=(6, 6))
plt.contourf(X1, X2, Z, levels=20, cmap="RdBu_r")
plt.scatter(X[y == 1, 0].cpu(), X[y == 1, 1].cpu(), color="blue", label="y=1")
plt.scatter(X[y == 0, 0].cpu(), X[y == 0, 1].cpu(), color="red", label="y=0")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Bayesian Logistic Regression Posterior Predictive Heat Map")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# Posterior samples and decision boundaries (using actual data)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))

# Plot prior (circle for illustration)
theta0 = np.linspace(0, 2 * np.pi, 100)
radius = sigma_w  # std of prior
prior_x = radius * np.cos(theta0)
prior_y = radius * np.sin(theta0)
ax.plot(prior_x, prior_y, color="blue", linestyle="--", label="Prior w ~ N(0, σ²)")

# Plot posterior samples
posterior_x = samples_w[:, 0].cpu().numpy()
posterior_y = samples_w[:, 1].cpu().numpy()
ax.scatter(
    posterior_x, posterior_y, color="red", alpha=0.3, label="Posterior samples of w"
)

# Decision boundaries from posterior samples
x_vals = np.linspace(X[:, 0].min().item() - 1, X[:, 0].max().item() + 1, 100)
for i in range(0, len(samples_w), max(1, len(samples_w) // 50)):  # sample ~50 lines
    w0, w1 = posterior_x[i], posterior_y[i]
    b_val = samples_b[i].item()
    if w1 != 0:
        y_vals = -(w0 * x_vals + b_val) / w1
        ax.plot(x_vals, y_vals, color="black", alpha=0.1)

# Plot actual data
ax.scatter(
    X[y == 1, 0].cpu(), X[y == 1, 1].cpu(), color="blue", label="y=1", edgecolor="k"
)
ax.scatter(
    X[y == 0, 0].cpu(), X[y == 0, 1].cpu(), color="red", label="y=0", edgecolor="k"
)

# Fix axes
ax.set_xlim(X[:, 0].min().item() - 1, X[:, 0].max().item() + 1)
ax.set_ylim(X[:, 1].min().item() - 1, X[:, 1].max().item() + 1)
ax.set_aspect("equal", "box")  # make scales equal

ax.set_xlabel("x1")
ax.set_ylabel("x2 / decision boundary")
ax.set_title("Posterior Samples and Decision Boundaries (Actual Data)")
ax.legend()
ax.grid(True)
plt.show()
