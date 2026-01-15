import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Create a conceptual prior (circle around origin)
# ============================================================
theta0 = np.linspace(0, 2 * np.pi, 100)
radius = 1.0
prior_x = radius * np.cos(theta0)
prior_y = radius * np.sin(theta0)

# ============================================================
# Conceptual likelihood (elongated along direction data pulls)
# ============================================================
# Let's say the data wants bigger w0 and smaller w1
likelihood_x = np.linspace(-1.5, 2.5, 100)
likelihood_y_upper = 0.5 * likelihood_x + 1.0  # top boundary
likelihood_y_lower = 0.5 * likelihood_x - 1.0  # bottom boundary

# ============================================================
# Conceptual posterior (overlap of prior and likelihood)
# ============================================================
posterior_center = [1.0, 0.3]
posterior_cov = [[0.2, 0.05], [0.05, 0.1]]
posterior_x = np.random.multivariate_normal(posterior_center, posterior_cov, 200)
posterior_y = posterior_x[:, 1]
posterior_x = posterior_x[:, 0]

# ============================================================
# Plot the diagram
# ============================================================
plt.figure(figsize=(7, 7))

# Prior
plt.plot(prior_x, prior_y, color="blue", linestyle="--", label="Prior w ~ N(0,σ²)")

# Likelihood (conceptual area)
plt.fill_between(
    likelihood_x,
    likelihood_y_lower,
    likelihood_y_upper,
    color="green",
    alpha=0.2,
    label="Likelihood from data",
)

# Posterior
plt.scatter(posterior_x, posterior_y, color="red", alpha=0.5, label="Posterior samples")

# Decision boundaries example
x_vals = np.linspace(-2, 2, 50)
for i in range(0, 50, 5):
    # simple linear boundaries from posterior samples
    w0, w1 = posterior_x[i], posterior_y[i]
    if w1 != 0:
        y_vals = -(w0 * x_vals) / w1
        plt.plot(x_vals, y_vals, color="black", alpha=0.1)

plt.xlabel("w0")
plt.ylabel("w1")
plt.title("Conceptual Prior, Likelihood, Posterior and Decision Boundaries")
plt.legend()
plt.grid(True)
plt.xlim(-2, 3)
plt.ylim(-2, 2)
plt.show()
