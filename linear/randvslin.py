import matplotlib.pyplot as plt
import torch

# ---- Evenly spaced X ----
x_lin = torch.linspace(-10, 10, 100).unsqueeze(1)
y_line = 2 * x_lin - 1

# ---- Random X from normal distribution ----
x_rand = torch.randn(100, 1)
y_rand = 2 * x_rand - 1

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(x_lin, y_line, s=10)
plt.title("Evenly spaced x (linspace)")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.scatter(x_rand, y_rand, s=10)
plt.title("Random x (normal distribution)")
plt.xlabel("x")

plt.tight_layout()
plt.show()
