import matplotlib.pyplot as plt
import torch
from rich import print

m_true = 2.0
b_true = -1.0

x = torch.linspace(-10, 10, 100).unsqueeze(1)
noise = torch.rand_like(x) * 5

y_true = m_true * x + b_true
y_noise = m_true * x + b_true + noise

print(x.shape, y_true.shape)
exit()

plt.scatter(x.numpy(), y_true.numpy(), s=10)
plt.scatter(x.numpy(), y_noise.numpy(), s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Linear Data")
plt.show()
