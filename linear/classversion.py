import matplotlib.pyplot as plt
import torch
imoport torch.nn as nn
from rich import print

# ------------------------------
# Mac silicon
# ------------------------------
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print("Using device:", device)

# ------------------------------
# Original Data
# ------------------------------
true_W = torch.tensor([[2.0]], device=device)
true_b = torch.tensor([-1.0], device=device)

X = torch.randn(100, 1, device=device)
noise = torch.rand_like(X) * 3

y_true = X @ true_W + true_b + noise

# ------------------------------
# Learning Steps
# ------------------------------
learning_rate = 0.01
epochs = 10000

linear_layer = torch.nn.Linear(in_features=1, out_features=1, device=device)
optimizer = torch.optim.SGD(linear_layer.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

for epoch in range(epochs):
    optimizer.zero_grad()
    y_hat = linear_layer(X)
    loss = loss_fn(y_hat, y_true)

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Loss: {loss}, W: {linear_layer.weight}, b: {linear_layer.bias}")

# ------------------------------
# Display results
# ------------------------------

x_line = torch.linspace(X.min(), X.max(), 100, device=device).unsqueeze(1)
y_line = linear_layer.weight * x_line + linear_layer.bias
# y_line = linear_layer(X)

plt.plot(
    x_line.cpu().detach().numpy(),
    y_line.cpu().detach().numpy(),
    label="Learned Line",
    color="red",
)
plt.scatter(X.cpu().numpy(), y_true.cpu().numpy(), s=10, label="Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Linear Data")
plt.show()
