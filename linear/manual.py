import matplotlib.pyplot as plt
import torch
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

W = torch.randn(1, 1, device=device, requires_grad=True)
b = torch.randn(1, device=device, requires_grad=True)

for epoch in range(epochs):
    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true) ** 2)
    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    if epoch % 1000 == 0:
        print(f"Loss: {loss}, W: {W.item()}, b: {b.item()}")

    W.grad.zero_()
    b.grad.zero_()


# ------------------------------
# Display results
# ------------------------------

x_line = torch.linspace(X.min(), X.max(), 100)
y_line = W.item() * x_line + b.item()
plt.plot(x_line.cpu().numpy(), y_line.cpu().numpy(), color="red", label="Learned Line")
plt.scatter(X.cpu().numpy(), y_true.cpu().numpy(), s=10, label="Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Linear Data")
plt.show()
