import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import gradient_descent
from momentum_optimizer import momentum_optimizer
from adagrad_optimizer import adagrad_optimizer
from rmsprop_optimizer import rmsprop_optimizer
from adam_optimizer import adam_optimizer

# 1. Generate synthetic data
np.random.seed(42)
X = np.random.rand(100) * 10
y = 2 * X + 3 + np.random.randn(100)
X = (X - np.mean(X)) / np.std(X)

# 2. Run each optimizer
_, _, gd_loss = gradient_descent(X, y)
_, _, momentum_loss = momentum_optimizer(X, y)
_, _, adagrad_loss = adagrad_optimizer(X, y)
_, _, rmsprop_loss = rmsprop_optimizer(X, y)
_, _, adam_loss = adam_optimizer(X, y)

# 3. Plot all losses together
plt.figure(figsize=(10, 6))
plt.plot(gd_loss, label="Gradient Descent")
plt.plot(momentum_loss, label="Momentum")
plt.plot(adagrad_loss, label="Adagrad")
plt.plot(rmsprop_loss, label="RMSProp")
plt.plot(adam_loss, label="Adam")
plt.title("Loss Curve Comparison of Optimizers")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

