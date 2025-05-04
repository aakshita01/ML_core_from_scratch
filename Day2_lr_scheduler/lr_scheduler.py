import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------
# STEP 1: Load and preprocess California Housing
# ---------------------------

data = fetch_california_housing()
X, y = data.data, data.target

X = X[:, [0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# ---------------------------
# STEP 2: Define Learning Rate Schedulers
# ---------------------------

def step_decay(epoch, initial_lr = 0.01, drop = 0.5, epoch_drop = 50):
    return initial_lr * (drop ** (epoch // epoch_drop))

def exp_decay(epoch, initial_lr = 0.01, decay_rate = 0.05):
    return initial_lr * np.exp(-decay_rate * epoch)

# ---------------------------
# STEP 3: Gradient Descent with Scheduler
# ---------------------------

def gradient_descent (X, y, lr = 0.01, epochs = 300, scheduler = None):
    m = 0
    b = 0
    n = len(X)
    loss_history = []
    lr_history = []
    for epoch in range (epochs):
        gradient_m = 0
        gradient_b = 0
        total_error = 0
        for i in range(n):
            x_i = X[i]
            y_i = y[i]

            y_pred = m * x_i + b
            error = y_pred - y_i
            total_error += error ** 2

            gradient_m += (2/n) * error * x_i
            gradient_b += (2/n) * error
        if scheduler:
            lr = scheduler(epoch)
        
        m -= lr * gradient_m
        b -= lr * gradient_b

        avg_loss = total_error / n
        loss_history.append(avg_loss)
        lr_history.append(lr)

    return m, b, loss_history, lr_history

# ---------------------------
# STEP 4: Run with different strategies
# ---------------------------

m1, b1, loss_fixed, _ = gradient_descent(X_train, y_train, lr = 0.01)
m2, b2, loss_step, _ = gradient_descent(X_train, y_train, scheduler = lambda e : step_decay(e))
m3, b3, loss_exp, _ = gradient_descent(X_train, y_train, scheduler = lambda e : exp_decay(e))

# ---------------------------
# STEP 5: Plot Loss Curves
# ---------------------------

plt.figure(figsize=(10,6))
plt.plot(loss_fixed, label = 'fixed LR')
plt.plot(loss_step, label = 'step decay')
plt.plot(loss_exp, label = 'exp decay')
plt.xlabel('Epochs')
plt.ylabel('MSE loss')
plt.title("Loss Curve Comparison - Learning Rate Strategies")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()        
