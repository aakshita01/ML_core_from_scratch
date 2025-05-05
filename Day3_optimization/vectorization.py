import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------------------
# STEP 1: Load and preprocess California Housing
# ---------------------------

data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias column
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

# ---------------------------
# STEP 2: Initialize parameters
# ---------------------------

n_samples, n_features = X_train.shape
weights = np.zeros(n_features)
epochs = 300
lr = 0.01

# ---------------------------
# STEP 3: Vectorized gradient descent 
# ---------------------------

loss_history = []
for epoch in range(epochs):
    y_pred = X_train @ weights 
    error = y_pred - y_train
    loss = np.mean(error ** 2)
    gradient = (2/n_samples) * (X_train.T @ error)
    weights -= lr * gradient
    loss_history.append(loss)

# ---------------------------
# STEP 4: Plot loss curve 
# ---------------------------

plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title("Loss Curve - Vectorized Gradient Descent")
plt.grid(True)
plt.show()

# ---------------------------
# STEP 5: Comparison with scikit-learn 
# ---------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("Sklearn Coefficients:\n", model.coef_)
print("Our Coefficients:\n", weights[:-1])  # exclude bias term
print("Sklearn Intercept:", model.intercept_)
print("Our Bias Term:", weights[-1])