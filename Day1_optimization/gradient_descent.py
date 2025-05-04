# gradient_descent.py
import numpy as np

def gradient_descent(X, y, lr=0.005, epochs=300):
    m = 0
    b = 0
    n = len(X)
    loss_history = []

    for epoch in range(epochs):
        total_error = 0
        gradient_m = 0
        gradient_b = 0

        for i in range(n):
            x_i = X[i]
            y_i = y[i]
            y_pred = m * x_i + b
            error = y_pred - y_i
            total_error += error ** 2
            gradient_m += (2/n) * error * x_i
            gradient_b += (2/n) * error

        m -= lr * gradient_m
        b -= lr * gradient_b

        avg_loss = total_error / n
        loss_history.append(avg_loss)

        if epoch > 5 and abs(loss_history[-1] - loss_history[-2]) < 0.0001:
            break

    return m, b, loss_history
