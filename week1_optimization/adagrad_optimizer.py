import numpy as np
def adagrad_optimizer(X, y, lr = 0.005, epochs = 300, epsilon = 1e-8):
    m = 0
    b = 0
    n = len(X)
    loss_history = []
    grad_squared_m = 0
    grad_squared_b = 0
    for epoch in range(epochs):
        total_error = 0
        gradient_m = 0
        gradient_b  = 0

        for i in range(n):
            x_i = X[i]
            y_i = y[i]

            y_pred = m * x_i + b
            error = y_pred - y_i

            total_error += error**2

            gradient_m += (2/n) * error * x_i
            gradient_b += (2/n) * error
        grad_squared_m += gradient_m ** 2
        grad_squared_b += gradient_b ** 2

        m -= (lr / (np.sqrt(grad_squared_m) + epsilon)) * gradient_m
        b -= (lr / (np.sqrt(grad_squared_b) + epsilon)) * gradient_b

        avg_loss = (total_error)/n
        loss_history.append(avg_loss)
    
    return m, b, loss_history
