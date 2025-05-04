import numpy as np
def adam_optimizer(X, y, lr=0.01, epochs=300, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_w = 0  # momentum for slope
    m_b = 0
    v_w = 0  # RMS for slope
    v_b = 0
    w = 0
    b = 0
    n = len(X)
    loss_history = []

    for epoch in range(epochs):
        grad_w = 0
        grad_b = 0
        total_error = 0

        for i in range(n):
            x_i = X[i]
            y_i = y[i]
            y_pred = w * x_i + b
            error = y_pred - y_i
            total_error += error ** 2

            grad_w += (2/n) * error * x_i
            grad_b += (2/n) * error

        # Update biased moment estimates
        m_w = beta1 * m_w + (1 - beta1) * grad_w
        m_b = beta1 * m_b + (1 - beta1) * grad_b

        v_w = beta2 * v_w + (1 - beta2) * (grad_w**2)
        v_b = beta2 * v_b + (1 - beta2) * (grad_b**2)

        # Bias correction
        m_w_hat = m_w / (1 - beta1 ** (epoch + 1))
        m_b_hat = m_b / (1 - beta1 ** (epoch + 1))
        v_w_hat = v_w / (1 - beta2 ** (epoch + 1))
        v_b_hat = v_b / (1 - beta2 ** (epoch + 1))

        w -= lr * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        avg_loss = total_error / n
        loss_history.append(avg_loss)

    return w, b, loss_history
