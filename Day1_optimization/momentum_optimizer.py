def momentum_optimizer(X, y, lr = 0.005, epochs = 300, beta = 0.9):
    m = 0
    b = 0
    v_m = 0
    v_b = 0
    n = len(X)
    loss_history = []
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
        v_m = beta * v_m + lr * gradient_m
        v_b = beta * v_b + lr * gradient_b

        m -= v_m
        b -= v_b

        avg_loss = (total_error)/n
        loss_history.append(avg_loss)
    
    return m, b, loss_history
