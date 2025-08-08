import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    m = len(y)  # число примеров

    for iteration in range(n_iterations):
        if method == 'batch':
            # Прогноз
            predictions = X @ weights
            # Градиент для всех данных
            gradient = (2 / m) * X.T @ (predictions - y)
            # Обновление весов
            weights -= learning_rate * gradient

        elif method == 'stochastic':
            for i in range(m):
                xi = X[i:i+1]
                yi = y[i:i+1]
                prediction = xi @ weights
                gradient = 2 * xi.T @ (prediction - yi)
                weights -= learning_rate * gradient

        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                predictions = X_batch @ weights
                gradient = (2 / len(y_batch)) * X_batch.T @ (predictions - y_batch)
                weights -= learning_rate * gradient

        else:
            raise ValueError("Invalid method. Use 'batch', 'stochastic', or 'mini_batch'.")

    return weights

# Example usage
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]) 
y = np.array([2, 3, 4, 5]) 
weights = np.zeros(X.shape[1]) 
learning_rate = 0.01 
n_iterations = 100
output = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')
print(output)

# Example usage 2
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]) 
y = np.array([2, 3, 4, 5]) 
weights = np.zeros(X.shape[1]) 
learning_rate = 0.01 
n_iterations = 100 
output = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic') 
print(output)

# Example usage 3
X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]) 
y = np.array([2, 3, 4, 5]) 
weights = np.zeros(X.shape[1]) 
learning_rate = 0.01 
n_iterations = 100 
batch_size = 2 
output = gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size, method='mini_batch') 
print(output)