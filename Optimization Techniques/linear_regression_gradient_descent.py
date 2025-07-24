import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape((m, 1))                 # преобразуем y в столбец

    for _ in range(iterations):
        predictions = X @ theta          # предсказания модели
        errors = predictions - y         # ошибки предсказаний
        gradient = (1 / m) * (X.T @ errors)  # градиент
        theta -= alpha * gradient        # обновление коэффициентов

    return np.round(theta.flatten(), 4)  # преобразуем в 1D-массив и округляем

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
alpha = 0.01
iterations = 1000
print(linear_regression_gradient_descent(X, y, alpha, iterations))      # [0.1107 0.9513]