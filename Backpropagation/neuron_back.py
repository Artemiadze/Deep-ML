import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray,
                 initial_weights: np.ndarray, initial_bias: float,
                 learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    """
    Обучает один нейрон с сигмоидной активацией и backpropagation.
    Использует градиентный спуск по MSE.
    """

    # Преобразуем входные данные в numpy-массивы (на случай, если переданы списки)
    X = np.array(features, dtype=float)  # (n_samples, n_features)
    y = np.array(labels, dtype=float)    # (n_samples,)
    
    # Инициализация весов и смещения
    w = np.array(initial_weights, dtype=float)  # (n_features,)
    b = float(initial_bias)
    
    # Список для хранения значений MSE по эпохам
    mse_values = []

    # Определение сигмоиды
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Цикл по эпохам обучения
    for _ in range(epochs):
        # ======== 1. Прямой проход ========
        # Линейная комбинация входов и весов
        z = np.dot(X, w) + b  # (n_samples,)
        # Применяем сигмоидную активацию
        y_pred = sigmoid(z)   # (n_samples,)
        
        # ======== 2. Вычисление ошибки ========
        errors = y_pred - y
        mse = np.mean(errors ** 2)  # MSE
        mse_values.append(round(mse, 4))
        
        # ======== 3. Обратное распространение (градиенты) ========
        # Производная MSE по y_pred: dL/dy_pred = 2 * (y_pred - y) / N
        dL_dy_pred = (2 / len(y)) * errors
        # Производная y_pred по z для сигмоиды: dy_pred/dz = y_pred * (1 - y_pred)
        dy_pred_dz = y_pred * (1 - y_pred)
        # Градиент по z: dL/dz = dL/dy_pred * dy_pred/dz
        dL_dz = dL_dy_pred * dy_pred_dz
        
        # Градиент по весам: dL/dw = X^T * dL_dz
        grad_w = np.dot(X.T, dL_dz)  # (n_features,)
        # Градиент по смещению: dL/db = сумма dL_dz по всем сэмплам
        grad_b = np.sum(dL_dz)
        
        # ======== 4. Обновление параметров ========
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    # Округляем финальные веса и bias до 4 знаков
    updated_weights = np.round(w, 4)
    updated_bias = round(b, 4)
    
    return updated_weights, updated_bias, mse_values

print(train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), np.array([1, 0, 0]), np.array([0.1, -0.2]), 0.0, 0.1, 2))
print(train_neuron(np.array([[1, 2], [2, 3], [3, 1]]), np.array([1, 0, 1]), np.array([0.5, -0.2]), 0, 0.1, 3))