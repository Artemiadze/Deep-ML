import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Инициализация RNN.
        :param input_size: размер входного вектора на шаге времени
        :param hidden_size: размер скрытого состояния
        :param output_size: размер выходного вектора
        Все веса инициализируются малыми случайными значениями, смещения -- нулями.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Веса для преобразования входа -> скрытое состояние
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01  # (H, D)
        # Веса для перехода скрытого состояния -> скрытое состояние
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # (H, H)
        # Веса для преобразования скрытого состояния -> выход
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01  # (O, H)

        # Смещения
        self.b_h = np.zeros((hidden_size, 1))  # (H, 1)
        self.b_y = np.zeros((output_size, 1))  # (O, 1)

    def forward(self, x):
        """
        Прямой проход по всей последовательности.
        :param x: np.array формы (T, input_size) или (T, input_size, 1)
        :return: outputs: np.array формы (T, output_size)
                 Также сохраняет внутренние состояния для BPTT: self.last_inputs, self.last_hs
        """
        # Приводим вход к форме (T, input_size)
        x = np.array(x, dtype=float)
        if x.ndim == 3 and x.shape[2] == 1:
            x = x.reshape(x.shape[0], x.shape[1])

        T = x.shape[0]

        # Подготавливаем контейнеры
        hs = [np.zeros((self.hidden_size, 1))]  # hs[0] = h_{-1} (нулевое начальное состояние)
        ys = []  # список выходов (по шагам)

        # Проходим по временным шагам
        for t in range(T):
            # x_t в виде колонки (D, 1)
            x_t = x[t].reshape(-1, 1)

            # Новое скрытое состояние: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            preact = self.W_xh @ x_t + self.W_hh @ hs[-1] + self.b_h  # (H,1)
            h_t = np.tanh(preact)  # (H,1)
            hs.append(h_t)

            # Линейный выход: y_t = W_hy * h_t + b_y
            y_t = self.W_hy @ h_t + self.b_y  # (O,1)
            ys.append(y_t)

        # Сохраняем входы и скрытые состояния для backward (BPTT)
        self.last_inputs = x  # (T, D)
        self.last_hs = hs     # список длины T+1, каждый (H,1)

        # Возвращаем outputs в форме (T, output_size)
        outputs = np.hstack([y.reshape(-1, 1) for y in ys]).T  # (T, O)
        return outputs

    def backward(self, x, y, learning_rate=0.01):
        """
        BPTT: вычисление градиентов и обновление весов.
        Используется суммарная потеря L = sum_t 0.5 * (y_t - yhat_t)^2.
        :param x: входная последовательность (T, input_size)
        :param y: целевая последовательность (T, output_size)
        :param learning_rate: шаг градиентного спуска
        """
        # Приведение форм
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if x.ndim == 3 and x.shape[2] == 1:
            x = x.reshape(x.shape[0], x.shape[1])
        if y.ndim == 3 and y.shape[2] == 1:
            y = y.reshape(y.shape[0], y.shape[1])

        T = x.shape[0]

        # Выполним forward (чтобы убедиться, что last_inputs/last_hs соответствуют текущему x)
        outputs = self.forward(x)  # (T, O)

        # Инициализируем градиенты нулями
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # dnext_h - градиент по следующему скрытому состоянию (h_{t} для предыдущего шага) при развороте
        dnext_h = np.zeros((self.hidden_size, 1))

        # Проходим по временным шагам в обратном порядке
        for t in reversed(range(T)):
            # Градиент по выходу на шаге t
            y_pred = outputs[t].reshape(-1, 1)   # (O,1)
            y_true = y[t].reshape(-1, 1)         # (O,1)

            # dL/dy_pred, где L_t = 0.5 * (y_pred - y_true)^2 -> derivative = (y_pred - y_true)
            dy = (y_pred - y_true)  # (O,1)

            # Градиенты для W_hy и b_y
            h_t = self.last_hs[t+1]  # скрытое состояние в момент t (H,1)
            dW_hy += dy @ h_t.T      # (O,H)
            db_y += dy               # (O,1)

            # Распространение градиента в скрытую область
            # dh = W_hy^T * dy + dnext_h
            dh = self.W_hy.T @ dy + dnext_h   # (H,1)

            # backprop через tanh: dh_raw = (1 - h_t^2) * dh
            dh_raw = (1 - h_t * h_t) * dh     # (H,1)

            # Градиенты по параметрам, влияющим на dh_raw
            x_t = x[t].reshape(-1, 1)         # (D,1)
            h_prev = self.last_hs[t]          # h_{t-1} (H,1)

            dW_xh += dh_raw @ x_t.T           # (H,D)
            dW_hh += dh_raw @ h_prev.T        # (H,H)
            db_h += dh_raw                     # (H,1)

            # Градиент для предыдущего скрытого состояния (для следующей итерации обратного хода)
            dnext_h = self.W_hh.T @ dh_raw    # (H,1)

        # Обновляем параметры градиентным спуском (в одну "мини-обновляющую" операцию)
        # Здесь градиенты суммированы по временной оси (мы не усредняем по T),
        # т.к. в постановке говорится суммировать потери по шагам.
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

        # Возвращаем суммарную потерю для контроля (опционально)
        total_loss = 0.5 * np.sum((outputs - y) ** 2)
        return total_loss, outputs

print("SimpleRNN example 1:")
input_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])
expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])
# Initialize RNN
rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)
    
# Forward pass
output = rnn.forward(input_sequence)
    
# Backward pass
rnn.backward(input_sequence, expected_output, learning_rate=0.01)
    
print(output)

print("\nSimpleRNN example 2:")
np.random.seed(42) 
input_sequence = np.array([[1.0,2.0], [7.0,2.0], [1.0,3.0], [12.0,4.0]]) 
expected_output = np.array([[2.0], [3.0], [4.0], [5.0]]) 
rnn = SimpleRNN(input_size=2, hidden_size=3, output_size=1) # Train the RNN over multiple epochs 
for epoch in range(100): 
    output = rnn.forward(input_sequence) 
    rnn.backward(input_sequence, expected_output, learning_rate=0.01) 
print(output)