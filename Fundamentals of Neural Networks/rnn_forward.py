import numpy as np

def rnn_forward(input_sequence: list[list[float]], 
                initial_hidden_state: list[float], 
                Wx: list[list[float]], 
                Wh: list[list[float]], 
                b: list[float]) -> list[float]:
    """
    Реализация простейшей RNN-ячейки для обработки последовательности входных векторов.

    Параметры:
    ----------
    input_sequence : list[list[float]]
        Последовательность входных векторов (каждый элемент - список признаков для одного шага времени)
    initial_hidden_state : list[float]
        Начальное скрытое состояние (h0)
    Wx : list[list[float]]
        Весовая матрица вход-в-состояние (input-to-hidden)
    Wh : list[list[float]]
        Весовая матрица состояние-в-состояние (hidden-to-hidden)
    b : list[float]
        Вектор смещения (bias)

    Возвращает:
    -----------
    list[float]
        Финальное скрытое состояние после обработки всей последовательности,
        округлённое до 4-х знаков после запятой.
    """

    # Преобразуем входы в numpy-массивы для удобства матричных операций
    h = np.array(initial_hidden_state, dtype=float)  # Скрытое состояние
    Wx = np.array(Wx, dtype=float)                   # Весовая матрица input-to-hidden
    Wh = np.array(Wh, dtype=float)                   # Весовая матрица hidden-to-hidden
    b = np.array(b, dtype=float)                     # Вектор смещения

    # Перебираем каждый входной вектор последовательности
    for x_t in input_sequence:
        # Преобразуем текущий вход в numpy-вектор
        x_t = np.array(x_t, dtype=float)

        # Линейная комбинация входа и предыдущего состояния
        # h' = Wx * x_t + Wh * h + b
        linear_output = np.dot(Wx, x_t) + np.dot(Wh, h) + b

        # Применяем функцию активации tanh для обновления состояния
        # h_t = tanh(linear_output)
        h = np.tanh(linear_output)

    # Округляем результат до 4-х знаков и конвертируем в обычный список
    final_hidden_state = np.round(h, 4).tolist()

    return final_hidden_state

print(rnn_forward([[1.0], [2.0], [3.0]], [0.0], [[0.5]], [[0.8]], [0.0]))
print(rnn_forward([[0.5], [0.1], [-0.2]], [0.0], [[1.0]], [[0.5]], [0.1]))
print(rnn_forward( [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [0.0, 0.0], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0]], [0.1, 0.2] ))