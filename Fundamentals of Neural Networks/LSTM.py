import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        """
        Конструктор LSTM.
        :param input_size: размерность входного вектора (кол-во признаков на timestep)
        :param hidden_size: размерность скрытого состояния h_t
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Инициализация весов для забывающего, входного, кандидатного и выходного блоков
        # Размер матрицы весов: (hidden_size, input_size + hidden_size), так как вход объединяется с предыдущим hidden
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)  # forget gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)  # input gate
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)  # candidate cell
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)  # output gate

        # Инициализация векторов смещений (bias)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Прямое распространение LSTM по входной последовательности.
        :param x: np.array формы (seq_len, input_size)
        :param initial_hidden_state: np.array формы (hidden_size, 1)
        :param initial_cell_state: np.array формы (hidden_size, 1)
        :return: (все скрытые состояния, финальное скрытое состояние, финальное состояние ячейки)
        """
        seq_len = x.shape[0]

        # Копируем, чтобы не трогать входные массивы извне
        h_t = initial_hidden_state.copy()
        c_t = initial_cell_state.copy()

        hidden_states = []

        for t in range(seq_len):
            # x_t: (input_size, 1)
            x_t = x[t].reshape(-1, 1)

            # ВАЖНО: порядок именно [h_{t-1}; x_t], т.к. так организованы веса в тесте
            combined = np.vstack((h_t, x_t))  # форма: (hidden_size + input_size, 1)

            # Gates
            f_t = self.sigmoid(self.Wf @ combined + self.bf)          # forget gate
            i_t = self.sigmoid(self.Wi @ combined + self.bi)          # input gate
            c_hat_t = np.tanh(self.Wc @ combined + self.bc)           # candidate cell
            c_t = f_t * c_t + i_t * c_hat_t                           # new cell
            o_t = self.sigmoid(self.Wo @ combined + self.bo)          # output gate
            h_t = o_t * np.tanh(c_t)                                  # new hidden

            hidden_states.append(h_t.copy())

        # Собираем скрытые состояния: (seq_len, hidden_size, 1)
        hidden_states = np.stack(hidden_states, axis=0)

        return hidden_states, h_t, c_t

print("Test 1")
input_sequence = np.array([[1.0], [2.0], [3.0]]) 
initial_hidden_state = np.zeros((1, 1)) 
initial_cell_state = np.zeros((1, 1)) 
lstm = LSTM(input_size=1, hidden_size=1) # Set weights and biases for reproducibility 
lstm.Wf = np.array([[0.5, 0.5]]) 
lstm.Wi = np.array([[0.5, 0.5]]) 
lstm.Wc = np.array([[0.3, 0.3]]) 
lstm.Wo = np.array([[0.5, 0.5]]) 
lstm.bf = np.array([[0.1]]) 
lstm.bi = np.array([[0.1]]) 
lstm.bc = np.array([[0.1]]) 
lstm.bo = np.array([[0.1]]) 
outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state) 
print(final_h)


print("Test 2")
input_sequence = np.array([[0.1, 0.2], [0.3, 0.4]]) 
initial_hidden_state = np.zeros((2, 1)) 
initial_cell_state = np.zeros((2, 1)) 
lstm = LSTM(input_size=2, hidden_size=2) # Set weights and biases for reproducibility 
lstm.Wf = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wi = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wc = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.Wo = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]) 
lstm.bf = np.array([[0.1], [0.2]]) 
lstm.bi = np.array([[0.1], [0.2]]) 
lstm.bc = np.array([[0.1], [0.2]]) 
lstm.bo = np.array([[0.1], [0.2]]) 
outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state) 
print(final_h)