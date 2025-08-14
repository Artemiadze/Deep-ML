import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int) -> np.ndarray:
    """
    Выполняет простую 2D-свёртку входной матрицы с заданным ядром, паддингом и шагом.
    
    :param input_matrix: np.ndarray формы (H, W) — исходная матрица (например, изображение в градациях серого)
    :param kernel: np.ndarray формы (kH, kW) — ядро свёртки
    :param padding: int — количество нулевых строк/столбцов, добавляемых по краям (симметрично)
    :param stride: int — шаг перемещения окна свёртки
    :return: np.ndarray формы (out_H, out_W) — результат свёртки
    """
    
    # Получаем размеры входа и ядра
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # === 1. Паддинг (добавляем вокруг входа нули) ===
    # np.pad добавляет "padding" нулей сверху/снизу и слева/справа
    padded_input = np.pad(
        input_matrix,
        pad_width=((padding, padding), (padding, padding)),  # ((верх, низ), (лево, право))
        mode='constant',
        constant_values=0
    )

    # === 2. Вычисляем размеры выходной матрицы ===
    # Формулы для вычисления размера:
    # out_H = ((H_padded - kH) / stride) + 1
    # out_W = ((W_padded - kW) / stride) + 1
    out_height = ((padded_input.shape[0] - kernel_height) // stride) + 1
    out_width = ((padded_input.shape[1] - kernel_width) // stride) + 1

    # === 3. Создаём матрицу для результата ===
    output_matrix = np.zeros((out_height, out_width), dtype=float)

    # === 4. Основной цикл по выходным координатам ===
    for y in range(out_height):
        for x in range(out_width):
            # Определяем координаты верхнего левого угла текущего окна
            start_y = y * stride
            start_x = x * stride

            # Извлекаем подматрицу из padded_input того же размера, что и kernel
            region = padded_input[start_y:start_y + kernel_height,
                                  start_x:start_x + kernel_width]

            # Скалярное произведение: умножаем поэлементно и суммируем
            conv_value = np.sum(region * kernel)

            # Записываем результат в соответствующую ячейку
            output_matrix[y, x] = conv_value

    return output_matrix

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)