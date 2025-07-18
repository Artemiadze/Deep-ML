def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    result = []
    if mode == 'column':
        for j in range(len(matrix[0])):
            sum_col = 0
            for i in range(len(matrix)):
                sum_col += matrix[i][j]
            result.append(sum_col / len(matrix))
    else:
        for i in range(len(matrix)):
            result.append(sum(matrix[i])/len(matrix))
    return result

print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'column'))
print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'row'))