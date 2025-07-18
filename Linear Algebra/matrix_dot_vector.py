def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    
    c = [0] * len(a)
    for i in range(len(a)):
        sum = 0
        for j in range(len(a[i])):
            sum += (a[i][j] * b[j])
        c[i] = sum
        sum = 0
    
    return c

print(matrix_dot_vector([[1, 2, 3], [2, 4, 5], [6, 8, 9]], [1, 2, 3]))
print(matrix_dot_vector([[1, 2], [2, 4], [6, 8], [12, 4]], [1, 2, 3]))
print(matrix_dot_vector([[1.5, 2.5], [3.0, 4.0]], [2, 1]))