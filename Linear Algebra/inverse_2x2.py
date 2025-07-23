def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b = matrix[0]
    c, d = matrix[1]
    det = a * d - b * c
    return [
        [ d * (1/det), -b * (1/det)],
        [-c * (1/det),  a * (1/det)],
    ] if det != 0 else None

print(inverse_2x2([[4, 7], [2, 6]]))
print(inverse_2x2([[2, 1], [6, 2]]))