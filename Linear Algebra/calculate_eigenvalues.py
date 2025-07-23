import math

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    a, b = matrix[0]
    c, d = matrix[1]
    
    trace = a + d
    det = a * d - b * c
    
    discriminant = trace**2 - 4 * det
    sqrt_discriminant = math.sqrt(discriminant)
    
    lambda1 = (trace + sqrt_discriminant) / 2
    lambda2 = (trace - sqrt_discriminant) / 2
    
    return sorted([lambda1, lambda2], reverse=True)

print(calculate_eigenvalues([[2, 1], [1, 2]]))
print(calculate_eigenvalues([[4, -2], [1, 1]]))