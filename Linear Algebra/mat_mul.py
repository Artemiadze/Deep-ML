def mat_mul(a: list[list[int | float]],
              b: list[list[int | float]]) -> list[list[int | float]]:
    rows_a = len(a)
    cols_a = len(a[0]) if rows_a > 0 else 0

    rows_b = len(b)
    cols_b = len(b[0]) if rows_b > 0 else 0

    if cols_a != rows_b:
        return -1
    
    c = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            sum = 0
            for k in range(cols_a):
                sum += a[i][k] * b[k][j]
            c[i][j] = sum
    return c

print(mat_mul([[1,2,3],[2,3,4],[5,6,7]],[[3,2,1],[4,3,2],[5,4,3]]))
print(mat_mul([[0,0],[2,4],[1,2]],[[0,0],[2,4]]))
print(mat_mul([[0,0],[2,4],[1,2]],[[0,0,1],[2,4,1],[1,2,3]]))